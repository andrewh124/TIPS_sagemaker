import argparse
import json
import os
import sys
from functools import partial

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from loguru import logger
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import Dataset
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    logging,
    set_seed,
)

logging.set_verbosity_error()
set_seed(42)


class IdeaDetectionDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        id2label_path: str,
        model_name: str,
        n_rows_limit: int = None,
    ):
        logger.info(f"Loading dataset from {data_path}")
        logger.info(f"Using id2label mapping from {id2label_path}")
        if n_rows_limit is not None:
            logger.warning(f"Limiting dataset to {n_rows_limit} rows for debugging")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.n_rows_limit = n_rows_limit  # for debugging
        self.index2tag_mapping = json.load(open(id2label_path))
        self.tag2index_mapping = {v: int(k) for k, v in self.index2tag_mapping.items()}
        self.n_labels = len(self.tag2index_mapping)
        (
            self.all_response_doc_id,
            self.all_response_tokens,
            self.all_response_tags,
        ) = self.load_dataset_from_path(data_path)

    def load_dataset_from_path(self, path: str) -> tuple[list]:
        df = pd.read_csv(path)
        (
            all_response_doc_id,
            all_response_tokens,
            all_response_tags,
        ) = self._process_dataset_into_samples(df)

        assert len(all_response_doc_id) == len(all_response_tokens)
        assert len(all_response_doc_id) == len(all_response_tags)

        return (
            all_response_doc_id,
            all_response_tokens,
            all_response_tags,
        )

    def _process_dataset_into_samples(self, df: pd.DataFrame) -> tuple[list]:
        all_response_doc_ids = []
        all_response_tokens = []
        all_response_labels = []

        response_doc_id = None
        response_tokens = []
        response_labels = []

        for i_row, row in df.iterrows():
            if self.n_rows_limit is not None and i_row >= self.n_rows_limit:
                break

            try:
                if response_doc_id != row["sample_id"]:
                    if response_doc_id is not None:
                        all_response_doc_ids.append(response_doc_id)
                        all_response_tokens.append(response_tokens)
                        all_response_labels.append(response_labels)

                    response_doc_id = row["sample_id"]
                    response_tokens = [row["token"]]
                    label = self._convert_label_to_index(row["span_label"])
                    response_labels = [label]
                else:
                    response_tokens.append(row["token"])
                    label = self._convert_label_to_index(row["span_label"])
                    response_labels.append(label)
            except BaseException as e:
                logger.error(f"KeyError: {e} at row {i_row} in {response_doc_id}")
                sys.exit(1)

        return (
            all_response_doc_ids,
            all_response_tokens,
            all_response_labels,
        )

    def _convert_label_to_index(self, span_label: str) -> list[int]:
        labels = span_label.split("|")
        indices = [
            self.tag2index_mapping[label] if label in self.tag2index_mapping else None
            for label in labels
        ]
        indices = list(set([label for label in indices if label is not None]))

        return indices

    def __len__(self):
        return len(self.all_response_doc_id)

    def __getitem__(self, idx) -> tuple:
        return (
            self.all_response_doc_id[idx],
            self.all_response_tokens[idx],
            self.all_response_tags[idx],
        )

    def collate_fn(self, batch: tuple) -> dict:
        (all_response_doc_id, all_response_tokens, all_response_tags) = zip(*batch)

        tokenized_inputs = self.tokenizer(
            all_response_tokens,
            is_split_into_words=True,
            truncation=True,
            padding="longest",
            return_attention_mask=True,
            return_tensors="pt",
            max_length=64,
        )

        # Sometimes the tokens are split into subtokens, we re-align the tags and token_ids here
        # All subtokens of a token share the same tag
        # based on tokenize_and_align_labels() in https://huggingface.co/docs/transformers/tasks/token_classification
        labels = []
        # Empty label vector for un-tagged special tokens

        for i_sample, sample_response_tags in enumerate(all_response_tags):
            sample_word_ids = tokenized_inputs.word_ids(batch_index=i_sample)
            label_ids = []
            for word_idx in sample_word_ids:
                if word_idx is None:  # Special token
                    label_ids.append(torch.zeros(self.n_labels, dtype=torch.int))
                else:
                    # convert list of label ids to multi-hot vector
                    label_vectors = torch.zeros(self.n_labels, dtype=torch.int)
                    for label_id in sample_response_tags[word_idx]:
                        label_vectors[label_id] = 1
                    label_ids.append(label_vectors)
            labels.append(label_ids)

        labels = [
            torch.stack(response_tags, dim=0) for response_tags in labels
        ]  # list[tensor: (seq_len, n_classes)]
        labels = torch.stack(labels, dim=0)  # tensor: (batch_size, seq_len, n_classes)

        return {
            "input_ids": tokenized_inputs["input_ids"],
            "attention_mask": tokenized_inputs["attention_mask"],
            "labels": labels,
        }


def compute_metrics(eval_pred, label2id):
    metrics = {}
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    # Apply sigmoid to get probabilities
    probs = torch.sigmoid(torch.from_numpy(logits)).numpy()
    # Threshold at 0.5 for multilabel binary
    preds = (probs > 0.5).astype(int)

    batch_size, seq_len, n_labels = preds.shape
    batch_size, seq_len, n_labels = labels.shape

    preds = preds.reshape(batch_size * seq_len, n_labels)
    labels = labels.reshape(batch_size * seq_len, n_labels)

    for label, id in label2id.items():
        mask = labels[:, id] != -100
        class_report = classification_report(
            labels[:, id][mask].tolist(),
            preds[:, id][mask].tolist(),
            zero_division=0,
            output_dict=True,
            labels=[0, 1],
            target_names=[f"not_{label}", label],
        )
        metrics.update(
            {
                f"f1_{label}": class_report[label]["f1-score"],
                f"precision_{label}": class_report[label]["precision"],
                f"recall_{label}": class_report[label]["recall"],
                f"support_{label}": class_report[label]["support"],
            }
        )

    labels = labels.astype(int)
    preds = preds.astype(int)

    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]

    metrics.update(
        {
            "accuracy": accuracy_score(labels, preds),
            "f1_micro": f1_score(
                labels, preds, average="micro", zero_division=0, labels=[1]
            ),
            "f1_macro": f1_score(
                labels, preds, average="macro", zero_division=0, labels=[1]
            ),
            "precision_micro": precision_score(
                labels, preds, average="micro", zero_division=0, labels=[1]
            ),
            "precision_macro": precision_score(
                labels, preds, average="macro", zero_division=0, labels=[1]
            ),
            "recall_micro": recall_score(
                labels, preds, average="micro", zero_division=0, labels=[1]
            ),
            "recall_macro": recall_score(
                labels, preds, average="macro", zero_division=0, labels=[1]
            ),
        }
    )

    return metrics


# https://discuss.huggingface.co/t/multi-label-token-classification/16509/7
class MultiLabelTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Use BCEWithLogitsLoss for multi-label classification
        logits = logits[labels != -100]
        labels = labels[labels != -100]
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        return (loss, outputs) if return_outputs else loss


def final_model_training(args):
    train_data_path = os.path.join(
        os.environ.get("SM_CHANNEL_DATASET_PATH"), "all.csv"
    )
    test_data_path = os.path.join(
        os.environ.get("SM_CHANNEL_DATASET_PATH"), "all.csv"
    )
    id2label_path = os.path.join(
        os.environ.get("SM_CHANNEL_DATASET_PATH"), 'id2label.json'
    )

    train_data = IdeaDetectionDataset(
        data_path=train_data_path,
        id2label_path=id2label_path,
        model_name=args.model_name,
    )
    test_data = IdeaDetectionDataset(
        data_path=test_data_path,
        id2label_path=id2label_path,
        model_name=args.model_name,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name,
        num_labels=train_data.n_labels,
        problem_type="multi_label_classification",
        label2id=train_data.tag2index_mapping,
        id2label=train_data.index2tag_mapping,
    )

    training_args = TrainingArguments(
        output_dir=os.environ.get("SM_OUTPUT_DIR"),
        logging_dir=os.environ.get("SM_OUTPUT_DATA_DIR"),
        remove_unused_columns=False,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=float(args.learning_rate),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=args.weight_decay,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        fp16=True,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to=["wandb", "tensorboard"],
    )

    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        data_collator=train_data.collate_fn,
        tokenizer=train_data.tokenizer,
        compute_metrics=partial(compute_metrics, label2id=train_data.tag2index_mapping),
    )

    trainer.train()
    wandb.finish()
    model.half()
    trainer.save_model(os.environ.get("SM_MODEL_DIR"))


def cross_validation_training(args):
    metrics_list = []
    hp_tuning_metrics = set(
        [
            "eval_f1_micro",
            "eval_f1_macro",
            "eval_precision_micro",
            "eval_precision_macro",
            "eval_recall_micro",
            "eval_recall_macro",
        ]
    )
    id2label_path = os.path.join(
        os.environ.get("SM_CHANNEL_DATASET_PATH"), 'id2label.json'
    )

    for fold in range(args.n_folds):
        train_path = os.path.join(
            os.environ.get("SM_CHANNEL_CV_DATA"), f"train_{fold}.csv"
        )
        test_path = os.path.join(
            os.environ.get("SM_CHANNEL_CV_DATA"), f"train_{fold}.csv"
        )

        train_data = IdeaDetectionDataset(
            data_path=train_path,
            id2label_path=id2label_path,
            model_name=args.model_name,
        )
        test_data = IdeaDetectionDataset(
            data_path=test_path, id2label_path=id2label_path, model_name=args.model_name
        )

        model = AutoModelForTokenClassification.from_pretrained(
            args.model_name,
            num_labels=train_data.n_labels,
            problem_type="multi_label_classification",
            label2id=train_data.tag2index_mapping,
            id2label=train_data.index2tag_mapping,
        )

        training_args = TrainingArguments(
            run_name=f"fold_{fold}",
            output_dir=os.environ.get("SM_OUTPUT_DIR"),
            logging_dir=os.environ.get("SM_OUTPUT_DATA_DIR"),
            remove_unused_columns=False,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            lr_scheduler_type=args.lr_scheduler_type,
            warmup_steps=args.warmup_steps,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=float(args.learning_rate),
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            weight_decay=args.weight_decay,
            metric_for_best_model="eval_macro_f1",
            greater_is_better=True,
            fp16=True,
            save_total_limit=1,
            load_best_model_at_end=True,
            report_to=["wandb", "tensorboard"],
        )

        trainer = MultiLabelTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            data_collator=train_data.collate_fn,
            tokenizer=train_data.tokenizer,
            compute_metrics=partial(
                compute_metrics, label2id=train_data.tag2index_mapping
            ),
        )

        trainer.train()
        eval_metrics = trainer.evaluate()
        metrics_list.append(eval_metrics)

    avg_metrics = {
        k: float(np.mean([m[k] for m in metrics_list]))
        for k in metrics_list[0]
        if k in hp_tuning_metrics
    }
    avg_metrics = {f"cv_avg_{k}": v for k, v in avg_metrics.items()}
    logger.info(avg_metrics)

    return avg_metrics


def main(args):
    os.environ["WANDB_PROJECT"] = args.project_name
    hp_keys = [k for k in vars(args) if not k.startswith("_")]
    group = "sweep_" + "_".join(f"{k}-{getattr(args, k)}" for k in hp_keys)
    os.environ["WANDB_RUN_GROUP"] = group

    if args.cv:
        cross_validation_training(args)
    else:
        final_model_training(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=str, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr_scheduler_type", type=str, default="linear")
    parser.add_argument("--project_name", type=str, default=None)
    parser.add_argument("--wandb_api_key", type=str, default=None)
    parser.add_argument("--train_dataset_name_stem", type=str)
    parser.add_argument("--test_dataset_name_stem", type=str)
    parser.add_argument(
        "--cv", action="store_true", help="Enable cross-validation mode"
    )
    parser.add_argument("--n_folds", type=int, default=10)

    args, _ = parser.parse_known_args()

    wandb.login(key=args.wandb_api_key)
    main(args)
