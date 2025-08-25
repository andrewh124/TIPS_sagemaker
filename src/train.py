import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functools import partial

from src.trainer.trainer import MultiLabelTrainer
from src.trainer.utils import compute_metrics
from src.dataloader.dataloader import IdeaDetectionDataset
from transformers import AutoModelForTokenClassification

from transformers import TrainingArguments
from transformers import logging
from transformers import set_seed

logging.set_verbosity_error()
set_seed(42)


def main():
    train_dataset = IdeaDetectionDataset(
        # data_path="data/processed/hela/train.csv",
        # id2label_path="data/processed/hela/id2label.json",
        data_path=os.path.join(os.environ.get("SM_CHANNEL_TRAIN"), 'processed/hela/train.csv'),
        id2label_path=os.path.join(os.environ.get("SM_CHANNEL_TRAIN"), 'processed/hela/id2label.json'),
        # n_rows_limit=50,  # Limit for debugging
    )
    eval_dataset = IdeaDetectionDataset(
        data_path=os.path.join(os.environ.get("SM_CHANNEL_TRAIN"), 'processed/hela/test.csv'),
        id2label_path=os.path.join(os.environ.get("SM_CHANNEL_TRAIN"), 'processed/hela/id2label.json')
    )

    model = AutoModelForTokenClassification.from_pretrained(
        "microsoft/deberta-v3-base",
        num_labels=26,
        problem_type="multi_label_classification",
        label2id=train_dataset.tag2index_mapping,
        id2label=train_dataset.index2tag_mapping,
    )

    # Initialize Trainer
    training_args = TrainingArguments(
        output_dir="output",
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        logging_dir="output/logs",
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=False,
        num_train_epochs=30,
        per_device_train_batch_size=16,
        logging_steps=10,
    )

    trainer = MultiLabelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=train_dataset.collate_fn,
        compute_metrics=partial(compute_metrics, label2id=train_dataset.tag2index_mapping),
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
