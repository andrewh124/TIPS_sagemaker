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
    dataset = IdeaDetectionDataset(
        data_path="data/processed/hela/train.csv",
        id2label_path='data/processed/hela/id2label.json',
        n_rows_limit=500  # Limit to 500 rows for debugging
    )

    model = AutoModelForTokenClassification.from_pretrained(
        'hf-internal-testing/tiny-bert',
        num_labels=26,
        problem_type='multi_label_classification',
        label2id=dataset.tag2index_mapping,
        id2label=dataset.index2tag_mapping
    )
    
    # Initialize Trainer
    training_args = TrainingArguments(
        output_dir="output",
        eval_strategy='epoch',
        save_strategy='no',
        logging_strategy='epoch',
        logging_dir='output/logs',
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
        train_dataset=dataset,
        eval_dataset=dataset,
        data_collator=dataset.collate_fn,
        compute_metrics=partial(compute_metrics, id2label=dataset.index2tag_mapping),
    )

    # # Start training
    # trainer.train()


if __name__ == "__main__":
    main()