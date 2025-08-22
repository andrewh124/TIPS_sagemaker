from pathlib import Path
import json
import sys

from loguru import logger
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class IdeaDetectionDataset(Dataset):
    def __init__(self, data_path: str, id2label_path: str, n_rows_limit: int = None):
        logger.info(f"Loading dataset from {data_path}")
        logger.info(f"Using id2label mapping from {id2label_path}")
        if n_rows_limit is not None:
            logger.warning(f"Limiting dataset to {n_rows_limit} rows for debugging")
        self.tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-bert")
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
        # df['span_label'] = df['span_label'].fillna('_')  # Fill NaN with empty string
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
