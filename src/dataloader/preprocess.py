import json
from pathlib import Path

import pandas as pd
from loguru import logger
from sklearn.model_selection import KFold, train_test_split
from tqdm.auto import tqdm


def extract_span_tags_metadata(export_dir: str, output_dir=None) -> pd.DataFrame:
    export_dir = Path(export_dir)  # type: ignore
    metadata_path = export_dir / "exportedproject.json"  # type: ignore

    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    tag_sets = metadata["tag_sets"]

    for tag_set in tag_sets:
        if tag_set["name"] == "Span labels":
            tags = tag_set["tags"]
            tag_names = [tag["tag_name"] for tag in tags]
            tag_descriptions = [tag["tag_description"] for tag in tags]

            tag_df = pd.DataFrame(
                {"tag_name": tag_names, "tag_description": tag_descriptions}
            )

            break

    logger.info(f"Extracted {len(tag_df)} tags from metadata.")
    logger.info(f"Tags: {tag_df['tag_name'].tolist()}")  # type: ignore

    if output_dir:
        output_path = Path(output_dir) / "tags_metadata.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tag_df.to_csv(output_path, index=False)
        logger.info(f"Saved tags metadata to {output_path}")

    return tag_df  # type: ignore


def tags_metadata_to_id2label_mapping(export_dir: str, output_dir=None) -> dict:
    # tag_df = pd.read_csv(Path(tag_metadata_path) / "tags_metadata.csv")
    tag_df = extract_span_tags_metadata(export_dir, output_dir)

    id2label = {i: tag for i, tag in enumerate(tag_df["tag_name"].tolist())}
    logger.info(f"Generated id2label mapping with {len(id2label)} entries.")

    if output_dir:
        output_path = Path(output_dir) / "id2label.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(id2label, f, indent=2)
        logger.info(f"Saved id2label mapping to {output_path}")

    return id2label


def _extract_file_paths(export_dir: str, preferred_annotator: str) -> dict:
    logger.info(
        f"Processing CSV export from {export_dir} for annotator {preferred_annotator}"
    )

    sample_id2path = {}
    annotation_parh = Path(export_dir) / "annotation"
    annotation_files = list(annotation_parh.glob("**/*.tsv"))
    curattion_path = Path(export_dir) / "curation"
    curation_files = list(curattion_path.glob("**/*.tsv"))

    for file in tqdm(annotation_files + curation_files):
        sample_id = str(file.parts[-2])
        if sample_id in sample_id2path and "curation" in str(file):
            sample_id2path[sample_id] = file
        elif sample_id in sample_id2path and preferred_annotator in str(file):
            sample_id2path[sample_id] = file
        else:
            sample_id2path[sample_id] = file

    logger.info(f"Found {len(sample_id2path)} unique samples with annotations.")

    return sample_id2path


def process_tsv_export_for_idea_detection(
    export_dir: str, preferred_annotator: str, output_dir=None
) -> pd.DataFrame:
    sample_id2path = _extract_file_paths(
        export_dir=export_dir,
        preferred_annotator=preferred_annotator,
    )
    out = []

    pbar = tqdm(sample_id2path.items())
    for sample_id, file_path in pbar:
        pbar.set_description(f"Processing {sample_id}")

        df = pd.read_csv(
            file_path,
            sep="\t",
            names=[
                "token_id",
                "character_offset",
                "token",
                "span_label",
                "placeholder",
            ],
            skiprows=5,
            # on_bad_lines='skip',
            # comment='#'
        )
        df = df.drop("placeholder", axis=1)
        df.insert(0, "sample_id", sample_id)
        # Remove square brackets and the content in the span label `[...]`
        # The brackets and the numbers are counting the number of ideas of that sample
        df["span_label"] = df["span_label"].fillna("_")
        df["span_label"] = df["span_label"].astype(str)
        df["span_label"] = df["span_label"].str.replace(r"\[.*?\]", "", regex=True)
        out.append(df)

    out_df = pd.concat(out, ignore_index=True)
    logger.info(f"Processed {len(out_df)} rows from {len(sample_id2path)} samples.")

    if output_dir:
        output_path = Path(output_dir) / "all.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_path, index=False)
        logger.info(f"Saved processed data to {output_path}")

    return out_df


def split_by_dialogue(data_path: str, output_dir: str) -> None:
    df = pd.read_csv(Path(data_path))
    logger.info(f"Loaded data from {data_path} with {len(df)} samples.")
    dialogue_ids = df["sample_id"].unique().tolist()
    logger.info(f"Found {len(dialogue_ids)} unique dialogue IDs.")

    train_ids, test_ids = train_test_split(dialogue_ids, test_size=0.1, random_state=42)
    train_df = df[df["sample_id"].isin(train_ids)]
    test_df = df[df["sample_id"].isin(test_ids)]

    logger.info(
        f"Split data into {len(train_df)} training and {len(test_df)} testing samples."
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train_df.to_csv(Path(output_dir) / "train.csv", index=False)
    test_df.to_csv(Path(output_dir) / "test.csv", index=False)
    logger.info(f"Saved split data to {output_dir}.")


def kfold_validation_split_for_idea_detection(
    df: pd.DataFrame, n_splits: int = 10, random_state: int = 42
) -> dict:
    # This is different from ki data because each row is a token of a response, not a full response, we can't split by row
    logger.info(f"Performing K-Fold validation with {n_splits} splits")
    dialogue_ids = df["sample_id"].unique()
    logger.info(f"Found {len(dialogue_ids)} unique dialogue IDs.")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = {}

    for i_fold, (train_index, test_index) in enumerate(kf.split(dialogue_ids)):
        train_ids = dialogue_ids[train_index]
        test_ids = dialogue_ids[test_index]
        train_df = df[df["sample_id"].isin(train_ids)].reset_index(drop=True)
        test_df = df[df["sample_id"].isin(test_ids)].reset_index(drop=True)
        splits[f"train_{i_fold}"] = train_df
        splits[f"test_{i_fold}"] = test_df

        logger.info(f"Fold {i_fold}: {len(train_df)=} -- {len(test_df)=}")

    logger.info(f"Created {n_splits} train-test splits")

    return splits


def main():
    # tag_df = extract_span_tags_metadata(
    #     export_dir="data/raw/project-hela-2025-08-08-171658",
    #     output_dir="data/raw",
    # )
    id2label = tags_metadata_to_id2label_mapping(
        export_dir="data/raw/project-hela-2025-08-08-171658",
        output_dir="data/processed/hela",
    )
    all_data = process_tsv_export_for_idea_detection(
        export_dir="data/raw/project-hela-2025-08-08-171658",
        preferred_annotator="kellybillings",
        output_dir="data/raw",
    )

    split_by_dialogue(
        data_path="data/raw/processed_data.csv",
        output_dir="data/processed/hela",
    )


if __name__ == "__main__":
    main()
