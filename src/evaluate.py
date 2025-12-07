import json
from typing import List

import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


JSONL_PATH = "data/processed/imdb_language_results.jsonl"


def load_results(path: str) -> pd.DataFrame:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def main():
    print(f"Loading results from {JSONL_PATH} ...")
    df = load_results(JSONL_PATH)
    print(f"Loaded {len(df)} rows")

    # ground truth
    df["label"] = df["label"].str.lower().str.strip()

    # predicted sentiment (skip rows with errors)
    def extract_pred(row):
        res = row["azure_result"]
        if not isinstance(res, dict):
            return None
        if res.get("error"):
            return None
        return str(res.get("sentiment", "")).lower().strip()

    df["pred"] = df.apply(extract_pred, axis=1)

    # drop error rows
    df = df[df["pred"].notna()].copy()
    print(f"After dropping error rows: {len(df)} rows")

    # Azure returns ['positive','negative','neutral','mixed']; IMDB only positive/negative.
    # For simplicity, map neutral/mixed to nearest:
    def map_to_binary(s: str) -> str:
        if s == "positive":
            return "positive"
        if s == "negative":
            return "negative"
        # neutral/mixed → we can choose one rule; here we treat them as 'neutral'
        # but they don't exist in GT, so we can consider them wrong in binary eval.
        return "neutral"

    df["pred_mapped"] = df["pred"].apply(map_to_binary)

    # Only evaluate on positive/negative; neutral we treat as misclassification
    # (because GT has no neutral class).
    # So we just leave it as third class and let classification_report handle it.
    print("\nLabel distribution (ground truth):")
    print(df["label"].value_counts())

    print("\nLabel distribution (predicted):")
    print(df["pred_mapped"].value_counts())

    print("\nConfusion matrix (labels: negative, neutral, positive):")
    labels = ["negative", "neutral", "positive"]
    print(confusion_matrix(df["label"], df["pred_mapped"], labels=labels))

    print("\nClassification report:")
    print(classification_report(df["label"], df["pred_mapped"], labels=labels))


if __name__ == "__main__":
    main()
