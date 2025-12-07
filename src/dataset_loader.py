import os
import random
from typing import List, Dict, Optional

import pandas as pd


def load_imdb_dataset(
    path: str = "data/raw/imdb/IMDB_Dataset_CLEANED.csv",
    max_rows: Optional[int] = None,
    random_seed: int = 42,
) -> List[Dict]:
    """
    Load IMDB 50K dataset and optionally sample max_rows rows.

    Expected columns: 'review', 'sentiment' (positive/negative).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Please download IMDB_Dataset.csv from Kaggle and place it there."
        )

    df = pd.read_csv(path)

    if "review" not in df.columns or "sentiment" not in df.columns:
        raise ValueError(f"Expected columns 'review' and 'sentiment' in {path}.")

    # normalize labels
    df["sentiment"] = df["sentiment"].str.lower().str.strip()

    if max_rows is not None and max_rows < len(df):
        df = df.sample(n=max_rows, random_state=random_seed)

    # build list of records
    records: List[Dict] = []
    for idx, row in df.iterrows():
        records.append(
            {
                "id": int(idx),
                "review": str(row["review"]),
                "label": str(row["sentiment"]),  # 'positive' or 'negative'
            }
        )

    return records
