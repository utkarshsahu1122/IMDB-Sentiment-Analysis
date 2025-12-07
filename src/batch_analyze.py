import json
import os
from typing import Optional, List, Dict, Any, Set

from tqdm import tqdm

from .language_client import LanguageService
from .dataset_loader import load_imdb_dataset


OUTPUT_PATH = "data/processed/imdb_language_results.jsonl"


def load_processed_ids(output_path: str) -> Set[int]:
    """
    Collect IDs of already processed reviews from existing JSONL file.
    """
    processed: Set[int] = set()
    if not os.path.exists(output_path):
        return processed

    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            rid = obj.get("id")
            if rid is not None:
                processed.add(int(rid))
    return processed


def run_batch(
    max_rows: Optional[int] = None,
    batch_size: int = 10,
    output_path: str = OUTPUT_PATH,
) -> None:
    """
    Run Azure sentiment analysis on IMDB dataset with resume support.

    - max_rows: how many rows to consider from the dataset (sampled).
    - batch_size: how many documents per API call (10 is safe).
    - Results appended to JSONL.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    all_records = load_imdb_dataset(max_rows=max_rows)
    print(f"✔ Loaded {len(all_records)} records from IMDB dataset (sampled).")

    already = load_processed_ids(output_path)
    print(f"✔ Already processed: {len(already)} rows")

    # filter
    records = [r for r in all_records if r["id"] not in already]
    print(f"✔ Remaining to process: {len(records)} rows")

    if not records:
        print("✔ Nothing left to process. Exiting.")
        return

    service = LanguageService()

    with open(output_path, "a", encoding="utf-8") as f_out:
        # process in batches
        for i in tqdm(range(0, len(records), batch_size), desc="Analyzing sentiment"):
            batch = records[i : i + batch_size]
            texts = [r["review"] for r in batch]

            try:
                results = service.analyze_sentiment_batch(texts)
            except Exception as e:
                print(f"[WARN] Batch {i} failed: {e}")
                continue

            for rec, res in zip(batch, results):
                row: Dict[str, Any] = {
                    "id": rec["id"],
                    "review": rec["review"],
                    "label": rec["label"],  # ground truth
                    "azure_result": res,
                }
                f_out.write(json.dumps(row) + "\n")
                f_out.flush()


if __name__ == "__main__":
    # Total: 49396 reviews → ~5000 API calls
    run_batch(max_rows=None, batch_size=10)
