"""Timestamp-based split of MP-grounded records (paper Methods).

Records without ``created_at`` are excluded (305 of 153,235 in the paper);
the remainder is sorted by ``created_at``; the most recent 2,000 records form
the initial held-out test set, the preceding 2,000 the validation set, and
everything older is training data.

    python -m matbrain.data.split --records data/mp/mp_records.jsonl --out-dir data/splits
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

from matbrain.data.formats import iter_records, write_jsonl


def _ts(value) -> datetime | None:
    if value in (None, "", "None"):
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def timestamp_split(records: list[dict], test_size: int = 2000, val_size: int = 2000) -> dict[str, list[dict]]:
    dated = [(r, _ts(r.get("created_at"))) for r in records]
    missing = [r for r, t in dated if t is None]
    ordered = [r for r, t in sorted(((r, t) for r, t in dated if t is not None), key=lambda x: (x[1], x[0]["material_id"]))]
    if len(ordered) < test_size + val_size + 1:
        raise ValueError("not enough dated records for the requested split sizes")
    return {
        "train": ordered[: len(ordered) - test_size - val_size],
        "val": ordered[len(ordered) - test_size - val_size : len(ordered) - test_size],
        "test": ordered[len(ordered) - test_size :],
        "excluded_no_timestamp": missing,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out-dir", default="data/splits")
    ap.add_argument("--test-size", type=int, default=2000)
    ap.add_argument("--val-size", type=int, default=2000)
    args = ap.parse_args()
    splits = timestamp_split(list(iter_records(args.records)), args.test_size, args.val_size)
    for name, rows in splits.items():
        write_jsonl(os.path.join(args.out_dir, f"{name}.jsonl"), rows)
        print(f"{name}: {len(rows)}")


if __name__ == "__main__":
    main()
