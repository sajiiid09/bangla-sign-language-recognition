import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def parse_sample_name(sample_name: str) -> dict | None:
    parts = Path(sample_name).stem.split("__")
    if len(parts) != 5:
        return None
    return {
        "word": parts[0],
        "signer": parts[1],
        "session": parts[2],
        "repetition": parts[3],
        "grammar": parts[4],
    }


def write_list(output_dir: Path, filename: str, items: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / filename
    with open(target, "w", encoding="utf-8") as f:
        for item in items:
            f.write(f"{item}\n")
    print(f"Created {filename} with {len(items)} samples")


def count_signers(files: list[str]) -> dict[str, int]:
    counts: Counter = Counter()
    for file_name in files:
        parsed = parse_sample_name(file_name)
        if parsed is not None:
            counts[parsed["signer"]] += 1
    return dict(sorted(counts.items()))


def parse_signer_list(value: str | None) -> set[str]:
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def split_random(
    files: list[str], train_ratio: float, val_ratio: float, seed: int
) -> tuple[list[str], list[str], list[str]]:
    shuffled = files[:]
    random.seed(seed)
    random.shuffle(shuffled)

    total = len(shuffled)
    train_cut = int(total * train_ratio)
    val_cut = int(total * val_ratio)

    train_files = shuffled[:train_cut]
    val_files = shuffled[train_cut : train_cut + val_cut]
    test_files = shuffled[train_cut + val_cut :]
    return train_files, val_files, test_files


def split_stratified_word_signer(
    files: list[str], train_ratio: float, val_ratio: float, seed: int
) -> tuple[list[str], list[str], list[str]]:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for file_name in files:
        parsed = parse_sample_name(file_name)
        if parsed is None:
            continue
        grouped[(parsed["word"], parsed["signer"])].append(file_name)

    random.seed(seed)
    train_files: list[str] = []
    val_files: list[str] = []
    test_files: list[str] = []

    for _, group_files in grouped.items():
        random.shuffle(group_files)
        n = len(group_files)

        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            if n_train + n_val >= n:
                n_val = max(1, n - n_train - 1)
                if n_train + n_val >= n:
                    n_train = max(1, n - n_val - 1)
        elif n == 2:
            n_train, n_val = 1, 0
        elif n == 1:
            n_train, n_val = 1, 0

        train_files.extend(group_files[:n_train])
        val_files.extend(group_files[n_train : n_train + n_val])
        test_files.extend(group_files[n_train + n_val :])

    random.shuffle(train_files)
    random.shuffle(val_files)
    random.shuffle(test_files)
    return train_files, val_files, test_files


def split_signer_holdout(
    files: list[str], train_signers: set[str], val_signers: set[str], test_signers: set[str]
) -> tuple[list[str], list[str], list[str]]:
    train_files: list[str] = []
    val_files: list[str] = []
    test_files: list[str] = []

    for file_name in files:
        parsed = parse_sample_name(file_name)
        if parsed is None:
            continue
        signer = parsed["signer"]
        if signer in train_signers:
            train_files.append(file_name)
        elif signer in val_signers:
            val_files.append(file_name)
        elif signer in test_signers:
            test_files.append(file_name)

    return train_files, val_files, test_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create train/val/test split files")
    parser.add_argument(
        "--normalized_dir",
        type=str,
        default="Data/processed/new_model/normalized",
        help="Directory containing normalized .npz files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="Data/processed/new_model",
        help="Directory to write split text files",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible shuffling",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train split ratio",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["random", "stratified-word-signer", "signer-holdout"],
        default="stratified-word-signer",
        help="Split strategy",
    )
    parser.add_argument(
        "--train_signers",
        type=str,
        default="",
        help="Comma-separated signer IDs for train set (signer-holdout mode)",
    )
    parser.add_argument(
        "--val_signers",
        type=str,
        default="",
        help="Comma-separated signer IDs for val set (signer-holdout mode)",
    )
    parser.add_argument(
        "--test_signers",
        type=str,
        default="",
        help="Comma-separated signer IDs for test set (signer-holdout mode)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]

    normalized_dir = (repo_root / args.normalized_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve()

    files = sorted([f.name for f in normalized_dir.glob("*.npz")])

    if args.mode == "random":
        train_files, val_files, test_files = split_random(
            files, args.train_ratio, args.val_ratio, args.seed
        )
    elif args.mode == "stratified-word-signer":
        train_files, val_files, test_files = split_stratified_word_signer(
            files, args.train_ratio, args.val_ratio, args.seed
        )
    else:
        train_signers = parse_signer_list(args.train_signers)
        val_signers = parse_signer_list(args.val_signers)
        test_signers = parse_signer_list(args.test_signers)
        if not train_signers or not val_signers or not test_signers:
            raise ValueError(
                "For signer-holdout mode, provide --train_signers, --val_signers, and --test_signers"
            )
        train_files, val_files, test_files = split_signer_holdout(
            files, train_signers, val_signers, test_signers
        )

    print(f"\n{'='*60}")
    print(f"📊 Creating dataset splits from {normalized_dir}")
    print(f"{'='*60}")
    print(f"Mode: {args.mode}")
    print(f"Found {len(files)} processed samples.")

    write_list(output_dir, "train_samples.txt", train_files)
    write_list(output_dir, "val_samples.txt", val_files)
    write_list(output_dir, "test_samples.txt", test_files)

    summary = {
        "mode": args.mode,
        "total": len(files),
        "train": len(train_files),
        "val": len(val_files),
        "test": len(test_files),
        "signer_distribution": {
            "train": count_signers(train_files),
            "val": count_signers(val_files),
            "test": count_signers(test_files),
        },
    }
    summary_path = output_dir / "split_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\nSigner distribution:")
    print(f"  Train: {summary['signer_distribution']['train']}")
    print(f"  Val:   {summary['signer_distribution']['val']}")
    print(f"  Test:  {summary['signer_distribution']['test']}")
    print(f"Summary JSON: {summary_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
