import argparse
import random
from pathlib import Path


def write_list(output_dir: Path, filename: str, items: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / filename
    with open(target, "w", encoding="utf-8") as f:
        for item in items:
            f.write(f"{item}\n")
    print(f"Created {filename} with {len(items)} samples")


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]

    normalized_dir = (repo_root / args.normalized_dir).resolve()
    output_dir = (repo_root / args.output_dir).resolve()

    files = sorted([f.name for f in normalized_dir.glob("*.npz")])

    random.seed(args.seed)
    random.shuffle(files)

    total = len(files)
    train_split = int(total * args.train_ratio)
    val_split = int(total * args.val_ratio)

    train_files = files[:train_split]
    val_files = files[train_split : train_split + val_split]
    test_files = files[train_split + val_split :]

    print(f"\n{'='*60}")
    print(f"📊 Creating dataset splits from {normalized_dir}")
    print(f"{'='*60}")
    print(f"Found {total} processed samples.")

    write_list(output_dir, "train_samples.txt", train_files)
    write_list(output_dir, "val_samples.txt", val_files)
    write_list(output_dir, "test_samples.txt", test_files)

    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
