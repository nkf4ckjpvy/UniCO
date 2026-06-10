from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unico.preprocessing import downsample_csv, generate_image_windows, standardize_label_column


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UniCO data preprocessing tools.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    downsample = subparsers.add_parser("downsample")
    downsample.add_argument("--input-csv", type=Path, required=True)
    downsample.add_argument("--output-csv", type=Path, required=True)
    downsample.add_argument("--step", type=int, default=10)
    downsample.add_argument("--seed", type=int, default=42)

    labels = subparsers.add_parser("standardize-label")
    labels.add_argument("--input-csv", type=Path, required=True)
    labels.add_argument("--output-csv", type=Path, required=True)
    labels.add_argument("--label-column", type=str, required=True)
    labels.add_argument("--output-label", type=str, default="label")

    images = subparsers.add_parser("generate-images")
    images.add_argument("--input-csv", type=Path, required=True)
    images.add_argument("--output-npy", type=Path, required=True)
    images.add_argument("--label-column", type=str, default="label")
    images.add_argument("--method", choices=["gaf", "mtf", "rp"], default="gaf")
    images.add_argument("--window-size", type=int, default=16)
    images.add_argument("--stride", type=int, default=1)
    images.add_argument("--max-rows", type=int, default=10000)
    images.add_argument("--max-channels", type=int, default=24)
    images.add_argument("--image-size", type=int, default=224)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "downsample":
        downsample_csv(args.input_csv, args.output_csv, args.step, args.seed)
    elif args.command == "standardize-label":
        standardize_label_column(args.input_csv, args.output_csv, args.label_column, args.output_label)
    elif args.command == "generate-images":
        generate_image_windows(
            args.input_csv,
            args.output_npy,
            label_column=args.label_column,
            method=args.method,
            window_size=args.window_size,
            stride=args.stride,
            max_rows=args.max_rows,
            max_channels=args.max_channels,
            image_size=args.image_size,
        )
    print("done")


if __name__ == "__main__":
    main()
