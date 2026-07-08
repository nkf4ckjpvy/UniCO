from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from unico.data import load_kdd, load_swat, load_wadi
from unico.training import UniCOTrainer
from unico.utils import project_root, resolve_device, set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate UniCO.")
    parser.add_argument("--dataset", choices=["swat", "wadi", "kdd"], required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--fusion-type", choices=["ib", "l2_mlp", "attention_concat", "concat"], default="ib")
    parser.add_argument("--normal-csv", type=str, default=None)
    parser.add_argument("--attack-csv", type=str, default=None)
    parser.add_argument("--data-file", type=str, default="kddcup.data_10_percent_corrected")
    parser.add_argument("--image-npy", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.5e-6)
    parser.add_argument("--reg-weight", "--beta", dest="reg_weight", type=float, default=0.1)
    parser.add_argument("--consistency-lambda", type=float, default=0.01)
    parser.add_argument("--threshold-percentile", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--table-input-channels", type=int, default=1)
    parser.add_argument("--table-feature-dim", type=int, default=64)
    parser.add_argument("--image-feature-dim", type=int, default=32)
    parser.add_argument("--fusion-bottleneck-dim", type=int, default=32)
    parser.add_argument("--final-dim", type=int, default=128)
    parser.add_argument("--lstm-hidden-dim", type=int, default=64)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--random-image-encoder", action="store_true")
    parser.add_argument("--no-image", action="store_true")
    parser.add_argument("--no-lstm", action="store_true")
    parser.add_argument("--no-point-adjust", action="store_true")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-test-batches", type=int, default=0)
    parser.add_argument("--train-window-offset", type=int, default=0)
    parser.add_argument("--test-window-offset", type=int, default=0)
    parser.add_argument("--max-train-windows", type=int, default=None)
    parser.add_argument("--max-test-windows", type=int, default=None)
    parser.add_argument("--wadi-attack-skip-rows", type=int, default=8000)
    parser.add_argument("--kdd-anomaly-ratio", type=float, default=0.2)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--lr-milestones", type=int, nargs="*", default=[30, 40])
    return parser


def default_files(args: argparse.Namespace) -> None:
    if args.dataset == "swat":
        args.normal_csv = args.normal_csv or "normal_swat_data.csv"
        args.attack_csv = args.attack_csv or "attack_swat_data.csv"
        args.image_npy = args.image_npy or "mtf_images_swat.npy"
    elif args.dataset == "wadi":
        args.normal_csv = args.normal_csv or "normal_wadi_data.csv"
        args.attack_csv = args.attack_csv or "attack_wadi_data.csv"
        args.image_npy = args.image_npy or "mtf_images_wadi.npy"
    else:
        args.image_npy = args.image_npy or "mtf_images_kdd.npy"


def load_data(args: argparse.Namespace):
    data_dir = args.data_dir or (project_root() / "Data")
    if args.dataset == "swat":
        return load_swat(
            data_dir,
            args.normal_csv,
            args.attack_csv,
            args.image_npy,
            args.window_size,
            args.stride,
            args.batch_size,
            args.train_window_offset,
            args.test_window_offset,
            args.max_train_windows,
            args.max_test_windows,
            load_images=not args.no_image,
        )
    if args.dataset == "wadi":
        return load_wadi(
            data_dir,
            args.normal_csv,
            args.attack_csv,
            args.image_npy,
            args.window_size,
            args.stride,
            args.batch_size,
            args.wadi_attack_skip_rows,
            args.train_window_offset,
            args.test_window_offset,
            args.max_train_windows,
            args.max_test_windows,
            load_images=not args.no_image,
        )
    return load_kdd(
        data_dir,
        args.data_file,
        args.image_npy,
        args.window_size,
        args.stride,
        args.batch_size,
        args.seed,
        args.kdd_anomaly_ratio,
        load_images=not args.no_image,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    default_files(args)
    set_seed(args.seed)

    bundle = load_data(args)
    args.num_features = bundle.num_features
    args.image_input_channels = bundle.image_channels
    args.pretrained_image_encoder = not args.random_image_encoder
    device = resolve_device(args.device)

    print(f"model=UniCO dataset={args.dataset} device={device}")
    print(f"Normal windows shape: {bundle.train_loader.dataset.table_windows.shape}")
    print(f"Attack windows shape: {bundle.test_loader.dataset.table_windows.shape}")
    print(f"Number of attack segments: {len(bundle.attack_segments)}")
    print(f"train_windows={len(bundle.train_loader.dataset)} test_windows={len(bundle.test_loader.dataset)}")
    print(f"num_features={args.num_features} image_channels={args.image_input_channels}")
    if not args.no_image:
        image_path = (args.data_dir or (project_root() / "Data")) / args.image_npy
        image_windows = bundle.train_loader.dataset.image_windows
        print(f"image_path={image_path}")
        print(
            "image_windows="
            f"shape={tuple(image_windows.shape)} "
            f"min={image_windows.min().item():.6f} "
            f"max={image_windows.max().item():.6f} "
            f"mean={image_windows.mean().item():.6f} "
            f"std={image_windows.std().item():.6f}"
        )

    trainer = UniCOTrainer(args, bundle.train_loader, bundle.test_loader, device)
    trainer.train()
    result = trainer.evaluate(bundle.attack_segments)

    run_name = args.run_name or f"unico_{args.dataset}_{args.fusion_type}"
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root() / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = result["metrics"]
    summary = {
        "model": "UniCO",
        "dataset": args.dataset,
        "fusion_type": args.fusion_type,
        "device": str(device),
        "image_npy": args.image_npy,
        "threshold_selection": "train_score_95th_percentile",
        "score_scale": "raw_svdd_distance",
        "evaluation_adjustment": metrics["adjustment"],
        "metrics": metrics,
    }

    (output_dir / f"{run_name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.save(output_dir / f"{run_name}_scores.npy", result["scores"])
    np.save(output_dir / f"{run_name}_train_scores.npy", result["train_scores"])
    np.save(output_dir / f"{run_name}_labels.npy", result["labels"])
    np.save(output_dir / f"{run_name}_predictions.npy", result["predictions"])

    print(json.dumps(summary, indent=2))
    print(f"saved_dir={output_dir}")


if __name__ == "__main__":
    main()
