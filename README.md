# UniCO: A Unified Cross-View Collaboration Network

## Version Notice

This repository was initially uploaded with a preliminary development version of the code. 
Some scripts in that version were used for intermediate experiments and debugging, and therefore may not exactly match the final experimental protocol described in the manuscript.

The current repository has been reorganized and updated to match the final implementation used for the reported results. 

## Layout

- `src/unico/data/`: dataset loading, scaling, sliding windows
- `src/unico/models/`: UniCO network and encoders
- `src/unico/training/`: SVDD-style training loop
- `src/unico/evaluation/`: metrics
- `src/unico/preprocessing/`: data preparation
- `src/unico/utils/`: runtime helpers
- `scripts/`: command line entry points
- `configs/`: default experiment settings

## Installation

```bash
cd UniCO
pip install -r requirements.txt
```

## Data Preparation

Put processed data under `UniCO/Data/`, or pass `--data-dir` to point at another data folder.

Standardize label names if the raw dataset uses a custom label column:

```bash
python scripts/preprocess_data.py standardize-label \
  --input-csv raw.csv \
  --output-csv Data/processed.csv \
  --label-column "Attack LABLE (0:No Attack, 1:Attack)"
```

Downsample large raw CSV files:

```bash
python scripts/preprocess_data.py downsample \
  --input-csv raw_swat.csv \
  --output-csv Data/normal_swat_data.csv \
  --step 10
```

Generate image windows for the auxiliary branch:

```bash
python scripts/preprocess_data.py generate-images \
  --input-csv Data/normal_swat_data.csv \
  --output-npy Data/gaf_images_swat.npy \
  --method gaf \
  --window-size 16 \
  --max-channels 24
```

## Training and Testing


```bash
python scripts/train_unico.py --dataset swat/wadi/kdd
```
