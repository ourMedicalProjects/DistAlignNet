# DistAlignNet

This repository provides a lung nodule detection framework trained on the **LUNA16** dataset, implementing an end-to-end 3D nodule detection network.

---

## Environment Setup

Install dependencies via `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Data Preparation

Before training, prepare the dataset as follows:

1. Preprocess the raw LUNA16 data into `.npy` format
2. Place all preprocessed `.npy` files into a single directory
3. Update the data path in `config.py`:

```python
data_config = {
    'data_dir': "your/data/directory",  # set to your actual data directory
    ...
}
```

---

## Training

### Single-Fold Training

Run training for a specific fold (0-9):

```bash
bash train.sh <fold_id>
```

For example, to train on fold 0:

```bash
bash train.sh 0
```

Main training arguments:

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 60 | Number of training epochs |
| `--batch-size` | 6 | Batch size |
| `--init-lr` | 1e-4 | Initial learning rate |
| `--num-workers` | 4 | Number of data loading workers |
| `--momentum` | 0.9 | Momentum |
| `--gpu_id` | 1 | GPU device ID |
| `--epoch-save` | 5 | Checkpoint saving interval (epochs) |
| `--warm-up` | 10 | Learning rate warm-up epochs |
| `--fold` | - | Fold index for cross-validation |

### 10-Fold Cross-Validation Training

Run the full 10-fold cross-validation:

```bash
bash train_10_fold.sh
```

---

## Network Architecture

The network architecture is defined in the `models/` directory and can be customized as needed:

```
models/
├── backbone/
│   ├── resnet.py       # ResNet backbone
│   └── unet.py         # UNet backbone
├── heads/
│   ├── cpm_head.py     # CPM detection head
│   ├── rpn.py          # RPN detection head
│   └── tood.py         # TOOD detection head
├── block.py            # Basic building blocks
├── cfg.py              # Network configuration
└── detector.py         # Detector main module
```

---

## Testing

Update the checkpoint path in `test_config` within `config.py`, then run:

```bash
bash test.sh
```

---

## Project Structure

```
DistAlignNet-origin/
├── annos/              # Annotation files
├── datasets/           # Dataset loading and preprocessing
├── evaluationScript/   # Official LUNA16 evaluation scripts
├── models/             # Network architecture definitions
├── spilts/luna16/      # 10-fold data split files
├── utils/              # Utility functions (loss, box utils, etc.)
├── config.py           # Global configuration file
├── train.py            # Training entry point
├── train.sh            # Single-fold training script
├── train_10_fold.sh    # 10-fold training script
└── test.py             # Testing entry point
```
