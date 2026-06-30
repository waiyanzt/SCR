<div align="center">

# Towards Graph Foundation Models: Training on Knowledge Graphs Enables Transferability to General Graphs #

[![pytorch](https://img.shields.io/badge/PyTorch_2.1+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![pyg](https://img.shields.io/badge/PyG_2.4+-3C2179?logo=pyg&logoColor=#3C2179)](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)
[![ULTRA arxiv](http://img.shields.io/badge/arxiv-2410.12609-yellow.svg)](https://arxiv.org/abs/2410.12609)
![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)

</div>

**Abstract**: Inspired by the success of large language models, there is a trend toward developing graph foundation models to conduct diverse downstream tasks in various domains. 
However, current models often require extra fine-tuning to apply their learned structural and semantic representations to new graphs, which limits their versatility.
Recent breakthroughs in zero-shot inductive reasoning on knowledge graphs (KGs), offer us a new perspective on extending KG reasoning to general graph applications.
In this paper, we introduce SCR, a unified graph reasoning framework designed to train on knowledge graphs and effectively generalize across a wide range of graph tasks and domains.
We begin by designing the task-specific KG structures to establish a unified topology for different task formats. 
Then we propose semantic-conditioned message passing, a novel mechanism addressing the inherent semantic isolation in traditional KG reasoning, by jointly modeling structural and semantic invariance patterns in graph representations.
Evaluated on 38 diverse datasets spanning node-, link-, and graph-level tasks, SCR achieves substantial performance gains over existing foundation models and supervised baselines, demonstrating its remarkable efficacy and adaptability.

Source code for evaluating and training the SCR model. Evaluation supports
zero-shot transfer across 39 datasets spanning three task types:

- **KG Inductive Reasoning** (25 datasets)
- **Node Classification** (6 datasets)
- **Graph Classification** (8 datasets)

Training can start from scratch, fine-tune the released SCR checkpoint, or
fine-tune from the ULTRA backbone.

## Data Acquisition

`resource/` (entity/relation text files used by semantic encoding, ~300 MB) and
`kg-datasets/` (evaluation datasets) are **not bundled** in this repository due
to size. Download both from Google Drive:

> 🔗 **[\[Google Drive\]](https://drive.google.com/file/d/1bk9EgIn83QGRmKDZFSqXrrUny1UwFWIt/view?usp=sharing)**

## Quick Start

### 1. Environment Setup

```bash
conda create -n scr python=3.11
conda activate scr

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data & Checkpoint

Follow [Data Acquisition](#data-acquisition) to populate `resource/` and
`kg-datasets/`. The checkpoints are already included in `ckpts/`:

```
ckpts/scr_3g.pth    (4.5 MB)   # released SCR model
ckpts/ultra_3g.pth  (2 MB)     # ULTRA backbone, for training variants
```

### 3. Run Evaluation

**Default config and checkpoint:**

```bash
python script/eval.py
```

**Specify GPU / custom config:**

```bash
python script/eval.py -c config/eval_all.yaml -g 0
```

### 4. Train

```bash
# Train from scratch
python script/train.py --ckpt none

# Fine-tune from the released SCR checkpoint
python script/train.py

# Fine-tune from a custom checkpoint on a specific GPU
python script/train.py --ckpt ./ckpts/my_model.pth -g 0
```

## Citation

```bibtex
@article{wang2024towards,
  title={Towards graph foundation models: Training on knowledge graphs enables transferability to general graphs},
  author={Wang, Kai and Luo, Siqiang and Shan, Caihua and Shen, Yifei},
  journal={arXiv preprint arXiv:2410.12609},
  year={2024}
}
```

## License

This code is provided for research purposes. The ULTRA codebase is subject to their respective licenses.

## Acknowledgement

We refer to the code of ULTRA. Thanks for their contributions.