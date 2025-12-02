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

## Requirements ##

- pytorch  2.1.0+cu121
- torch_geometric  2.4.0
- torch_scatter 2.1.2+pt21cu121
- transformers 4.41.1
  
Accepted by NeurIPS 2025. Source code is coming soon.

## Citation ##

If you find this codebase useful in your research, please cite the original paper.

```bibtex
@article{wang2024towards,
  title={Towards graph foundation models: Training on knowledge graphs enables transferability to general graphs},
  author={Wang, Kai and Luo, Siqiang and Shan, Caihua and Shen, Yifei},
  journal={arXiv preprint arXiv:2410.12609},
  year={2024}
}
```

## Acknowledgement ##

We refer to the code of [ULTRA](https://github.com/DeepGraphLearning/ULTRA). Thanks for their contributions.
