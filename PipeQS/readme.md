# PipeQS-code
PipeQS is an adaptive quantization and staleness-aware pipeline distributed training system for GNNs. It dynamically adjusts the bit-width of message quantization and manages staleness to reduce both communication overhead and communication waiting time.

## Setup

### Environment

#### Software Dependencies

- Ubuntu 22.04
- Python 3.12
- CUDA 12.1
- [PyTorch 2.3.0](https://github.com/pytorch/pytorch)
- [DGL 2.3.0](https://github.com/dmlc/dgl)

### Installation

#### Step 1: Install Software Dependencies

please  install PyTorch and DGL.

```
conda install -c dglteam/label/th23_cu121 dgl
```

#### Step 2: Setup Quantization Code

```
python helper/quantization/setup.py install
```

### Run with Single Compute Node

please run `scripts/reddit_full.sh`,  `scripts/ogbn-products_full.sh` or  `scripts/yelp_full.sh`

### Run with Multiple Compute Nodes

please run `obgn_multi_node.sh`
