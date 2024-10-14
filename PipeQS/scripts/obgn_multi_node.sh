#!/bin/bash

n_partitions=8

parts_per_node=$((n_partitions / 2))

python main.py \
  --port 18129 \
  --enable-pipeline \
  --partition-method metis \
  --dataset ogbn-products \
  --dropout 0.5 \
  --lr 0.01 \
  --n-partitions $n_partitions \
  --n-epochs 3000 \
  --model graphsage \
  --n-layers 4 \
  --n-hidden 256 \
  --n-class 41 \
  --n-feat 602 \
  --n-train 153431 \
  --master-addr 127.0.0.1 \
  --node-rank 1 \
  --parts-per-node $parts_per_node \
  --log-every 10 \
  --fix-seed \
  --use-pp
