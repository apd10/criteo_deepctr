Please refer to the original github repo by 
https://github.com/shenweichen/DeepCTR-Torch

This is a just a wrapper to run criteo data on multiple models

To run criteo with compressed embeddings - UMA  (also called RMA)

1) create the criteo kaggle dataset as directed on original DeepCTR webpage

2) Download the repo https://github.com/apd10/universal_memory_allocation
  - and run python3 setup install in that repo

In the criteo_run folder
3) CUDA_VISIBLE_DEVICES=0 python3 train_criteo_kaggle.py --config config.yml


sample config.yml
```
seed: 2
train:
    batch_size: 2048
model: "xdeepfm"
xdeepfm:
    reg: 0
embedding:
    size: 16
    etype: "rma"
    rma:
        memory: 500000
```


