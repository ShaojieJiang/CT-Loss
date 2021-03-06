# Contrastive Token loss function for PyTorch

This repo is the clean (PyTorch) implementation of the **contrastive token loss** proposed in our paper: _A Simple Contrastive Learning Objective for Alleviating Neural Text Degeneration._
The source code for reproducing our results reported in the paper, including data pre-processing scripts, our trained models and interactive Google Colab notebooks, can be found in [this repo](https://github.com/ShaojieJiang/lit-seq).

## Installation

`pip install ct-loss`

## Usage
You can use our CT objective when **pretraining** or **finetuning** your augoregressive language models.
With CT, the resulting language models will have significantly less **repetitive** generations, even with deterministic decoding such as greedy and beam search.
It only takes several lines of code to use CT loss, around where you calculate PyTorch's `CrossEntropyLoss`.
Here is an example:
```python
import torch

# Suppose we already have the model output logits and labels (sequences of token indices).
# For example when the batch size is 10, sequence length is 50 and vocabulary size is 1000:
logits = torch.rand(10, 50, 1000)
labels = torch.randint(0, 999, (10, 50))

# This is how you normally use cross-entropy for a language model:
from torch.nn import CrossEntropyLoss
ce_criterion = CrossEntropyLoss()
ce_loss = ce_criterion(logits.view(-1, 1000), labels.view(-1))

# This is how you can use our contrastive token loss:
from ct.ct_loss import ContrastiveTokenLoss
ct_criterion = ContrastiveTokenLoss(pad_id=999) # we need pad tokens for masking out tokens in a sequence that should not be used as negative tokens
ct_loss = ct_criterion(logits, labels)

# In our paper [1], we use CE and CT together
loss = ce_loss + ct_loss

print(ce_loss, ct_loss)

>>> tensor(6.9536) tensor(1.5848)
```

## Cite our paper

```
@article{jiang2022contrastive,
  doi = {10.48550/ARXIV.2205.02517},
  url = {https://arxiv.org/abs/2205.02517},
  author = {Jiang, Shaojie and Zhang, Ruqing and Vakulenko, Svitlana and de Rijke, Maarten},
  title = {A Simple Contrastive Learning Objective for Alleviating Neural Text Degeneration},
  publisher = {arXiv},
  year = {2022},
}
```