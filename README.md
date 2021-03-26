# Secret Sauce

Secret Sauce is the code to train a music generating neural network.

# Requirements

- python 3.9
- poetry

# Setup

Poetry doesnt play nicely with cuda versions of torch

```sh
poetry install
activate venv
pip install --upgrade pip
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 pytorch-lightning==1.2.5 -f https://download.pytorch.org/whl/torch_stable.html
```
