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
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio===0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
```
