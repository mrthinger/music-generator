# Secret Sauce

Secret Sauce is the code to train a music generating neural network.

# Requirements

- python 3.9
- poetry
- linux

# Setup

Poetry doesnt play nicely with cuda versions of torch

```sh
./setup.sh
wget https://storage.googleapis.com/tensorbeat-public/savant-train-wdlc.zip
unzip savant-train-wdlc.zip
rm savant-train-wdlc.zip
```

```sh
git config --global user.name "Evan Pierce"
git config --global user.email "mrthinger@gmail.com"
```