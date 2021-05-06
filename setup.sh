poetry install
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install deepspeed==0.3.16 https://storage.googleapis.com/tensorbeat-public/pytorch_fast_transformers-0.4.0-cp39-cp39-linux_x86_64.whl
