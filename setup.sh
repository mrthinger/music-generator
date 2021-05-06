poetry install
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install git+https://github.com/microsoft/DeepSpeed.git https://storage.googleapis.com/tensorbeat-public/pytorch_fast_transformers-0.4.0-cp39-cp39-linux_x86_64.whl
./setup_apex.sh