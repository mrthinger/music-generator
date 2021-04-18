poetry install
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
pip install deepspeed==0.3.13