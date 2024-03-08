python -m venv .venv
.venv/Scripts/activate
python.exe -m pip install --upgrade pip
pip install tensorflow
pip freeze > requirements.txt
pip install matplotlib