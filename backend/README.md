### Setting up
```
python3 -m venv .venv
. .venv/bin/activate
pip install flask
pip install flask-sqlalchemy
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pandas
```

### Running
```
flask --app backend init-db
```