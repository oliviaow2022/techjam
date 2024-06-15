### Setting up
```
python3 -m venv .venv
. .venv/bin/activate
pip install flask
pip install flask-sqlalchemy
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pandas
pip install PyJWT

```

### Running


```
flask run --debug 
```

If access to localhost is denied:
```
python app.py
```

https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html