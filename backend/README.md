### Setting up
create virtual environment
```
python3 -m venv .venv
```
activate virtual environment
```
. .venv/bin/activate
```
installed dependencies
```
pip install flask
pip install flask-sqlalchemy
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pandas
pip install PyJWT
pip install scikit-learn
```

### Running
make database and add starting data
```
flask seed
```

```
flask run --debug 
```

If access to localhost is denied:
```
python app.py
```

https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html