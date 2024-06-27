### Setting up
create virtual environment
```
python3 -m venv .venv
```
activate virtual environment
```
. .venv/bin/activate
```
install dependencies
```
pip install -r requirements.txt
```
installed dependencies
```
pip install flask
pip install flask-sqlalchemy
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install pandas
pip install scikit-learn
pip install tqdm
pip install flask-jwt-extended
pip install Flask-Cors
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

https://github.com/munniomer/pytorch-tutorials/blob/master/beginner_source/finetuning_torchvision_models_tutorial.py