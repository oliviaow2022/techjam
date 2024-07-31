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
pip install python-dotenv
pip install celery redis
pip install git+https://github.com/modAL-python/modAL.git
```

make database and add starting data
```
flask seed
```

### Setting up Celery with Redis
ensure celery and redis are installed 
```
pip install celery redis
```

install redis [https://redis.io/docs/latest/operate/oss_and_stack/install/install-redis/]


start the redis server on localhost
```
redis-server
```
start a celery worker process
```
celery -A app.celery_app worker --loglevel=debug
```

start a celery worker process on WINDOWS
```
celery -A app.celery_app worker --loglevel=debug --pool=solo
```

start a celery worker process on MACOS
```
celery -A app.celery_app worker --loglevel=debug --pool threads
```

### Running
```
flask run --debug 
```

If access to localhost is denied:
```
python app.py
```