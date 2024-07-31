## Setting up
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
pip install flask-socketio
```

make database and add starting data
```
flask seed
```
run the backend
```
python app.py
```

## Quick Backend Crash Course
example .env
```
S3_BUCKET=
S3_ACCESS_KEY=
S3_SECRET_ACCESS_KEY=
REDIS_ENDPOINT='redis://localhost:6379'

```
General information
1. All the database tables are in **models.py**
2. All the routes are found under separate Blueprint files in /routes
3. Helper functions used in the routes are found in /services under the same filename
4. The database is the file app.db
5. (Some) configs for the app are found in config.py
6. Refer to S3ImageDataset.py for functions and classes to interact with s3

General flow of the backend
1. **User creates a project and dataset instance in the database** (general.py)
    - The project is assigned a folder with a random UUID name which will contain the dataset and models
    - The bucket name is stored under Project.bucket
    - This folder name is stored under Project.prefix
2. **User uploads dataset in the form of zip file of images / csv file** (dataset.py)
    - The images are uploaded to s3
    - Their respective s3 file paths are saved under DataInstance.data
3. **Display batch of data points (with lowest confidence / highest entropy) in the frontend for labelling** (dataset.py)
4. **Update new label when user annotates data point** (data_instance.py)
    - The labels are saved as a comma separated string of numbers in the database under DataInstance.labels, with the number to class mapping found in Dataset.class_to_label_mapping
    - The data point is labelled as manually_processed in the database
5. **User trains model with parameters such as number of epochs and the model is run on unlabelled data points to calculate confidence/entropy** (model.py)
    - The model is exported and uploaded into s3
    - S3 file path for the model is saved in the database under Model.saved
    - The training results (accuracy, precision, f1, recall) are saved in the database under History
    - The results for each epoch (training/validation accuracy and loss) are saved in the database under Epoch
    - One History instance is tied to one or many Epochs
    - Confidence/entropy is saved in the database under DataInstance.entropy
6. **Display training results** (history.py)
7. **User can download model / dataset** (model.py / dataset.py


## Setting up Celery with Redis
ensure celery and redis are installed 
```
pip install celery redis
```

install redis [https://redis.io/docs/latest/operate/oss_and_stack/install/install-redis/]


make sure you have the redis endpoint in your .env file


start the redis server on localhost
```
redis-server
```
start a celery worker process
```
celery -A app.celery_app worker --loglevel=debug
```

start a celery worker process on WINDOWS (not sure if this works)
```
celery -A app.celery_app worker --loglevel=debug --pool=solo
```

start a celery worker process on MACOS
```
celery -A app.celery_app worker --loglevel=debug --pool threads
```

## Quick Celery Crash Course
Starting a background task:
1. **long_running_task in tasks.py**: Add the decorator @shared_task to the top of the function. 
    - If you use bind=True, it means that you need to pass in self as an argument, and this allows you to keep updating the state of the task using self.update_state while it's still running
    - If you use base=AbortableTask, it means that you can cancel the task while it is running halfway
2. **start_task in app.py**: Use <task_function_name>.delay() to create a new task and pass in the necessary arguments
    - You should see that the task is received in the terminal running the celery worker
3. **task_result in app.py**: Use AsyncResult(<task_id>) to fetch the status and info of the task
    - These are the possible states: PENDING, STARTED, RETRY, FAILURE, SUCCESS. When I used self.update_state, I added a new state PROGRESS and metadata of the current iteration. 
    - The metadata is stored in result.info, which is a dictionary. Note that the keys depend on the state of the task, and result.info may be None.

How to use this in training models:
1. **run_training in model.py**: Add the decorator @shared_task to your training function (not the api route)
2. **new_training_job in model.py**: In the API route function, use <training_function_name>.delay() to start running the task in celery. Save the task ID in the task_id column under History in the database and return the task ID to the frontend.
3. **socketio in app.py**: Use AsyncResult in the websocket to get the status updates of the task on the frontend. Note that only ONE task is monitored at a time per user, coded using the dictionary current_tasks.
