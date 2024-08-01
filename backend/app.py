import click
import csv
import os
import threading
import time

from dotenv import load_dotenv
from create_app import create_app
from flask.cli import with_appcontext
from models import db, User, Project, Dataset, DataInstance, Model
from flask import jsonify, request
from celery.result import AsyncResult
from services.tasks import long_running_task
from flask_socketio import emit

load_dotenv()

flask_app, celery_app, socketio = create_app()
flask_app.app_context().push()

@flask_app.get("/trigger_task")
def start_task() -> dict[str, object]:
    iterations = request.args.get('iterations')
    print(iterations)
    result = long_running_task.delay(int(iterations))
    return {"result.id": result.id}


@flask_app.get("/get_result")
def task_result() -> dict[str, object]:
    result_id = request.args.get('result_id')
    result = AsyncResult(result_id)
    print(result)
    return {
        "id": result.id,
        "state":  result.state,
        "current": str(result.info),
    }

# Dictionary to keep track of the current task (ONLY ONE) being monitored for each client
current_tasks = {}

def monitor_task(result_id, sid):
    while True:
        result = AsyncResult(result_id)
        socketio.emit('task_update', {
            "id": result.id,
            "state": result.state,
            "info": result.info,
        }, room=sid)
        if result.state in ('SUCCESS', 'FAILURE'):
            break
        time.sleep(2)
    # Final state
    socketio.emit('task_update', {
            "id": result.id,
            "state": result.state,
            "info": result.info,
        }, room=sid)

# WebSocket event for connecting
@socketio.on('connect')
def handle_connect():
    emit('Client connected')


# WebSocket event for disconnecting
@socketio.on('disconnect')
def handle_disconnect():
    emit('Client disconnected')


# WebSocket event for monitoring task result
@socketio.on('monitor_task')
def handle_monitor_task(data):
    result_id = data.get('result_id')
    sid = request.sid
    if result_id:
        if sid in current_tasks:
            current_tasks.pop(sid, None)
        
        current_tasks[sid] = result_id
        threading.Thread(target=monitor_task, args=(result_id, sid)).start()
    else:
        emit('error', {'message': 'Missing result_id'})


@flask_app.route('/')
def hello_world():
   return jsonify({"message": "Welcome to the API!"})


@flask_app.cli.command('seed')
@with_appcontext
def seed():
    db.drop_all()
    db.create_all()

    """Seed the database."""
    user = User(email='test@gmail.com', username='testuser')
    user.set_password('testuser')
    db.session.add(user)
    db.session.commit()

    ants_bees = Project(name="Multi-Class Classification", user_id=user.id, bucket=os.getenv('S3_BUCKET'), prefix='transfer-antsbees', type="Single Label Classification")
    db.session.add(ants_bees)
    db.session.commit()

    ants_bees_ds = Dataset(name="Ants and Bees", project_id=ants_bees.id, num_classes=2, class_to_label_mapping={0: 'ants', 1: 'bees'})
    db.session.add(ants_bees_ds)
    db.session.commit()

    resnet18 = Model(name='ResNet-18', project_id=ants_bees.id)
    densenet121 = Model(name='DenseNet-121', project_id=ants_bees.id)
    alexnet = Model(name='AlexNet', project_id=ants_bees.id)
    convnext_base = Model(name='ConvNext Base', project_id=ants_bees.id)
    db.session.add_all([resnet18, densenet121, alexnet, convnext_base])
    db.session.commit()

    """Seed the database from a CSV file."""
    file_path = 'dataset.csv'
    with open(file_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data_instance = DataInstance(
                data=row['filename'],
                labels=row['label'],
                dataset_id=1,
                manually_processed=bool(row.get('manually_processed', False))
            )
            db.session.add(data_instance)
        db.session.commit()

    click.echo('Seed data added successfully.')


if __name__ == "__main__":
    socketio.run(flask_app, debug=True, port=5001)
