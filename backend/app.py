import click
import csv

from create_app import create_app
from flask.cli import with_appcontext
from models import db, User, Project, Dataset, DataInstance, Model
from flask import jsonify, request
from celery.result import AsyncResult
from services.tasks import long_running_task

flask_app, celery_app = create_app()
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
        "value": result.result if result.ready() else None,
        "current": str(result.info),
    }


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
   fashion_mnist = Project(name="Fashion Mnist", user_id=user.id, bucket=os.getenv('S3_BUCKET'), prefix='fashion-mnist', type="Single Label Classification")
   cifar10 = Project(name="Cifar-10", user_id=user.id, type="1")
   db.session.add_all([ants_bees, fashion_mnist, cifar10])
   db.session.commit()


   ants_bees_ds = Dataset(name="Ants and Bees", project_id=ants_bees.id, num_classes=2, class_to_label_mapping={0: 'ants', 1: 'bees'})
   db.session.add(ants_bees_ds)
   db.session.commit()


   resnet18 = Model(name='ResNet-18', project_id=ants_bees.id)
   densenet121 = Model(name='DenseNet-121', project_id=ants_bees.id)
   alexnet = Model(name='AlexNet', project_id=ants_bees.id)
   convnext_base = Model(name='ConvNext Base', project_id=ants_bees.id)
   resnet18_2 = Model(name='ResNet-18', project_id=fashion_mnist.id)
   resnet18_3 = Model(name='ResNet-18', project_id=cifar10.id)
   db.session.add_all([resnet18, densenet121, alexnet, convnext_base, resnet18_2, resnet18_3])
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
    flask_app.run(debug=True, port=5001)
