import os
import pandas as pd
import zipfile

from flask import Blueprint, request, jsonify, send_file, make_response
from models import db, Model, Project, Dataset, DataInstance, History
from services.senti import run_training, run_zero_shot_labelling
from tempfile import TemporaryDirectory
from io import BytesIO
from botocore.exceptions import ClientError
from services.S3ImageDataset import s3

senti_routes = Blueprint('senti', __name__)


@senti_routes.route('<int:dataset_id>/upload', methods=['POST'])
def upload_file(dataset_id):
    text_column = request.form.get("text_column")
    file = request.files['file']

    if not all([file, text_column]):
        return jsonify({"error": "Missing fields required"}), 400
    
    # Ensure the file is either a CSV or a TXT
    filename, file_extension = os.path.splitext(file.filename)
    if file_extension.lower() not in ['.csv', '.txt']:
        return jsonify({"error": "Invalid file type. Only CSV and TXT files are allowed."}), 400

    # Read the file based on its extension
    if file_extension.lower() == '.csv':
        df = pd.read_csv(file)
    elif file_extension.lower() == '.txt':
        df = pd.read_csv(file, delimiter="\t")

    print(df)

    for index, row in df.iterrows():
        data_instance = DataInstance(
            data = row[text_column],
            dataset_id = dataset_id
        )
        db.session.add(data_instance)
        db.session.commit()

        
    return jsonify({"message": "File uploaded successfully"}), 200

@senti_routes.route('<int:project_id>/zero-shot', methods=['POST'])
def zero_shot(project_id):
    project = Project.query.get_or_404(project_id, description="Project ID not found")
    dataset = Dataset.query.filter_by(project_id=project.id).first()

    task = run_zero_shot_labelling.delay(dataset.to_dict())

    return jsonify({"task_id": task.id}), 200


@senti_routes.route('<int:project_id>/train', methods=['POST'])
def train_model(project_id):
    print(request.json)
    model_name = request.json.get('model_name')
    model_description = request.json.get('model_description')
    train_test_split_ratio = request.json.get('train_test_split', 0.8)
    test_size = 1 - train_test_split_ratio

    project = Project.query.get_or_404(project_id, description="Project ID not found")
    dataset = Dataset.query.filter_by(project_id=project.id).first()
    model = Model.query.filter_by(project_id=project.id).first()
    if not model:
        if not model_name:
            return jsonify({"error": "Model name required"}), 400

        model = Model(name=model_name, project_id=project.id, description=model_description)
        db.session.add(model)
        db.session.commit()

    if not dataset:
       return jsonify({"error": "Dataset does not exist"}), 400
    
    history = History(model_id=model.id)
    db.session.add(history)
    db.session.commit()

    task = run_training.delay(project.to_dict(), model.to_dict(), dataset.to_dict(), test_size)

    history.task_id = task.id
    db.session.commit()

    return jsonify({'task_id': task.id}), 200


@senti_routes.route('<int:history_id>/download', methods=['GET'])
def download_model(history_id):
    history = History.query.get_or_404(history_id)
    model_db = Model.query.get_or_404(history.model_id, description="Model ID not found")
    project_db = Project.query.get_or_404(model_db.project_id, description="Project ID not found")

    print(model_db)
    if not history.model_path:
        return jsonify({'Message': 'Model file not found'}), 404 

    try:
        model_file_path = history.model_path

        # Extract the directory path and base file name from the S3 path
        model_dir = os.path.dirname(model_file_path)
        model_file_name = os.path.basename(model_file_path)

        # Remove the file extension from the base file name
        model_file_stem = os.path.splitext(model_file_name)[0]

        # Create the new file name for the vectorizer
        vectorizer_file_name = f"{model_file_stem}_vectorizer.pkl"

        # Combine the directory and new file name to form the full path
        vectorizer_file_path = os.path.join(model_dir, vectorizer_file_name)

        print(history.model_path, vectorizer_file_name)

        # Create temporary directory
        with TemporaryDirectory() as temp_dir:
            # Define local paths for the files
            local_model_file_path = os.path.join(temp_dir, model_file_name)
            local_vectorizer_file_path = os.path.join(temp_dir, vectorizer_file_name)

            print(local_model_file_path, local_vectorizer_file_path)

            # Download files from S3
            s3.download_file(project_db.bucket, history.model_path, local_model_file_path)
            s3.download_file(project_db.bucket, vectorizer_file_path, local_vectorizer_file_path)

            # Check if the files were downloaded
            if not os.path.exists(local_model_file_path) or not os.path.exists(local_vectorizer_file_path):
                return jsonify({"error": "One or both files not found"}), 404

            # Create a ZIP file with both files
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
                zip_file.write(local_model_file_path, model_file_name)
                zip_file.write(local_vectorizer_file_path, vectorizer_file_name)
            zip_buffer.seek(0)

            # Send the ZIP file
            response = make_response(send_file(zip_buffer, as_attachment=True, download_name=f'{model_file_stem}_files.zip'))
            response.headers['Content-Type'] = 'application/zip'
            return response

    except ClientError as e:
        print(e)
        return jsonify({"error": f"Error downloading file from S3: {str(e)}"}), 500
    except Exception as e:
        print(e)
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500
        