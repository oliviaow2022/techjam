from flask import Blueprint, request, jsonify
from models import db, History, Epoch, Project, Model
from flasgger import swag_from

history_routes = Blueprint('history', __name__)

# for debugging only
@history_routes.route('/all', methods=['GET'])
def get_all_history():
    history = History.query.all()
    history_list = [h.to_dict() for h in history]
    return jsonify(history_list), 200


@history_routes.route('/<int:project_id>/info', methods=['GET'])
def get_project_history(project_id):
    project = Project.query.get_or_404(project_id, description="Project ID not found")
    models = Model.query.filter_by(project_id=project.id).all()

    model_ids = [model.id for model in models]
        
    # Retrieve history objects where model_id is in the list of model IDs
    history_list = History.query.filter(History.model_id.in_(model_ids)).all()
    
    history_with_epochs = []
    for history in history_list:
        epochs = Epoch.query.filter_by(history_id=history.id).all()
        epoch_list = [epoch.to_dict() for epoch in epochs]
        history_with_epochs.append({
            'history': history.to_dict(),
            'epochs': epoch_list
        })
    
    return jsonify(history_with_epochs), 200


@history_routes.route('/<int:id>/info', methods=['GET'])
@swag_from({
    'tags': ['History'],
    'summary': 'Get model statistics',
    'parameters': [
        {
            'name': 'id',
            'in': 'path',
            'required': True,
            'description': 'ID of the history to retrieve',
            'schema': {'type': 'integer'}
        }
    ]
})
def get_training_info(id):
    history = History.query.get_or_404(id, description="History ID not found")
    epochs = Epoch.query.filter_by(history_id=history.id).all()
    epoch_list = [epoch.to_dict() for epoch in epochs]
    return jsonify({'history': history.to_dict(), 'epochs': epoch_list}), 200