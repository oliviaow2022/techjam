from flask import Blueprint, request, jsonify
from models import db, History, Epoch

history_routes = Blueprint('history', __name__)

# for debugging only
@history_routes.route('/all', methods=['GET'])
def get_all_history():
    history = History.query.all()
    history_list = [h.to_dict() for h in history]
    return jsonify(history_list), 200

@history_routes.route('/<int:id>/info', methods=['GET'])
def get_training_info():
    history = History.query.get_or_404(id, description="History ID not found")
    epochs = Epoch.query.filter_by(history_id=history.id).all()
    epoch_list = [epoch.to_dict() for epoch in epochs]
    return jsonify({'history': history.to_dict(), 'epochs': epoch_list}), 200