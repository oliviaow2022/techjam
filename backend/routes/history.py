from flask import Blueprint, request, jsonify
from models import db, History

history_routes = Blueprint('history', __name__)

# for debugging only
@history_routes.route('/all', methods=['GET'])
def get_all_history():
    history = History.query.all()
    history_list = [h.to_dict() for h in history]
    return jsonify(history_list), 200