from flask import Blueprint, request, jsonify
from models import db, Epoch

epoch_routes = Blueprint('epoch', __name__)

# for debugging only
@epoch_routes.route('/all', methods=['GET'])
def get_all_epochs():
    epochs = Epoch.query.all()
    epoch_list = [epoch.to_dict() for epoch in epochs]
    return jsonify(epoch_list), 200