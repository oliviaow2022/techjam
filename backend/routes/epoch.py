from flask import Blueprint, request, jsonify
from models import db, Epoch
from flask_jwt_extended import jwt_required

epoch_routes = Blueprint('epoch', __name__)

# for debugging only
@epoch_routes.route('/all', methods=['GET'])
@jwt_required()
def get_all_epochs():
    epochs = Epoch.query.all()
    epoch_list = [epoch.to_dict() for epoch in epochs]
    return jsonify(epoch_list), 200