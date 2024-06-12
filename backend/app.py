from flask import Flask, jsonify
from config import Config
from models import db
from user import user_routes
from project import project_routes

app = Flask(__name__)
app.register_blueprint(user_routes, url_prefix='/user')
app.register_blueprint(project_routes, url_prefix='/project')

app.config.from_object(Config)
db.init_app(app)

@app.before_request
def create_tables():
    # The following line will remove this handler, making it
    # only run on the first request
    app.before_request_funcs[None].remove(create_tables)

    db.create_all()

@app.route('/')
def hello_world():
   return jsonify({"message": "Welcome to the API!"})

if __name__ == '__main__':
    app.run(debug=True)