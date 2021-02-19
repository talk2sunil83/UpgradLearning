from flask import Flask

from utils.extensions import mail
from utils.endpoints import register_endpoints
from utils.settings import ProdConfig

# app context needs to be accessible at the module level
# for the send_message.send_
app = Flask(__name__)


def create_app(config=ProdConfig):
    """ configures and returns the the flask app """
    app.config.from_object(config)

    register_extensions()
    register_endpoints(app)

    return app


def register_extensions():
    """ connects flask extensions to the app """
    mail.init_app(app)
