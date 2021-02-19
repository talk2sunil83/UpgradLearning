# Ref: https://github.com/etiology/Flask-RESTful-with-async-Flask-Mail

from utils.api_endpoints import SendEmailResource
from utils.api_endpoints import SimpleResource
from utils.extensions import api


def register_endpoints(app):
    """ connects flask_restful endpoints to the app """
    api.app = app

    api.add_resource(SimpleResource, '/')
    api.add_resource(SendEmailResource, '/0.1/send_email/')
