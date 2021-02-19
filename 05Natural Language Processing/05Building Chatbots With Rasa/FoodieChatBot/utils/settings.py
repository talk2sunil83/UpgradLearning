# Ref: https://github.com/etiology/Flask-RESTful-with-async-Flask-Mail
import os


class BaseConfig():
    """ Basic settings required by all classes """
    MAIL_SERVER = 'smtp.gmail.com'
    MAIL_PORT = 465
    MAIL_USE_SSL = True
    MAIL_USERNAME = "YourEmail@gmail.com"
    MAIL_PASSWORD = "YourPassword"
    MAIL_DEFAULT_SENDER = "address.to.send.from@email.com"
    MAIL_DEBUG = True

    # Flask
    DEBUG = True


class DevConfig(BaseConfig):
    """ A collection of settings to use during dev work """
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER')
    MAIL_DEBUG = True


class ProdConfig(BaseConfig):
    """ Settings for production

    NOTE: Remember that for production you want a server like Apache, nginx, or gunicorn to run the WSGI app.
    See: www.digitalocean.com/community/tutorials/how-to-serve-flask-applications-with-uwsgi-and-nginx-on-ubuntu-14-04)
    """
    DEBUG = False
    MAIL_DEBUG = False
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_DEFAULT_SENDER')
