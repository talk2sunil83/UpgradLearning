import ast
from flask import current_app
from flask import request
from flask_restful import Resource
from utils.extensions import api


class SimpleResource(Resource):
    def get(self):
        return 'To send and email, point your browser to: {send_email_url}'.format(send_email_url=api.url_for(SendEmailResource))


class SendEmailResource(Resource):
    def get(self):
        current_app.logger.info('{cls}.GET called'.format(cls=self.__class__.__name__))
        message = dict(subject=request.args.get('subject', 'test'),
                       recipients=ast.literal_eval(request.args.get('recipients', [])),
                       text_body=request.args.get('body', "Test Email"),
                       html_body=request.args.get('html', None),
                       sender=request.args.get('sender'))

        try:
            # Avoiding a circular import - there must be a better way...
            from utils.mail_service import send_email
            send_email(**message)
            return "Email sent to: {r}".format(r=message['recipients'])
        except Exception as err:
            current_app.logger.error("Failed to send email: {error}".format(error=err))
            return "Oops! An error occurred when sending the email"
