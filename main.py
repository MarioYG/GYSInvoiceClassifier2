import numpy as np
from extract import ReconocerImagen
import time
import json
# Copyright 2015 Google Inc. All Rights Reserved.
import logging

from flask import Flask,request,Response
import base64

app = Flask(__name__)

@app.route('/')
def hello():
    """Return a friendly HTTP greeting."""
    return 'Hello World!'


@app.route('/api/v1/SmartFac', methods=['POST'])
def Amadeus():
    """Return a friendly HTTP greeting."""
    data = request.get_json(silent=True)
    try:
        imgstring = data["factura"]
        imgdata = base64.b64decode(imgstring)
        filename = 'test.jpg' 
        with open(filename, 'wb') as f:
            f.write(imgdata)
        data = ReconocerImagen('test.jpg')
        js = json.dumps(data)
        resp = Response(js, status=200, mimetype='application/json')
        return resp

    except ValueError:
    	data = {
    		'Message'  : "Request Incorrecto"
    	}
    	js = json.dumps(data)
    	resp = Response(js, status=400, mimetype='application/json')
    	return resp


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

if __name__ == '__main__':
    # This is used when running locally. Gunicorn is used to run the
    # application on Google App Engine. See entrypoint in app.yaml.
    # [END app]
    app.run(host='127.0.0.1', port=8080, debug=True)

# compute sigmoid nonlinearity


