# -*- coding: utf-8 -*-
"""
@author:David Euler
@description:
python3 -m pip install flask
"""
import sys

from flask import Flask, request
from loguru import logger

sys.path.append("../..")
from pycorrector import Corrector
from pycorrector.macbert.macbert_corrector import MacBertCorrector

app = Flask(__name__)
kenlm_model = Corrector()
kenlm_model.check_corrector_initialized()
macbert_model = MacBertCorrector()

help = """
You can request the service by HTTP get: <br> 
   http://0.0.0.0:5001/macbert_correct?text=我从北京南做高铁到南京南<br>
   
or HTTP post with json: <br>  
   {"text":"xxxx"} <p>
Post example: <br>
  curl -H "Content-Type: application/json" -X POST -d '{"text":"我从北京南做高铁到南京南"}' http://0.0.0.0:5001/macbert_correct
"""


@app.route("/", methods=['POST', 'GET'])
def hello_world():
    return help


@app.route('/kenlm_correct', methods=['POST', 'GET'])
def kenlm_correct():
    if request.method == 'POST':
        data = request.json
        logger.info("Received data: {}".format(data))
        text = data["text"]
        corrected_sent, detail = kenlm_model.correct(text)
        return corrected_sent + " " + str(detail)
    else:
        if "text" in request.args:
            text = request.args.get("text")
            logger.info("Received data: {}".format(text))
            corrected_sent, detail = kenlm_model.correct(text)
            return corrected_sent + " " + str(detail)
    return help


@app.route('/macbert_correct', methods=['POST', 'GET'])
def correct_api():
    if request.method == 'POST':
        data = request.json
        logger.info("Received data: {}".format(data))
        text = data["text"]
        results = macbert_model.correct(text)
        return results[0] + " " + str(results[1])
    else:
        if "text" in request.args:
            text = request.args.get("text")
            logger.info("Received data: {}".format(text))
            results = macbert_model.correct(text)
            return results[0] + " " + str(results[1])
    return help


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5001, debug=True)
