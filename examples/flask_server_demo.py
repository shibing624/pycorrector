# -*- coding: utf-8 -*-
"""
@author:David Euler
@description:
python3 -m pip install flask

flask --app flask_server_demo run

You can test the service by :
curl -H "Content-Type: application/json" -X POST -d '{"text":"我从北京南做高铁到南京南战，总共花四个小时的事件"}' http://127.0.0.1:5000/c
"""
import sys
from flask import Flask, request
from loguru import logger

sys.path.append("..")
from pycorrector import Corrector
from pycorrector.macbert.macbert_corrector import MacBertCorrector

app = Flask(__name__)
rule_model = Corrector()
macbert_model = MacBertCorrector()

help = """
You can request the service by HTTP get: <br> 
   /c?text=xxxxx, <p> 
or HTTP post with json: <br>  
   {"text":"xxxx"} <p>
Post example: 
  curl -H "Content-Type: application/json" -X POST -d '{"text":"我从北京南做高铁到南京南"}' http://127.0.0.1:5000/macbert_correct  
"""


@app.route("/", methods=['POST', 'GET'])
def hello_world():
    return help


@app.route('/rule_correct', methods=['POST', 'GET'])
def rule_correct():
    if request.method == 'POST':
        data = request.json
        logger.info("Received data: {}".format(data))
        text = data["text"]
        corrected_sent, detail = rule_model.correct(text)
        return corrected_sent + " " + str(detail)
    else:
        if "text" in request.args:
            text = request.args.get("text")
            logger.info("Received data: {}".format(text))
            corrected_sent, detail = rule_model.correct(text)
            return corrected_sent + " " + str(detail)
    return help


@app.route('/macbert_correct', methods=['POST', 'GET'])
def correct_api():
    if request.method == 'POST':
        data = request.json
        logger.info("Received data: {}".format(data))
        text = data["text"]
        results = macbert_model.macbert_correct(text)
        return results[0] + " " + str(results[1])
    else:
        if "text" in request.args:
            text = request.args.get("text")
            logger.info("Received data: {}".format(text))
            results = macbert_model.macbert_correct(text)
            return results[0] + " " + str(results[1])
    return help
