# A pycorrector flask demo service (for development & test)
# Author: David Euler
# Date: 2022/09/04

# python3 -m pip install flask

# flask --app flask_server_demo run

#You can test the service by :
# curl -H "Content-Type: application/json" -X POST -d '{"text":"我从北京南做高铁到南京南战，总共花四个小时的事件"}' http://127.0.0.1:5000/c


from pycorrector.macbert.macbert_corrector import MacBertCorrector
from pycorrector import config
import pycorrector
from flask import Flask, request, jsonify
from loguru import logger
import json

app = Flask(__name__)

help = """You can request the service by HTTP get: <br> 
               /c?text=xxxxx, <p> 
           or HTTP post with json: <br>  
               {"text":"xxxx"} <p>
          Post example: 
              curl -H "Content-Type: application/json" -X POST -d '{"text":"我从北京南做高铁到南京南"}' http://127.0.0.1:5000/c  
        """

@app.route("/", methods=['POST','GET'])
def hello_world():
    return help

correct = MacBertCorrector(config.macbert_model_dir).macbert_correct

@app.route('/pyc', methods=['POST','GET']) 
def py_correct():
    if request.method == 'POST':
      data = request.json
      logger.info("Received data: {}".format(data))
      text = data["text"]
      corrected_sent, detail = pycorrector.correct(text)
      return corrected_sent + " " + str(detail)
    else:
      if "text" in request.args:
        text = request.args.get("text")
        logger.info("Received data: {}".format(text))
        corrected_sent, detail = pycorrector.correct(text)
        return corrected_sent + " " + str(detail)
    return help
    

@app.route('/c', methods=['POST','GET']) 
def correct_api():
    if request.method == 'POST':
      data = request.json
      logger.info("Received data: {}".format(data))
      text = data["text"]
      results = correct(text)
      return results[0] + " " + str(results[1])
    else:
      if "text" in request.args:
        text = request.args.get("text")
        logger.info("Received data: {}".format(text))
        results = correct(text)
        return results[0] + " " + str(results[1])
    return help
      
