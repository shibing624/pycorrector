from pycorrector.macbert.macbert_corrector import MacBertCorrector
from pycorrector import config
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
      
