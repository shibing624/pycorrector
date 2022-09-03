#!/bin/bash
# python3 -m pip install flask

echo You can test the service by :
echo "curl -H \"Content-Type: application/json\" -X POST -d '{\"text\":\"我从北京南做高铁到南京南战，总共花四个小时的事件\"}' http://127.0.0.1:5000/c"

flask --app flask_server run
