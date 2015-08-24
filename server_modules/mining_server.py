#!/usr/bin/python
#-*- coding:utf-8 -*-
from flask.ext.cors import CORS
from flask import Flask, render_template, request, request, current_app, make_response, jsonify
import ResponseMachine
import json
import logging
import sys

app = Flask(__name__)
module_obj = None

# Load module object from Mining Engine.
def init():
    CORS(app)
    app.config['CORS_HEADERS'] = 'Content-Type'
    global module_obj
    module_obj = ResponseMachine.ResponseClassifierRequests(topn=20)

def apply_logger():
	app.logger.setLevel(logging.DEBUG)
	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	file_handler = logging.FileHandler("debug.log")
	file_handler.setFormatter(formatter)
	app.logger.addHandler(file_handler)

# Proccess queries with mining engine and response json type result data to client.
# Resopnse data contain each genger score , 10 category score and 10 related queries that other users searched before.
@app.route('/server/query', methods=['GET', 'POST'])
def proccess(): # The query is type of list like apple pizza
	data = json.loads(request.data,encoding='utf-8')
	queries = json.loads(data.get('queries'),encoding='utf-8').encode('utf-8')
	#queries = unicode(json.loads(data.get('queries')),'euc-kr')
	#queries = request.get_json().get('queries')
	app.logger.info(type(queries))
	app.logger.info("--")
	app.logger.info(' '.join(queries))
	result_data = module_obj.response_query([queries])
	app.logger.info(result_data)
	

	if type(result_data) is str:
		no_data = jsonify(dict([['result','']]))
		return no_data
	else:
		return jsonify(result_data)

# Get data that contain selected gender type and category type from client.
# Resopnse corresponded top 10 queries.
@app.route('/server/segment', methods=['GET', 'POST'])
def process2():
	data = json.loads(request.data,encoding='utf-8')
	gender = int(json.loads(data.get('gender')).encode('utf-8'))
	category = int(json.loads(data.get('category')).encode('utf-8'))

	result_data = module_obj.response_ranking(gender,category)
	
	result = {'word':result_data[0], 'score':result_data[1]}

	return jsonify(result)


if __name__ == '__main__':
	reload(sys)
	sys.setdefaultencoding('utf-8')
	init()
	apply_logger()
	app.run(host='0.0.0.0',debug=True)
