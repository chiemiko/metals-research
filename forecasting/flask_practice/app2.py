from flask import Flask, jsonify, request

'''TUTORIAL UDEMY #61 - creating endpoints'''

'''
FOUR HTTP VERBS:
1. GET - grabs data from server
2. POST - Adds data to server
3. 
4. 
'''


# created object of flask 
app = Flask(__name__)


stores = [
	{
	'name': 'Store Name 1',
	'items': [{'name': 'Item Name _ Spaghetti', 'price': 5.99}]
	}]



###### 5 ENDPOINTS BELOW ####

# This browser will only allow for POSTING methods (default is GET)

# POST /store data
@app.route('/store', methods=['POST'])
def create_store():
	request_data = request.get_json()

# Get /store/<string:name>
@app.route('/store/<string:name>')
def get_store(name):
	pass

# GET /store 
@app.route('/store')
def get_stores():
	# Must assign stores list into a dictionary of stores
	return jsonify('stores': stores)

# POST /store/<string:name>/item {name:, price} json?
@app.route('/store/<string:name>item', methods=['POST'])
def create_item_in_store(name):
	pass


# GET /store/<string:name>/item
@app.route('/store/<string:name>/item')
def get_items_in_store(name):


	pass

app.run(port=5000)