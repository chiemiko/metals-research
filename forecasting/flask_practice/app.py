from flask import Flask

# created object of flask 
app = Flask(__name__)

# Created a route which was for the homepage (using /)
@app.route('/')

# Method that has to return something OUTPUTTED TO THE BROWSER
def home():
	return "Hello, World!"

app.run(port=5000)