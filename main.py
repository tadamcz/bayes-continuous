from flask import Flask
from flask import request
import bayes

app = Flask(__name__)

@app.route('/')
def my_form():
	form = '''<form method="POST">
	Prior
	<input name="prior-location">
	<input name="prior-scale">

	<br>
	Likelihood
	<input name="likelihood-location">
	<input name="likelihood-scale">

	<br>
 	<input type="submit"> 
	</form>'''
	return form

@app.route("/", methods=['POST'])
def hello():
	inp = request.form
	plot = bayes.plot_normals_flask(inp)

	return plot

if __name__ == "__main__":
    app.run()

