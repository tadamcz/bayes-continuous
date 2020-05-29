from flask import Flask
from flask import request

import bayes

app = Flask(__name__)


form = '''<form method="POST">
	<b>Prior</b>

	<br>Normal
	<input name="norm-prior-location" placeholder="Location">
	<input name="norm-prior-scale" placeholder="Scale">

	<br>Lognormal
	<input name="lognorm-prior-mu" placeholder="Mu">
	<input name="lognorm-prior-sigma" placeholder="Sigma">

	<br>Beta
	<input name="beta-prior-alpha" placeholder="Alpha">
	<input name="beta-prior-beta" placeholder="Beta">

	<br>
	<br><b>Likelihood</b> (Must be Normal)
	<input name="likelihood-location" placeholder="Mean">
	<input name="likelihood-scale" placeholder="Standard error">

	<br>
 	<input type="submit"> 
	</form>'''

@app.route('/')
def my_form():
	return form

@app.route("/", methods=['POST'])
def hello():
	my_input = request.form

	plot = bayes.plot_out_html(my_input)
	return form+plot

if __name__ == "__main__":
    app.run(debug=True)

