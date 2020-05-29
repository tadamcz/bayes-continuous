from flask import Flask
from flask import request

import bayes

app = Flask(__name__)
app.secret_key = '85471922274287851509 97062761986949020795 57366896783488140597 40749154936961460411 00411694272886184644 96318878629748624589 43282652160909407164 81507837348880085650 94716104893291003189 95680230903613490699'



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

