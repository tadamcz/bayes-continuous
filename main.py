from flask import Flask, render_template, request
from flask import request


from flask_wtf import FlaskForm
from wtforms import StringField, DecimalField, FormField, BooleanField, SelectField
from wtforms.validators import Regexp, NumberRange, EqualTo, DataRequired, InputRequired

import bayes

app = Flask(__name__)
app.secret_key = '85471922274287851509 97062761986949020795 57366896783488140597 40749154936961460411 00411694272886184644 96318878629748624589 43282652160909407164 81507837348880085650 94716104893291003189 95680230903613490699'

class TwoParamsForm(FlaskForm):
	param1 = DecimalField()
	param2 = DecimalField(validators=[NumberRange(min=1,max=2,message='no!')])

class DistrForm(FlaskForm):
	select_distribution_family = SelectField(choices=['normal','lognormal','beta'])
	normal = FormField(TwoParamsForm)
	lognormal = FormField(TwoParamsForm)
	beta = FormField(TwoParamsForm)

class PercentileForm(FlaskForm):
	compute_percentiles_exact = BooleanField('Compute percentiles of posterior distribution (exact, may fail)')
	compute_percentiles_mcmc = BooleanField('Approximate percentiles of posterior distribution using MCMC')


class DistrForm2(FlaskForm):
	prior = FormField(DistrForm)
	likelihood = FormField(DistrForm)
	percentiles = FormField(PercentileForm)
 

def label_form(form):
	'''I couldn't figure it out without this silly boilerplate'''
	form.prior.normal.param1.label = "mean"
	form.prior.normal.param2.label = "sd"
	form.prior.lognormal.param1.label = "mu"
	form.prior.lognormal.param2.label = "sigma"

	form.prior.beta.param1.label = "alpha"
	form.prior.beta.param2.label = "beta"

	'''exactly the same thing but for the likelihood!'''
	form.likelihood.normal.param1.label = "mean"
	form.likelihood.normal.param2.label = "sd"

	form.likelihood.lognormal.param1.label = "mu"
	form.likelihood.lognormal.param2.label = "sigma"

	form.likelihood.beta.param1.label = "alpha"
	form.likelihood.beta.param2.label = "beta"

@app.route('/')
def submit():
	form = DistrForm2()
	label_form(form)
	return render_template('hw.html',form=form)

@app.route("/", methods=['POST'])
def hello():
	form = DistrForm2()
	label_form(form)
	form = render_template('hw.html',form=form)
	my_input = request.form
	plot = bayes.out_html(my_input)
	return form+plot


if __name__ == "__main__":
	app.run(debug=True)
