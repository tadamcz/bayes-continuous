from flask import Flask, render_template, request
from flask import request, jsonify, redirect


from flask_wtf import FlaskForm
from wtforms import StringField, DecimalField, FormField, BooleanField, SelectField
from wtforms.validators import Regexp, NumberRange, EqualTo, DataRequired, InputRequired

import bayes
from flask_executor import Executor
import random
from sys import stderr

app = Flask(__name__)
app.secret_key = '85471922274287851509 97062761986949020795 57366896783488140597 40749154936961460411 00411694272886184644 96318878629748624589 43282652160909407164 81507837348880085650 94716104893291003189 95680230903613490699'
executor = Executor(app)

parameter_names = ['likelihood-family', 'prior-family',
				   'prior-normal-param1', 'prior-normal-param2' 'prior-lognormal-param1', 'prior-lognormal-param2',
				   'prior-beta-param1', 'prior-beta-param2', 'prior-uniform-param1', 'prior-uniform-param2',
				   'likelihood-normal-param1', 'likelihood-normal-param2', 'likelihood-lognormal-param1',
				   'likelihood-lognormal-param2', 'likelihood-beta-param1', 'likelihood-beta-param2',
				   'likelihood-uniform-param1', 'likelihood-uniform-param2','graphrange-param1','graphrange-param2']

class TwoParamsForm(FlaskForm):
	param1 = DecimalField()
	param2 = DecimalField()

class DistrForm(FlaskForm):
	family = SelectField(choices=['normal','lognormal','beta','uniform'])
	normal = FormField(TwoParamsForm)
	lognormal = FormField(TwoParamsForm)
	beta = FormField(TwoParamsForm)
	uniform = FormField(TwoParamsForm)


class DistrForm2(FlaskForm):
	prior = FormField(DistrForm)
	likelihood = FormField(DistrForm)
	graphrange = FormField(TwoParamsForm,"Override default settings for graph domain? (optional)")
 

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

	form.graphrange.param1.label = "From"
	form.graphrange.param2.label = "To"


@app.route('/')
def submit():
	form = DistrForm2()
	label_form(form)
	my_input = dict(request.args)


	if len(my_input) >=4:
		for x in parameter_names:
			if x not in my_input.keys():
				my_input[x] = ''
		graph = bayes.graph_out(my_input)
		thread_id_exact = str(random.randint(0, 10000))
		executor.submit_stored(thread_id_exact, bayes.percentiles_out_exact, my_input)

		return render_template('hw.html', form=form, graph=graph, thread_id_exact=thread_id_exact,
							   check_on_background_task=1)
	else:
		return render_template('hw.html',form=form,check_on_background_task=0,thread_id_exact=None)

@app.route("/", methods=['POST'])
def hello():
	form = DistrForm2()
	label_form(form)

	my_input = dict(request.form)
	graph = bayes.graph_out(my_input)
	thread_id_exact = str(random.randint(0, 10000))
	executor.submit_stored(thread_id_exact, bayes.percentiles_out_exact, my_input)

	link_to_this = '/?'
	for x in my_input:
		if 'csrf' not in x and my_input[x] !='':
			link_to_this += str(x) + '=' + str(my_input[x]) + '&'

	return render_template('hw.html',form=form,graph=graph,thread_id_exact=thread_id_exact,
						   check_on_background_task=1,link_to_this=link_to_this)

@app.route('/get-result/<thread_id>')
def get_result(thread_id):
	if not executor.futures.done(thread_id):
		return jsonify({'status': executor.futures._state(thread_id)})
	future = executor.futures.pop(thread_id)
	return jsonify({'status': 'done', 'result': future.result()})


if __name__ == "__main__":
	app.run(debug=True)
