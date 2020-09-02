from flask import Flask, render_template, request
from flask import request, jsonify, redirect


from flask_wtf import FlaskForm
from wtforms import StringField, DecimalField, FormField, BooleanField, SelectField
from wtforms.validators import Regexp, NumberRange, EqualTo, DataRequired, InputRequired

import bayes
from flask_executor import Executor
import random

from decimal import Decimal

import json

app = Flask(__name__)
app.secret_key = '85471922274287851509 97062761986949020795 57366896783488140597 40749154936961460411 00411694272886184644 96318878629748624589 43282652160909407164 81507837348880085650 94716104893291003189 95680230903613490699'
executor = Executor(app)

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
	custompercentiles = StringField("Provide custom percentiles? (optional, comma-separated)")

def label_form(form):
	for obj in [form.prior,form.likelihood]:
		obj.normal.param1.label = "mean"
		obj.normal.param2.label = "sd"

		obj.lognormal.param1.label = "mu"
		obj.lognormal.param2.label = "sigma"

		obj.beta.param1.label = "alpha"
		obj.beta.param2.label = "beta"

	form.graphrange.param1.label = "From"
	form.graphrange.param2.label = "To"
	return form

def recursively_remove_csrf(dictionary):
	dictionary.pop('csrf_token')
	for key in dictionary:
		if type(dictionary[key]) is dict:
			recursively_remove_csrf(dictionary[key])

def link_to_this_string(dictionary,remove_csrf=False):
	if remove_csrf:
		recursively_remove_csrf(dictionary)
	return '/?data=' + json.dumps(dictionary)


@app.route('/')
def view_without_form_input():
	form = label_form(DistrForm2())
	if len(request.args)>0:
		url_input = request.args['data']
		url_input = json.loads(url_input)
	else:
		url_input = None


	# If URL parameters are provided (a more sophisticated version would check that the input is valid)
	if url_input:
		form = label_form(DistrForm2(data=url_input))
		link_to_this = link_to_this_string(url_input)

		graph = bayes.graph_out(url_input)
		thread_id_exact = str(random.randint(0, 10000))
		executor.submit_stored(thread_id_exact, bayes.percentiles_out, url_input)


		return render_template('index.html', form=form, graph=graph, thread_id_exact=thread_id_exact,
							   check_on_background_task=1, link_to_this=link_to_this)
	else:
		return render_template('index.html', form=form, check_on_background_task=0, thread_id_exact=None)

@app.route("/", methods=['POST'])
def input_and_output_view():
	form = DistrForm2()
	label_form(form)

	form_input = form.data
	graph = bayes.graph_out(form_input)
	thread_id_exact = str(random.randint(0, 10000))
	executor.submit_stored(thread_id_exact, bayes.percentiles_out, form_input)

	link_to_this = link_to_this_string(form_input,remove_csrf=True)


	return render_template('index.html', form=form, graph=graph, thread_id_exact=thread_id_exact,
						   check_on_background_task=1, link_to_this=link_to_this)

@app.route('/get-result/<thread_id>')
def get_result(thread_id):
	if not executor.futures.done(thread_id):
		return jsonify({'status': executor.futures._state(thread_id)})
	future = executor.futures.pop(thread_id)
	return jsonify({'status': 'done', 'result': future.result()})


if __name__ == "__main__":
	app.run(debug=True)
