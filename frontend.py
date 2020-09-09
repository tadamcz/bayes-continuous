from flask import Flask, render_template
from flask import request, jsonify, redirect
from flask_wtf import FlaskForm, CSRFProtect
from wtforms import StringField, DecimalField, FormField, BooleanField, SelectField
from wtforms.validators import Regexp, NumberRange, EqualTo, DataRequired, InputRequired
from flask_executor import Executor

import backend
import random
import decimal
import json
from scipy import stats
import math

app = Flask(__name__)
app.secret_key = '85471922274287851509 97062761986949020795 57366896783488140597 40749154936961460411 00411694272886184644 96318878629748624589 43282652160909407164 81507837348880085650 94716104893291003189 95680230903613490699'
csrf = CSRFProtect(app)

executor = Executor(app)


class TwoParamsForm(FlaskForm):
    param1 = DecimalField()
    param2 = DecimalField()


class DistrForm(FlaskForm):
    family = SelectField(choices=['normal', 'lognormal', 'beta', 'uniform'])
    normal = FormField(TwoParamsForm)
    lognormal = FormField(TwoParamsForm)
    beta = FormField(TwoParamsForm)
    uniform = FormField(TwoParamsForm)


class DistrForm2(FlaskForm):
    prior = FormField(DistrForm)
    likelihood = FormField(DistrForm)
    graphrange = FormField(TwoParamsForm, "Override default settings for graph domain? (optional)")
    custompercentiles = StringField("Provide custom percentiles? (optional, comma-separated)")


# @csrf.exempt
# @app.route('/')
# def view_without_form_input():
# 	form = label_form(DistrForm2())
# 	if len(request.args)>0:
# 		url_input = request.args['data']
# 		url_input = json.loads(url_input)
# 	else:
# 		url_input = None
#
#
# 	# If URL parameters are provided (a more sophisticated version would check that the input is valid)
# 	if url_input:
# 		form = label_form(DistrForm2(data=url_input))
# 		link_to_this = link_to_this_string(url_input)
#
# 		url_input_parsed = parse_user_inputs(url_input)
# 		graph = backend.graph_out(url_input_parsed)
# 		thread_id_exact = str(random.randint(0, 10000))
# 		executor.submit_stored(thread_id_exact, backend.percentiles_out, url_input_parsed)
#
#
# 		return render_template('index.html', form=form, graph=graph, thread_id_exact=thread_id_exact,
# 							   check_on_background_task=1, link_to_this=link_to_this)
# 	else:
# 		return render_template('index.html', form=form, check_on_background_task=0, thread_id_exact=None)
#
# @csrf.exempt
# @app.route("/", methods=['POST'])
# def input_and_output_view():
# 	form = DistrForm2()
# 	label_form(form)
#
# 	link_to_this = link_to_this_string(form.data,remove_csrf=True)
#
# 	form_input_parsed = parse_user_inputs(form.data)
# 	graph = backend.graph_out(form_input_parsed)
#
# 	thread_id = str(random.randint(0, 10000))
# 	executor.submit_stored(thread_id, backend.percentiles_out, form_input_parsed)
#
#
# 	return render_template('index.html', form=form, graph=graph, thread_id_exact=thread_id,
# 						   check_on_background_task=1, link_to_this=link_to_this)

@csrf.exempt
@app.route("/", methods=['GET', 'POST'])
def index():
    form = DistrForm2()
    label_form(form)
    user_input_given = False
    if request.method == 'GET':
        if len(request.args) > 0:
            url_input = request.args['data']
            url_input = json.loads(url_input)
            user_input_given = True
        else:
            url_input = None

        # If URL parameters are provided (a more sophisticated version would check that the input is valid)
        if url_input:
            form = label_form(DistrForm2(data=url_input))
            link_to_this_string = create_link_to_this_string(url_input)
            user_input_parsed = parse_user_inputs(url_input)

    if request.method == 'POST':
        user_input_given = True
        link_to_this_string = create_link_to_this_string(form.data, remove_csrf=True, convert_decimal_to_float=True)
        user_input_parsed = parse_user_inputs(form.data)

    if user_input_given:
        graph = backend.graph_out(user_input_parsed)
        thread_id_exact = str(random.randint(0, 10000))
        executor.submit_stored(thread_id_exact, backend.percentiles_out, user_input_parsed)

        return render_template('index.html', form=form, graph=graph, thread_id_exact=thread_id_exact,
                               check_on_background_task=1, link_to_this=link_to_this_string)
    else:
        return render_template('index.html', form=form, check_on_background_task=0, thread_id_exact=None)


@csrf.exempt
@app.route('/get-result/<thread_id>')
def get_result(thread_id):
    if not executor.futures.done(thread_id):
        return jsonify({'status': executor.futures._state(thread_id)})
    future = executor.futures.pop(thread_id)
    return jsonify({'status': 'done', 'result': future.result()})


def label_form(form):
    for obj in [form.prior, form.likelihood]:
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


def recursively_convert_decimal_to_float(dictionary):
    for key, value in dictionary.items():
        if type(value) == decimal.Decimal:
            dictionary[key] = float(value)
        if type(value) is dict:
            recursively_convert_decimal_to_float(value)


def create_link_to_this_string(dictionary, remove_csrf=False, convert_decimal_to_float=False):
    if remove_csrf:
        recursively_remove_csrf(dictionary)
    if convert_decimal_to_float:
        recursively_convert_decimal_to_float(dictionary)
    return '/?data=' + json.dumps(dictionary)


def parse_user_inputs(dictionary):
    def recursively_convert_Decimal_to_float(dictionary):
        for key in dictionary:
            if type(dictionary[key]) is decimal.Decimal:
                dictionary[key] = float(dictionary[key])
            if type(dictionary[key]) is dict:
                recursively_convert_Decimal_to_float(dictionary[key])

    recursively_convert_Decimal_to_float(dictionary)
    print("User input:", json.dumps(dictionary, indent=4))

    def parse_prior_likelihood(dictionary, p_or_l):
        if dictionary[p_or_l]['family'] == 'normal':
            scipy_distribution_object = stats.norm(loc=dictionary[p_or_l]['normal']['param1'],
                                                   scale=dictionary[p_or_l]['normal']['param2'])

        if dictionary[p_or_l]['family'] == 'lognormal':
            scipy_distribution_object = stats.lognorm(scale=math.exp(dictionary[p_or_l]['lognormal']['param1']),
                                                      s=dictionary[p_or_l]['lognormal']['param2'])

        if dictionary[p_or_l]['family'] == 'beta':
            scipy_distribution_object = stats.beta(dictionary[p_or_l]['beta']['param1'],
                                                   dictionary[p_or_l]['beta']['param2'])

        if dictionary[p_or_l]['family'] == 'uniform':
            loc = dictionary[p_or_l]['uniform']['param1']
            scale = dictionary[p_or_l]['uniform']['param2'] - loc
            scipy_distribution_object = stats.uniform(loc, scale)

        return scipy_distribution_object

    prior = parse_prior_likelihood(dictionary, 'prior')
    likelihood = parse_prior_likelihood(dictionary, 'likelihood')

    override_graph_range = False
    if dictionary['graphrange']['param1'] is not None and dictionary['graphrange']['param2'] is not None:
        override_graph_range = dictionary['graphrange']['param1'], dictionary['graphrange']['param2']

    custom_percentiles = False
    dictionary['custompercentiles'] = dictionary['custompercentiles'].replace(' ', '')  # sanitize input
    if dictionary['custompercentiles'] != '':
        custom_percentiles = dictionary['custompercentiles']
        custom_percentiles = custom_percentiles.split(',')
        custom_percentiles = [float(p) for p in custom_percentiles]
        custom_percentiles = [p for p in custom_percentiles if 0 < p < 1]

    return {'prior': prior,
            'likelihood': likelihood,
            'override_graph_range': override_graph_range,
            'custom_percentiles': custom_percentiles
            }


if __name__ == "__main__":
    app.run(debug=True)
