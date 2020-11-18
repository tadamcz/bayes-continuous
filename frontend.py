from flask import Flask, render_template
from flask import request, jsonify, redirect
from flask_wtf import FlaskForm
from wtforms import StringField, DecimalField, FormField, BooleanField, SelectField
from wtforms.validators import Regexp, NumberRange, EqualTo, DataRequired, InputRequired, Optional, Required
from flask_executor import Executor

import backend
import random
import decimal
import json
from scipy import stats
import math

app = Flask(__name__)
app.config['WTF_CSRF_ENABLED'] = False  # not needed, there are no user accounts

executor = Executor(app)


class TwoParamsForm(FlaskForm):
    param1 = DecimalField(validators=[Optional()])
    param2 = DecimalField(validators=[Optional()])

class DistrForm(FlaskForm):
    family = SelectField(choices=[('normal','Normal'), ('lognormal','Lognormal'), ('beta','Beta'), ('uniform','Uniform')])
    normal = FormField(TwoParamsForm)
    lognormal = FormField(TwoParamsForm)
    beta = FormField(TwoParamsForm)
    uniform = FormField(TwoParamsForm)

    def validate(self):
        if not super(DistrForm, self).validate():
            return False
        if self.family.data == 'normal':
            distribution_to_check = self.normal
        if self.family.data == 'lognormal':
            distribution_to_check = self.lognormal
        if self.family.data == 'beta':
            distribution_to_check = self.beta
        if self.family.data == 'uniform':
            distribution_to_check = self.uniform

        if all(x is not None for x in [distribution_to_check.param1.data,distribution_to_check.param2.data]):
            return True
        else:
            distribution_to_check.param1.errors.append('Distribution parameters are required')
            return False


class DistrForm2(FlaskForm):
    prior = FormField(DistrForm)
    likelihood = FormField(DistrForm)
    graphrange = FormField(TwoParamsForm, "Override default settings for graph domain? (optional)")
    custompercentiles = StringField("Provide custom percentiles? (optional, comma-separated, e.g. 0.2,0.76)")


@app.route("/", methods=['GET', 'POST'])
def index():
    form = DistrForm2()
    label_form(form)
    user_input_given = False
    user_input_valid = False
    if request.method == 'GET':
        if len(request.args) > 0:
            url_input = request.args['data']
            url_input = json.loads(url_input)
            user_input_given = True
        else:
            url_input = None

        if url_input:
            form = label_form(DistrForm2(data=url_input))
            link_to_this_string = create_link_to_this_string(url_input)
            if form.validate():
                user_input_parsed = parse_user_inputs(url_input)
                user_input_valid = True

    if request.method == 'POST':
        user_input_given = True
        link_to_this_string = create_link_to_this_string(form.data, convert_decimal_to_float=True)
        if form.validate():
            user_input_parsed = parse_user_inputs(form.data)
            user_input_valid = True

    if user_input_given and user_input_valid:
        graph = backend.graph_out(user_input_parsed)
        thread_id_exact = str(random.randint(0, 10000))
        executor.submit_stored(thread_id_exact, backend.percentiles_out, user_input_parsed)

        return render_template('index.html', form=form, graph=graph, thread_id_exact=thread_id_exact,
                               check_on_background_task=1, link_to_this=link_to_this_string)
    else:
        return render_template('index.html', form=form, check_on_background_task=0, thread_id_exact=None)


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


def recursively_convert_decimal_to_float(dictionary):
    for key, value in dictionary.items():
        if type(value) == decimal.Decimal:
            dictionary[key] = float(value)
        if type(value) is dict:
            recursively_convert_decimal_to_float(value)


def create_link_to_this_string(dictionary, convert_decimal_to_float=False):
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
