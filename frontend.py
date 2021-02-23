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
import numpy as np

app = Flask(__name__)
app.config['WTF_CSRF_ENABLED'] = False  # not needed, there are no user accounts

executor = Executor(app)


class TwoParamsForm(FlaskForm):
    param1 = DecimalField(validators=[Optional()])
    param2 = DecimalField(validators=[Optional()])

class FourParamsForm(FlaskForm):
    param1 = DecimalField(validators=[Optional()])
    param2 = DecimalField(validators=[Optional()])
    param3 = DecimalField(validators=[Optional()])
    param4 = DecimalField(validators=[Optional()])

class DistrFrom(FlaskForm):
    # Only used as a superclass for PriorForm and LikelihoodForm below
    def validate(self):
        if not super(DistrFrom, self).validate():
            return False
        if self.family.data == 'normal':
            try:
                if self.normal_95_ci_bool.data:
                    distribution_to_check = self.normal_95_ci
                else:
                    distribution_to_check = self.normal
            except AttributeError:
                distribution_to_check = self.normal
        if self.family.data == 'lognormal':
            try:
                if self.lognormal_95_ci_bool.data:
                    distribution_to_check = self.lognormal_95_ci
                else:
                    distribution_to_check = self.lognormal
            except AttributeError:
                distribution_to_check = self.lognormal
        if self.family.data == 'beta':
            distribution_to_check = self.beta
        if self.family.data == 'uniform':
            distribution_to_check = self.uniform
        if self.family.data == 'binomial':
            distribution_to_check = self.binomial
        if self.family.data == 'diff_log_betas':
            distribution_to_check = self.diff_log_betas
        if self.family.data == 'ratio_betas':
            distribution_to_check = self.ratio_betas

        if all(x is not None for x in [distribution_to_check.param1.data,distribution_to_check.param2.data]):
            return True
        else:
            distribution_to_check.param1.errors.append('Distribution parameters are required')
            return False

def label_form(form,i,family_list):
    if 'normal' in family_list:
        form.normal.param1.label = "\( \mu_"+str(i)+"\)"
        form.normal.param2.label = "\( \sigma_"+str(i)+"\)"

    if 'lognormal' in family_list:
        form.lognormal.param1.label = "\( \mu_"+str(i)+"\)"
        form.lognormal.param2.label = "\( \sigma_"+str(i)+"\)"

    if 'beta' in family_list:
        form.beta.param1.label = "\( \\alpha_"+str(i)+"\)"
        form.beta.param2.label = "\( \\beta_"+str(i)+"\)"

    if 'binomial' in family_list:
        form.binomial.param1.label = "successes \(s\)"
        form.binomial.param2.label = "failures \(f\)"

    if 'diff_log_betas' in family_list:
        form.diff_log_betas.param1.label = 'Numerator Beta, parameter \( a_1 \)'
        form.diff_log_betas.param2.label = 'Numerator Beta, parameter \( b_1 \)'
        form.diff_log_betas.param3.label = 'Denominator Beta, parameter \( a_2 \)'
        form.diff_log_betas.param4.label = 'Denominator Beta, parameter \( b_2 \)'

    if 'ratio_betas' in family_list:
        form.ratio_betas.param1.label = 'Numerator Beta, parameter \( a_1 \)'
        form.ratio_betas.param2.label = 'Numerator Beta, parameter \( b_1 \)'
        form.ratio_betas.param3.label = 'Denominator Beta, parameter \( a_2 \)'
        form.ratio_betas.param4.label = 'Denominator Beta, parameter \( b_2 \)'
    return form

class PriorForm(DistrFrom):
    family = SelectField(choices=[('normal','Normal'), ('lognormal','Lognormal'), ('beta','Beta'), ('uniform','Uniform'),('diff_log_betas','log(Beta) - log(Beta)'),('ratio_betas','Beta/Beta')])
    normal = FormField(TwoParamsForm)
    lognormal = FormField(TwoParamsForm)
    beta = FormField(TwoParamsForm)
    uniform = FormField(TwoParamsForm)
    diff_log_betas = FormField(FourParamsForm)
    ratio_betas = FormField(FourParamsForm)
    def __init__(self, *args, **kwargs):
        super(PriorForm, self).__init__(*args, **kwargs)
        self = label_form(self,i=0,family_list=['normal','lognormal','beta','uniform','diff_log_betas', 'ratio_betas'])




class LikelihoodForm(DistrFrom):
    family = SelectField(choices=[('normal','Normal'), ('lognormal','Lognormal'), ('beta','Beta'), ('uniform','Uniform'),
                                  ('binomial','Binomial (as a function of success probability)')])
    normal = FormField(TwoParamsForm)
    normal_95_ci_bool = BooleanField("Use 95% interval mode?")
    normal_95_ci = FormField(TwoParamsForm)
    lognormal = FormField(TwoParamsForm)
    lognormal_95_ci_bool = BooleanField("Use 95% interval mode?")
    lognormal_95_ci = FormField(TwoParamsForm)
    beta = FormField(TwoParamsForm)
    uniform = FormField(TwoParamsForm)
    binomial = FormField(TwoParamsForm)

    def __init__(self, *args, **kwargs):
        super(LikelihoodForm, self).__init__(*args, **kwargs)
        self = label_form(self,i=1,family_list=['normal','lognormal','beta','uniform','binomial'])
        self.normal_95_ci.param1.label = '2.5%'
        self.normal_95_ci.param2.label = '97.5%'
        self.lognormal_95_ci.param1.label = '2.5%'
        self.lognormal_95_ci.param2.label = '97.5%'



class DistrForm2(FlaskForm):
    prior = FormField(PriorForm)
    likelihood = FormField(LikelihoodForm)
    graphrange = FormField(TwoParamsForm, "Override default settings for graph domain? (optional)")
    custompercentiles = StringField("Provide custom percentiles? (optional, comma-separated, e.g. 0.2,0.76)")
    def __init__(self, *args, **kwargs):
        super(DistrForm2, self).__init__(*args, **kwargs)
        self.graphrange.param1.label = "From"
        self.graphrange.param2.label = "To"


@app.route("/", methods=['GET', 'POST'])
def index():
    form = DistrForm2()
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
            form = DistrForm2(data=url_input)
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
        posterior = backend.Posterior(user_input_parsed['prior'],user_input_parsed['likelihood'],user_input_parsed)
        graph = posterior.graph_out()
        thread_id_exact = str(random.randint(0, 10000))
        executor.submit_stored(thread_id_exact, posterior.distribution_information_out)

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
    # print("User input:", json.dumps(dictionary, indent=4))

    def parse_prior_likelihood(dictionary, p_or_l):
        if dictionary[p_or_l]['family'] == 'normal':
            if p_or_l == 'likelihood' and dictionary['likelihood']['normal_95_ci_bool']:
                x1, x2 = dictionary['likelihood']['normal_95_ci']['param1'],dictionary['likelihood']['normal_95_ci']['param2']
                loc,scale = backend.normal_parameters(x1,2.5/100,x2,97.5/100)
            else:
                loc = dictionary[p_or_l]['normal']['param1']
                scale = dictionary[p_or_l]['normal']['param2']

            scipy_distribution_object = stats.norm(loc=loc, scale=scale)

        if dictionary[p_or_l]['family'] == 'lognormal':
            if p_or_l == 'likelihood' and dictionary['likelihood']['lognormal_95_ci_bool']:
                x1, x2 = dictionary['likelihood']['lognormal_95_ci']['param1'],dictionary['likelihood']['lognormal_95_ci']['param2']
                x1, x2 = np.log(x1), np.log(x2)
                mu, sigma = backend.normal_parameters(x1,2.5/100,x2,97.5/100)
            else:
                mu = dictionary[p_or_l]['lognormal']['param1']
                sigma = dictionary[p_or_l]['lognormal']['param2']

            scipy_distribution_object = stats.lognorm(scale=np.exp(mu), s=sigma)

        if dictionary[p_or_l]['family'] == 'beta':
            scipy_distribution_object = stats.beta(dictionary[p_or_l]['beta']['param1'],
                                                   dictionary[p_or_l]['beta']['param2'])

        if dictionary[p_or_l]['family'] == 'uniform':
            loc = dictionary[p_or_l]['uniform']['param1']
            scale = dictionary[p_or_l]['uniform']['param2'] - loc
            scipy_distribution_object = stats.uniform(loc, scale)

        if dictionary[p_or_l]['family'] == 'binomial':
            successes = dictionary[p_or_l]['binomial']['param1']
            failures = dictionary[p_or_l]['binomial']['param2']
            trials = successes+failures
            binomial_pdf = lambda theta: stats.binom.pmf(successes,trials,theta)
            scipy_distribution_object = backend.CustomFromPDF(binomial_pdf,a=0,b=1)

        if dictionary[p_or_l]['family'] == 'diff_log_betas':
            a1 = dictionary[p_or_l]['diff_log_betas']['param1']
            b1 = dictionary[p_or_l]['diff_log_betas']['param2']
            a2 = dictionary[p_or_l]['diff_log_betas']['param3']
            b2 = dictionary[p_or_l]['diff_log_betas']['param4']

            scipy_distribution_object = backend.DiffLogBetas(a1, b1, a2, b2)

        if dictionary[p_or_l]['family'] == 'ratio_betas':
            a1 = dictionary[p_or_l]['ratio_betas']['param1']
            b1 = dictionary[p_or_l]['ratio_betas']['param2']
            a2 = dictionary[p_or_l]['ratio_betas']['param3']
            b2 = dictionary[p_or_l]['ratio_betas']['param4']

            scipy_distribution_object = backend.RatioBetas(a1, b1, a2, b2)

        if dictionary[p_or_l]['family'] == 'ratio_betas':
            scipy_distribution_object.use_log_transform = True
        else:
            scipy_distribution_object.use_log_transform = False

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
    app.run()
