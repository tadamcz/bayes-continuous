import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy import integrate
from scipy import optimize
import sys
import time
import decimal

import mpld3

ppf_tol = 0.01

def intersect_intervals(two_tuples):
	d1 , d2 = two_tuples

	d1_left,d1_right = d1[0],d1[1]
	d2_left,d2_right = d2[0],d2[1]

	if d1_right < d2_left or d2_right < d2_left:
		raise ValueError("the distributions have no overlap")
	
	intersect_left,intersect_right = max(d1_left,d2_left),min(d1_right,d2_right)

	return intersect_left,intersect_right

def extremeties_intervals(two_tuples):
	d1,d2 = two_tuples

	d1_left, d1_right = d1[0], d1[1]
	d2_left, d2_right = d2[0], d2[1]

	extreme_left = min(d1_left,d1_right,d2_left,d2_right)
	extreme_right = max(d1_left,d1_right,d2_left,d2_right)

	return extreme_left,extreme_right

def split_integral(f,splitpoint,integrate_to,support=(-np.inf,np.inf)):
	'''
	https://stackoverflow.com/questions/47193045/why-does-integrate-quadlambda-x-xexp-x2-2-sqrt2pi-0-0-100000-give-0
	https://stackoverflow.com/questions/34877147/limits-of-quad-integration-in-scipy-with-np-inf

	if you try

	for x in range(100):
		print('cdf(',x,') = ',distr.cdf(x))

	CDF goes to 1 and then becomes
	a tiny value or 0. Due to problem of integrating over an area that
	is mostly 0. See stackoverflow links above.

	This creates problems when trying to use numerical equation solvers
	on the CDF. e.g. a bisection algo will first try CDF(very_large_number)
	and this will return 0.

	you can point quad to the region of the function
	where the peak(s) is/are with by supplying the points argument
	(points where 'where local difficulties of the integrand may occur'
	according to the documentation)

	But you can't pass points when one of the integration bounds
	is infinite.

	My solution: do the integration in two parts.
	First the left, then the right.
	Won't work for every function, but should cover many cases.
	'''
	a,b = support[0],support[1]
	if integrate_to < splitpoint:
		# just return the integral normally
		return integrate.quad(f,a,integrate_to)[0]
	else:
		integral_left = integrate.quad(f, a, splitpoint)[0]
		integral_right = integrate.quad(f, splitpoint, integrate_to)[0]
		return integral_left + integral_right

class Posterior_scipyrv(stats.rv_continuous):
	def __init__(self,d1,d2):
		super(Posterior_scipyrv, self).__init__()

		self.d1= d1
		self.d2= d2

		self.cdf_lookup = {}

		self.xtol = ppf_tol #The tolerance for fixed point calculation for generic ppf.

		'''
		defining the support of the product pdf is important
		because, when we use a numerical equation solver on the CDF,
		it will only need to look for solutions in the support, instead
		of on the entire real line.
		'''
		a1, b1 = d1.support()

		a2,b2 = d2.support()
		
		self.a , self.b = intersect_intervals([(a1,b1),(a2,b2)])

		'''
		the mode is used in my custom definition of _cdf() below.
		it's important that we don't run optimize.fmin every time cdf 
		is called, so I run it during init.
		'''
		initial_guess_for_mode = (self.d1.expect()+self.d2.expect())/2
		self.mode = initial_guess_for_mode # could be improved

		'''find normalization constant in init, so don't have to run integration every time'''
		self.normalization_constant = split_integral(f=self.unnormalized_pdf,splitpoint=self.mode,integrate_to=self.b,
													 support=self.support())

	def unnormalized_pdf(self,x):
		return self.d1.pdf(x) * self.d2.pdf(x)

	def _pdf(self,x):
		return self.unnormalized_pdf(x)/self.normalization_constant

	def neg_pdf(self,x):
		return -self.pdf(x)

	def _cdf(self,x):
		for x_lookup in self.cdf_lookup:
			if x_lookup < x and np.around(self.cdf_lookup[x_lookup],5)==1.0:
				return 1
		ret = split_integral(f=self.pdf,splitpoint=self.mode,integrate_to=x,support=self.support())
		self.cdf_lookup[float(x)] = ret
		return ret

	def ppf_with_bounds(self, q, leftbound, rightbound):
		'''
		wraps scipy ppf function
		https://github.com/scipy/scipy/blob/4c0fd79391e3b2ec2738bf85bb5dab366dcd12e4/scipy/stats/_distn_infrastructure.py#L1681-L1699
		'''

		factor = 10.
		left, right = self._get_support()

		if leftbound is not None:
			left = leftbound
		if rightbound is not None:
			right = rightbound

		if np.isinf(left):
			left = min(-factor, right)
			while self._ppf_to_solve(left, q) > 0.:
				left, right = left * factor, left
		# left is now such that cdf(left) <= q
		# if right has changed, then cdf(right) > q

		if np.isinf(right):
			right = max(factor, left)
			while self._ppf_to_solve(right, q) < 0.:
				left, right = right, right * factor
		# right is now such that cdf(right) >= q

		return optimize.brentq(self._ppf_to_solve,
							   left, right, args=q, xtol=self.xtol)


	def compute_percentiles_exact(self, percentiles_list):
		start = time.time()
		result = {}
		print('Running compute_percentiles_exact. Support: ', self.support(), file=sys.stderr)
		percentiles_list.sort()
		percentiles_reordered = sum(zip(percentiles_list,reversed(percentiles_list)), ())[:len(percentiles_list)] #https://stackoverflow.com/a/17436999/8010877

		def get_bounds(dict, p):
			keys = list(dict.keys())
			keys.append(p)
			keys.sort()
			i = keys.index(p)
			if i != 0:
				leftbound = dict[keys[i - 1]]
			else:
				leftbound = None
			if i != len(keys) - 1:
				rightbound = dict[keys[i + 1]]
			else:
				rightbound = None
			return leftbound, rightbound

		for p in percentiles_reordered:
			print("trying to compute the", p, "th percentile")
			try:
				leftbound , rightbound = get_bounds(result,p)
				res = self.ppf_with_bounds(p,leftbound,rightbound)
				result[p] = np.around(res,2)
			except RuntimeError as e:
				result[p] = e

		sorted_result = {key:value for key,value in sorted(result.items())}

		end = time.time()

		description_string = 'Computed in ' + str(np.around(end - start, 1)) + ' seconds'
		return {'result': sorted_result, 'runtime': description_string}

def parse_user_inputs(dictionary):
	def recursively_convert_Decimal_to_float(dictionary):
		for key in dictionary:
			if type(dictionary[key]) is decimal.Decimal:
				dictionary[key] = float(dictionary[key])
			if type(dictionary[key]) is dict:
				recursively_convert_Decimal_to_float(dictionary[key])

	recursively_convert_Decimal_to_float(dictionary)

	for p_or_l in ['prior','likelihood']:
		if dictionary[p_or_l]['family'] == 'normal':
			distr = stats.norm(loc=dictionary[p_or_l]['normal']['param1'], scale=dictionary[p_or_l]['normal']['param2'])
		if dictionary[p_or_l]['family'] == 'lognormal':
			distr = stats.lognorm(scale=math.exp(dictionary[p_or_l]['lognormal']['param1']), s=dictionary[p_or_l]['lognormal'][
				'param2'])
		if dictionary[p_or_l]['family'] == 'beta':
			distr = stats.beta(dictionary[p_or_l]['beta']['param1'], dictionary[p_or_l]['beta']['param2'])
		if dictionary[p_or_l]['family'] == 'uniform':
			loc = dictionary[p_or_l]['uniform']['param1']
			scale = dictionary[p_or_l]['uniform']['param2'] - loc
			distr = stats.uniform(loc, scale)

		if p_or_l == 'prior':
			prior = distr
		if p_or_l == 'likelihood':
			likelihood = distr


	override_graph_range = False
	if dictionary['graphrange']['param1'] is not None and dictionary['graphrange']['param2'] is not None:
		override_graph_range = (dictionary['graphrange']['param1'], dictionary['graphrange']['param2'])

	custom_percentiles = False
	dictionary['custompercentiles'] = dictionary['custompercentiles'].replace(' ','')
	if dictionary['custompercentiles'] != '':
		custom_percentiles = dictionary['custompercentiles']
		custom_percentiles = custom_percentiles.split(',')
		custom_percentiles = [float(p) for p in custom_percentiles]
		custom_percentiles = [p for p in custom_percentiles if 0<p<1]

	return {'prior':prior,
			'likelihood':likelihood,
			'override_graph_range':override_graph_range,
			'custom_percentiles':custom_percentiles
			}

def plot_pdfs(dict_of_dists,x_from,x_to):
	x_from ,x_to = float(x_from),float(x_to)
	x = np.linspace(x_from,x_to,100)

	fig, ax = plt.subplots()
	for dist in dict_of_dists:
		ax.plot(x,dict_of_dists[dist].pdf(x),label=dist)
	ax.legend()
	ax.set_xlabel("X")
	ax.set_ylabel("Probability density")
	return fig

def plot_pdfs_bayes_update(prior,likelihood,posterior,x_from=-50,x_to=50):
	prior_string = "P(X)"
	likelihood_string = "P(E|X)"
	posterior_string = "P(X|E)"

	plot = plot_pdfs({	prior_string:prior,
						likelihood_string:likelihood,
						posterior_string:posterior}
						,x_from,x_to)
	return plot



def intelligently_set_graph_domain(prior,likelihood):

	prior_mean,prior_sd = prior.mean(),prior.std()
	p = 0.1
	prior_range = prior.ppf(p) , prior.ppf(1-p)
	likelihood_range = likelihood.ppf(p) , likelihood.ppf(1-p)

	ranges = extremeties_intervals([prior_range,likelihood_range])

	supports = intersect_intervals([prior.support(),likelihood.support()])

	domain = intersect_intervals([supports,ranges])

	buffer = 0.1
	buffer = abs(buffer*(domain[1]-domain[0]))

	domain = domain[0]-buffer,domain[1]+buffer

	return domain

plt.rcParams.update({'font.size': 16})
def graph_out(dict):
	# parse inputs
	user_inputs = parse_user_inputs(dict)
	prior = user_inputs['prior']
	likelihood = user_inputs['likelihood']
	override_graph_range = user_inputs['override_graph_range']
	
	# compute posterior pdf
	s = time.time()
	posterior = Posterior_scipyrv(prior,likelihood)
	e = time.time()
	print(e-s,'seconds to get posterior pdf',file=sys.stderr)

	# Plot
	if override_graph_range:
		x_from,x_to = override_graph_range
	else:
		x_from , x_to = intelligently_set_graph_domain(prior,likelihood)

	s = time.time()
	plot = plot_pdfs_bayes_update(prior,likelihood,posterior,x_from=x_from,x_to=x_to)
	plot = mpld3.fig_to_html(plot)
	e = time.time()
	print(e-s,'seconds to make plot', file=sys.stderr)

	# Expected value
	ev = np.around(posterior.expect(),2)
	ev_string = 'Posterior expected value: '+str(ev)+'<br>'
	
	return plot+ev_string

def percentiles_out_exact(dict):
	# Parse inputs
	user_inputs = parse_user_inputs(dict)

	prior = user_inputs['prior']
	likelihood = user_inputs['likelihood']	
	
	# compute posterior pdf
	posterior = Posterior_scipyrv(prior,likelihood)

	#percentiles
	percentiles_exact_string = ''

	if user_inputs['custom_percentiles']:
		p = user_inputs['custom_percentiles']
	else:
		p = [0.1,0.25,0.5,0.75,0.9]
	percentiles_exact = posterior.compute_percentiles_exact(p)

	percentiles_exact_string = percentiles_exact['runtime'] +'<br>'
	for x in percentiles_exact['result']:
		percentiles_exact_string += str(x) + ', ' + str(percentiles_exact['result'][x]) + '<br>'
	return percentiles_exact_string