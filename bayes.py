import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy import integrate
from scipy import optimize
import sys
import time

import mpld3
import emcee

def is_a_distribution(obj):
	if issubclass(type(obj),stats.rv_continuous):
		return True
	if isinstance(obj,stats._distn_infrastructure.rv_frozen):
		return True
	return False

def intersect_intervals(two_tuples):
	d1 , d2 = two_tuples

	d1_left,d1_right = d1[0],d1[1]
	d2_left,d2_right = d2[0],d2[1]

	if d1_right < d2_left or d2_right < d2_left:
		raise ValueError("the distributions have no overlap")
	
	intersect_left,intersect_right = max(d1_left,d2_left),min(d1_right,d2_right)

	return intersect_left,intersect_right

class Product_pdf(stats.rv_continuous):
	def __init__(self,d1,d2):
		super(Product_pdf,self).__init__()
		if not is_a_distribution(d1):
			raise TypeError("First argument must be a distribution")
		if type(d2) is not float and type(d2) is not int and not is_a_distribution(d2):
			raise TypeError("Second argument must be a distribution or a number")
		self.d1= d1
		self.d2= d2

		'''
		defining the support of the product pdf is important
		because, when we use a numerical equation solver on the CDF,
		it will only need to look for solutions in the support, instead
		of on the entire real line.
		'''
		a1, b1 = d1.support()

		if is_a_distribution(d2):
			a2,b2 = d2.support()
		else:
			a2,b2 = -np.inf,np.inf
		
		self.a , self.b = intersect_intervals([(a1,b1),(a2,b2)])

		'''
		the mode is used in my custom definition of _cdf() below.
		it's important that we don't run optimize.fmin every time cdf 
		is called, so I run it during init.
		'''
		initial_guess_for_mode = self.d1.expect()
		self.mode = optimize.fmin(self.neg_pdf,initial_guess_for_mode)


	def _pdf(self,x):
		if type(self.d2) is float:
			ret = self.d1.pdf(x)*self.d2
		if type(self.d2) is stats._distn_infrastructure.rv_frozen:
			ret = self.d1.pdf(x)*self.d2.pdf(x)
		return ret

	def neg_pdf(self,x):
		return -self.pdf(x)

	def _cdf(self,x):
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

		mode = self.mode
		a,b = self.support()

		if x<mode:
			# just return the cdf using normal method
			return super(Product_pdf,self)._cdf(x)
		else:
			integral_left = integrate.quad(self.pdf,a,mode)[0]
			integral_right = integrate.quad(self.pdf,mode,x)[0]
			return integral_left + integral_right


def normalize(distr):
	integral = integrate.quad(distr.pdf,-np.inf,np.inf)[0]
	ret = Product_pdf(distr,1/integral)
	return ret

def update(prior,likelihood):
	unnormalized_posterior = Product_pdf(prior,likelihood)
	posterior = normalize(unnormalized_posterior)
	return posterior

def parse_user_inputs(dict):
	dict = dict.to_dict()

	# debugging print('Received dict',dict, file=sys.stderr)

	for key in dict.keys():
		cond1 = 'likelihood' in key
		cond2 = 'prior' in key
		cond3 = 'graph_range' in key
		yes = cond1 or cond2 or cond3

		cond1 = 'percentiles' in key
		cond2 = 'family' in key
		cond3 = 'csrf' in key
		cond4 = len(dict[key])==0

		no = cond1 or cond2 or cond3 or cond4

		if yes and not no:
			# debugging print(key, file=sys.stderr)
			dict[key] = float(dict[key])
		
	
	if dict["prior-select_distribution_family"] == "normal":
		prior = stats.norm(loc = dict['prior-normal-param1'], scale =dict['prior-normal-param2'])

	elif dict["prior-select_distribution_family"] == "lognormal":
		prior = stats.lognorm(scale = math.exp(dict['prior-lognormal-param1']), s =dict['prior-lognormal-param2'])

	elif dict["prior-select_distribution_family"] == "beta":
		prior = stats.beta(dict["prior-beta-param1"],dict["prior-beta-param2"])

	elif dict["prior-select_distribution_family"] == "uniform":
		loc = dict["prior-uniform-param1"]
		scale = dict["prior-uniform-param2"] - loc
		prior = stats.uniform(loc,scale)

	'''Redundant, will refactor'''
	if dict["likelihood-select_distribution_family"] == "normal":
		likelihood = stats.norm(loc = dict['likelihood-normal-param1'], scale =dict['likelihood-normal-param2'])

	elif dict["likelihood-select_distribution_family"] == "lognormal":
		likelihood = stats.lognorm(scale = math.exp(dict['likelihood-lognormal-param1']), s =dict['likelihood-lognormal-param2'])

	elif dict["likelihood-select_distribution_family"] == "beta":
		likelihood = stats.beta(dict["likelihood-beta-param1"],dict["likelihood-beta-param2"])

	elif dict["likelihood-select_distribution_family"] == "uniform":
		loc = dict["likelihood-uniform-param1"]
		scale = dict["likelihood-uniform-param2"] - loc
		likelihood = stats.uniform(loc,scale)

	override_graph_range = False
	if dict['graph_range-param1'] !='' and dict['graph_range-param2'] !='':
		override_graph_range = (dict['graph_range-param1'],dict['graph_range-param2'])
	
	return {'prior':prior, 'likelihood':likelihood, 'override_graph_range':override_graph_range}

def plot_pdfs(dict_of_dists,x_from,x_to):
	if x_from:
		x = np.linspace(x_from,x_to,100)
	else:
		x = np.linspace(-50,50,100)



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

def mcmc_sample(distr,nwalkers=10,nruns=500):
	ndim = 1
	location_first_guess = distr.expect()
	p0 = np.random.normal(loc=location_first_guess,size=(nwalkers, ndim))
	# the default is to use a standard normal N(0,1) p0 = np.random.randn(nwalkers, ndim)

	def log_prob(x):
		if distr.pdf(x)>0:
			return math.log(distr.pdf(x))
		else:
			return -np.inf
	sampler = emcee.EnsembleSampler(nwalkers, 1, log_prob)

	# burn-in
	state = sampler.run_mcmc(p0, 1000)
	sampler.reset()
	
	# Main run
	sampler.run_mcmc(state, nruns)

	samples = sampler.get_chain(flat=True)[:, 0]
	return samples

def percentiles_from_list(list,percentiles_list):
	list.sort()
	l = len(list)
	ret = []
	for p in percentiles_list:
		index = math.floor(p*l)
		ret.append((p,list[index]))
	return ret

def mcmc_percentiles(distr,percentiles_list):
	nwalkers = 10 
	nruns = 500
	start = time.time()
	samples = mcmc_sample(distr,nwalkers,nruns)
	ret = percentiles_from_list(samples,percentiles_list)
	end = time.time()
	description_string = 'Approximated using emcee with '+str(nwalkers*nruns)+' samples, in '+str(np.around(end-start,1))+' seconds'
	return {'result': ret, 'runtime':description_string}

def compute_percentiles_exact(distr,percentiles_list):
	start = time.time()
	result = {}
	print('Running compute_percentiles_exact. Support: ',distr.support(),file=sys.stderr)
	for p in percentiles_list:
		print("trying to compute the",p,"th percentile")
		try:
			result[p] = np.around(distr.ppf(p),2)
		except RuntimeError as e:
			result[p] = e
	end = time.time()
	description_string = 'Computed in '+str(np.around(end-start,1))+' seconds'
	return {'result':result,'runtime':description_string}

def intelligently_set_graph_domain(distr):
	left_default,right_default = -50,50


	d_left,d_right = distr.support()

	left_default,right_default = intersect_intervals([(left_default,right_default),(d_left,d_right)])

	mean = distr.expect()
	
	y_val = 0.01
	max_domain = 1e6

	def f(x):
		return distr.pdf(x)-y_val
	

	try:
		s = time.time()
		left_root = optimize.root_scalar(f,rtol=.3,bracket=(-max_domain,mean))
		right_root = optimize.root_scalar(f,rtol=.3,bracket=(mean,max_domain))
		e = time.time()
		print(e-s,'seconds to find roots', file=sys.stderr)
		print(left_root,file=sys.stderr)
		print(right_root,file=sys.stderr)
	except ValueError:
		print("intelligently_set_graph_domain failed, using defaults", file=sys.stderr)
		return left_default,right_default

	if left_root.converged and right_root.converged:
		left_root, right_root= left_root.root, right_root.root
		width = right_root-left_root
		buffer = 0.2*width
		left_root, right_root = left_root-buffer, right_root + buffer
		return left_root,right_root


plt.rcParams.update({'font.size': 16})
def graph_out(dict):
	# parse inputs
	user_inputs = parse_user_inputs(dict)
	prior = user_inputs['prior']
	likelihood = user_inputs['likelihood']
	override_graph_range = user_inputs['override_graph_range']
	
	# compute posterior pdf
	s = time.time()
	posterior = update(prior,likelihood)
	e = time.time()
	print(e-s,'seconds to get posterior pdf',file=sys.stderr)

	# Plot
	if override_graph_range:
		x_from,x_to = override_graph_range
	else:
		x_from , x_to = intelligently_set_graph_domain(posterior)

	s = time.time()
	plot = plot_pdfs_bayes_update(prior,likelihood,posterior,x_from=x_from,x_to=x_to)
	plot = mpld3.fig_to_html(plot)
	e = time.time()
	print(e-s,'seconds to make plot', file=sys.stderr)

	# Expected value
	ev = np.around(posterior.expect(),2)
	ev_string = 'Posterior expected value: '+str(ev)+'<br>'
	
	return plot+ev_string

def percentiles_out_mcmc(dict):
	# Parse inputs
	user_inputs = parse_user_inputs(dict)

	prior = user_inputs['prior']
	likelihood = user_inputs['likelihood']	
	
	# compute posterior pdf
	posterior = update(prior,likelihood)

	#percentiles
	percentiles_mcmc_string = ''
	print("running percentiles mcmc", file=sys.stderr)
	percentiles_mcmc = mcmc_percentiles(posterior,[0.1,0.25,0.5,0.75,0.9])
	percentiles_mcmc_result = percentiles_mcmc['result']
	percentiles_mcmc_runtime = percentiles_mcmc['runtime']

	percentiles_mcmc_string = percentiles_mcmc_runtime+'<br>'
	for x in percentiles_mcmc_result:
		percentiles_mcmc_string += str(x[0]) +', '+ str(np.around(x,2)[1]) + '<br>'
	return percentiles_mcmc_string

def percentiles_out_exact(dict):
	# Parse inputs
	user_inputs = parse_user_inputs(dict)

	prior = user_inputs['prior']
	likelihood = user_inputs['likelihood']	
	
	# compute posterior pdf
	posterior = update(prior,likelihood)

	#percentiles
	percentiles_exact_string = ''
	print("running percentiles exact", file=sys.stderr)
	percentiles_exact = compute_percentiles_exact(posterior,[0.1,0.25,0.5,0.75,0.9])
	percentiles_exact_result = percentiles_exact['result']
	percentiles_exact_runtime = percentiles_exact['runtime']

	percentiles_exact_string = percentiles_exact_runtime +'<br>'
	for x in percentiles_exact_result:
		percentiles_exact_string += str(x) + ', ' + str(percentiles_exact_result[x]) + '<br>'
	return percentiles_exact_string

prior = stats.lognorm(scale=5,s=8)
likelihood = stats.norm(loc=10,scale=3)
posterior = update(prior,likelihood)
intelligently_set_graph_domain(posterior)
print(posterior.support())

# if __name__ == "__main__":
# 	model = pm.Model()
# 	with model:
# 	    mu1 = pm.Normal("mu1", mu=0, sigma=1, shape=1)
# 	    step = pm.NUTS()
# 	    trace = pm.sample(2000, tune=1000, init=None, step=step, cores=2)
# 	print(model.trace)

# s = time.time()
# percentiles_result = np.around(posterior.ppf([0.1,0.25,0.5,0.75,0.9]),3)
# e = time.time()
# print(e-s,'seconds')


# s = time.time()
# compute_percentiles_exact(posterior,[0.1,0.25,0.5,0.75,0.9])
# e = time.time()
# print(e-s,'seconds')

# x = np.arange(-100, 100)
# import time
# ps = [0.1,0.25,0.5,0.75,0.9]
# start = time.time()
# percentiles = np.around(posterior.ppf(ps),3)
# percentiles = zip(ps,percentiles)
# end = time.time()
# print('ppf')
# for x in percentiles:
# 	print(x)


# def p(x):
# 	return posterior.pdf(x)

# def q(x):
# 	return stats.norm.pdf(x, loc=0, scale=100)

# x = np.arange(-100, 100)
# k = max(p(x) / q(x))


# def rejection_sampling(iter):
# 	samples = []

# 	for i in range(iter):
# 		z = np.random.normal(50, 30)
# 		u = np.random.uniform(0, k*q(z))

# 		if u <= p(z):
# 			samples.append(z)

# 	return np.array(samples)



# import time
# if __name__ == '__main__':
# 	plt.plot(x, p(x))
# 	plt.plot(x, k*q(x))
# 	# plt.show()
# 	start = time.time()
# 	s = rejection_sampling(iter=5000)
# 	end = time.time()
# 	print(len(s))
# 	print(s)
# 	print('rejection sampling method:')

# 	print(end-start,'seconds')
# 	percentiles_from_list(s)
	




# print(end-start,'seconds')


# import emcee

# ndim, nwalkers = 1, 20
# ivar = 1. / np.random.rand(ndim)
# p0 = np.random.randn(nwalkers, ndim)

# start = time.time()
# def log_prob(x):
# 	if posterior.pdf(x)>0:
# 		return math.log(posterior.pdf(x))
# 	else:
# 		return -np.inf
# sampler = emcee.EnsembleSampler(nwalkers, 1, log_prob)
# sampler.run_mcmc(p0, 5000)
# s = list(sampler.get_chain(flat=True)[:,0])
# print('len',len(s))
# percentiles_from_list(s)
# end = time.time()
# print(end-start,'seconds mcmc')




# ps = [0.1,0.25,0.5,0.75,0.9]
# percentiles = posterior.ppf(ps)
# percentiles = zip(ps,percentiles)
# percentiles_string = 'Percentiles of posterior distribution: <br>'
# for x in percentiles:
# 	percentiles_string += str(x) + '<br>'
# print(percentiles_string)
# import time
# from pynverse import inversefunc
# start = time.time()
# inv = inversefunc(posterior.cdf, 1, 10)
# end = time.time()
# print(end-start,'seconds')

# start = time.time()
# for p in [0.1,0.25,0.5,0.75,0.9]:
# 	print(posterior.ppf(p))
# end = time.time()

# print(end-start,'seconds')




# x = np.linspace(0,1,10)
# fig, ax = plt.subplots()
# # ax.plot(x,posterior.cdf(x))
# # ax.plot(x,likelihood.cdf(x))
# # ax.plot(x,prior.cdf(x))
# # plt.show()


# # for i in range(40,50):
# # 	print('cdf of',i,posterior.cdf(i))
 


# from multiprocessing import Pool

# def f(x):
# 	return posterior.ppf(x)

# start = time.time()
# if __name__ == '__main__':
#     with Pool() as p:
#         p.map(f, [0.01,0.1,0.25,0.5,0.75,0.9,0.99])
# end = time.time()
# print(end-start,'seconds with Pool')

# for p in [0.01,0.1,0.25,0.5,0.75,0.9,0.99]:
# 	f(p)
