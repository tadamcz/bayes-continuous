import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy import integrate
import sys

import mpld3
import emcee

def is_a_distribution(obj):
	if issubclass(type(obj),stats.rv_continuous):
		return True
	if isinstance(obj,stats._distn_infrastructure.rv_frozen):
		return True
	return False

class Product_pdf(stats.rv_continuous):
	def __init__(self,d1,d2):
		super(Product_pdf,self).__init__()
		if not is_a_distribution(d1):
			raise TypeError("First argument must be a distribution")
		if type(d2) is not float and type(d2) is not int and not is_a_distribution(d2):
			raise TypeError("Second argument must be a distribution or a number")
		self.d1= d1
		self.d2= d2
	def _pdf(self,x):
		if type(self.d2) is float:
			ret = self.d1.pdf(x)*self.d2
		if type(self.d2) is stats._distn_infrastructure.rv_frozen:
			ret = self.d1.pdf(x)*self.d2.pdf(x)
		return ret

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
		yes = cond1 or cond2

		cond3 = 'percentiles' in key
		cond4 = 'family' in key
		cond5 = 'csrf' in key
		cond6 = len(dict[key])==0

		no = cond4 or cond5 or cond6

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

	compute_percentiles_exact = False
	compute_percentiles_mcmc = False
	
	if 'percentiles-compute_percentiles_exact' in dict.keys():
		compute_percentiles_exact = True
	
	if 'percentiles-compute_percentiles_mcmc' in dict.keys():
		compute_percentiles_mcmc = True

	compute_perentiles_any = compute_percentiles_mcmc or compute_percentiles_exact
	
	return {'prior':prior,
				'likelihood':likelihood,
				'compute_percentiles_exact':compute_percentiles_exact,
				'compute_percentiles_mcmc':compute_percentiles_mcmc,
				'compute_percentiles_any':compute_perentiles_any}

def plot_pdfs(dict_of_dists,x_from=-5,x_to=5):
	x = np.linspace(x_from,x_to,(x_to-x_from)*10)
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

import time
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
	state = sampler.run_mcmc(p0, 100)
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
	try:
		percentiles_result = np.around(distr.ppf(percentiles_list),3)
	except RuntimeError:
		return {'result':'','runtime':'RuntimeError'}
	end = time.time()
	description_string = 'Computed in '+str(np.around(end-start,1))+' seconds'
	percentiles_result = zip(percentiles_list,percentiles_result)
	return {'result':percentiles_result,'runtime':description_string}



plt.rcParams.update({'font.size': 16})
def graph_out(dict):
	# parse inputs
	user_inputs = parse_user_inputs(dict)
	prior = user_inputs['prior']
	likelihood = user_inputs['likelihood']
	
	# compute posterior pdf
	posterior = update(prior,likelihood)


	# Plot
	plot = plot_pdfs_bayes_update(prior,likelihood,posterior)
	plot = mpld3.fig_to_html(plot)

	# Expected value
	ev = np.around(posterior.expect(),3)
	ev_string = 'Posterior expected value: '+str(ev)+'<br>'
	
	return plot+ev_string



def percentiles_out(dict):
	# Parse inputs
	user_inputs = parse_user_inputs(dict)

	compute_percentiles_exact_setting = user_inputs['compute_percentiles_exact']
	compute_percentiles_mcmc_setting = user_inputs['compute_percentiles_mcmc']

	prior = user_inputs['prior']
	likelihood = user_inputs['likelihood']
	
	# compute posterior pdf
	posterior = update(prior,likelihood)

	# percentiles, exact
	percentiles_exact_string = ''
	if compute_percentiles_exact_setting:
		print("running percentiles exact", file=sys.stderr)
		percentiles_exact = compute_percentiles_exact(posterior,[0.1,0.25,0.5,0.75,0.9])
		percentiles_exact_result = percentiles_exact['result']
		percentiles_exact_runtime = percentiles_exact['runtime']

		percentiles_exact_string = '<br> Percentiles of posterior distribution (exact): <br> '+percentiles_exact_runtime +'<br>'#very inelegant to have html in here
		for x in percentiles_exact_result:
			percentiles_exact_string += str(x) + '<br>'

	# percentiles, mcmc
	percentiles_mcmc_string = ''
	if compute_percentiles_mcmc_setting:
		print("running percentiles mcmc", file=sys.stderr)
		percentiles_mcmc = mcmc_percentiles(posterior,[0.1,0.25,0.5,0.75,0.9])
		percentiles_mcmc_result = percentiles_mcmc['result']
		percentiles_mcmc_runtime = percentiles_mcmc['runtime']

		percentiles_mcmc_string = '<br> Percentiles of posterior distribution (MCMC): <br>'+percentiles_mcmc_runtime+'<br>'
		for x in percentiles_mcmc_result:
			percentiles_mcmc_string += str(x[0]) +', '+ str(np.around(x,3)[1]) + '<br>'

	return percentiles_exact_string + percentiles_mcmc_string


# prior = stats.lognorm(scale=math.exp(2),s=2)
# likelihood = stats.norm(loc=5,scale=15)
# posterior = update(prior,likelihood)

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
