import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy import integrate

import mpld3

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
	posterior = Product_pdf(prior,likelihood)
	posterior = normalize(posterior)
	return posterior

def parse_user_inputs(dict):
	dict = dict.to_dict()

	'''Very ugly'''
	for key in ['prior-normal-param1','prior-normal-param2',
				'prior-lognormal-param1','prior-lognormal-param2',
				'prior-beta-param1','prior-beta-param2',
				'likelihood-normal-param1','likelihood-normal-param2',
				'likelihood-lognormal-param1','likelihood-lognormal-param2',
				'likelihood-beta-param1','likelihood-beta-param2']:
		if len(dict[key])>0:
			dict[key] = float(dict[key])
	
	if dict["prior-select_distribution_family"] == "normal":
		prior = stats.norm(loc = dict['prior-normal-param1'], scale =dict['prior-normal-param2'])

	elif dict["prior-select_distribution_family"] == "lognormal":
		prior = stats.lognorm(scale = math.exp(dict['prior-lognormal-param1']), s =dict['prior-lognormal-param2'])

	elif dict["prior-select_distribution_family"] == "beta":
		prior = stats.beta(dict["prior-beta-param1"],dict["prior-beta-param2"])

	'''Redundant, will refactor'''
	if dict["likelihood-select_distribution_family"] == "normal":
		likelihood = stats.norm(loc = dict['likelihood-normal-param1'], scale =dict['likelihood-normal-param2'])

	elif dict["likelihood-select_distribution_family"] == "lognormal":
		likelihood = stats.lognorm(scale = math.exp(dict['likelihood-lognormal-param1']), s =dict['likelihood-lognormal-param2'])

	elif dict["likelihood-select_distribution_family"] == "beta":
		likelihood = stats.beta(dict["likelihood-beta-param1"],dict["likelihood-beta-param2"])

	if 'compute_percentiles' in dict.keys():
		return {'prior':prior,'likelihood':likelihood,'compute_percentiles':True}
	else:
		return {'prior':prior,'likelihood':likelihood,'compute_percentiles':False}

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


plt.rcParams.update({'font.size': 16})
def out_html(dict):
	user_inputs = parse_user_inputs(dict)
	prior = user_inputs['prior']
	likelihood = user_inputs['likelihood']
	compute_percentiles = user_inputs['compute_percentiles']
	posterior = update(prior,likelihood)
	plot = plot_pdfs_bayes_update(prior,likelihood,posterior)
	plot = mpld3.fig_to_html(plot)

	ev = np.around(posterior.expect(),3)
	ev_string = 'Posterior expected value: '+str(ev)

	if compute_percentiles:
		percentiles_string = '<br> Percentiles of posterior distribution: <br> ' #very inelegant to have html in here
		ps = [0.1,0.25,0.5,0.75,0.9]
		percentiles = np.around(posterior.ppf(ps),3)
		percentiles = zip(ps,percentiles)
		for x in percentiles:
			percentiles_string += str(x) + '<br>'
		return plot + ev_string + percentiles_string

	return plot + ev_string

# prior = stats.norm(10,1)
# likelihood = stats.norm(20,1)
# posterior = update(prior,likelihood)
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
