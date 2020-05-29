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
	for key in dict:
		if len(dict[key])>0:
			dict[key] = float(dict[key])

	if dict["norm-prior-location"] != "":
		prior = stats.norm(loc = dict['norm-prior-location'], scale =dict['norm-prior-scale'])

	elif dict["lognorm-prior-mu"] != "":
		prior = stats.lognorm(scale = math.exp(dict['lognorm-prior-mu']), s =dict['lognorm-prior-sigma'])

	elif dict["beta-prior-alpha"] != "":
		prior = stats.beta(dict["beta-prior-alpha"],dict["beta-prior-beta"])


	likelihood = stats.norm(loc = dict['likelihood-location'], scale =dict['likelihood-scale'])

	print(prior,likelihood)
	return {'prior':prior,'likelihood':likelihood}

def plot_pdfs(dict_of_dists,x_from,x_to):
	x = np.linspace(x_from,x_to,50)
	fig, ax = plt.subplots()
	for dist in dict_of_dists:
		ax.plot(x,dict_of_dists[dist].pdf(x),label=dist)
	ax.legend()
	ax.set_xlabel("X")
	ax.set_ylabel("Probability density")
	return fig


def plot_pdfs_bayes_update(prior,likelihood,posterior,x_from=-5,x_to=5):
	prior_string = "P(X)"
	likelihood_string = "P(E|X)"
	posterior_string = "P(X|E)"

	plot = plot_pdfs({	prior_string:prior,
						likelihood_string:likelihood,
						posterior_string:posterior}
						,x_from,x_to)
	return plot


plt.rcParams.update({'font.size': 16})
def plot_out_html(dict):
	user_inputs = parse_user_inputs(dict)
	prior = user_inputs['prior']
	likelihood = user_inputs['likelihood']
	posterior = update(prior,likelihood)
	plot = plot_pdfs_bayes_update(prior,likelihood,posterior)
	plot = mpld3.fig_to_html(plot)
	return plot