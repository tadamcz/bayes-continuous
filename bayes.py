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

def translate_normal_dict(dict):
	dict = dict.to_dict()
	for key in dict:
		dict[key] = float(dict[key])
	prior = stats.norm(loc = dict['prior-location'], scale =dict['prior-scale'])
	likelihood = stats.norm(loc = dict['likelihood-location'], scale =dict['likelihood-scale'])
	return {'prior':prior,'likelihood':likelihood}


# o = stats.gaussian_kde(tom_data.l)

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

	beta_hat = "TODO"
	prior_string = "P(X)"
	likelihood_string = "P(Ê="+str(beta_hat)+"|X)"
	posterior_string = "P(X | Ê="+str(beta_hat)+")"

	plot = plot_pdfs({	prior_string:prior,
						likelihood_string:likelihood,
						posterior_string:posterior}
						,x_from,x_to)
	plot = mpld3.fig_to_html(plot)
	return plot


def plot_normals_flask(dict):
	prior = translate_normal_dict(dict)['prior']
	likelihood = translate_normal_dict(dict)['likelihood']
	posterior = update(prior,likelihood)
	out = plot_pdfs_bayes_update(prior,likelihood,posterior)
	return out