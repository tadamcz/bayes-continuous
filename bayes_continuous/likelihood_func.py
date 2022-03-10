"""
In this file, the model parameter that the likelihood function takes as argument is always called theta.
This naming pattern should help make this file more readable.
"""
from typing import Callable

import numpy as np
from scipy import stats, optimize


class LikelihoodFunction:
	def __init__(self, likelihood: Callable[[float], float], left_bound=-np.inf, right_bound=np.inf):
		self.function = likelihood
		self.neg_likelihood = lambda x: -self.function(x)
		self.left_bound = left_bound
		self.right_bound = right_bound
		self.domain = self.left_bound, self.right_bound

	def mode(self):
		if self.left_bound == -np.inf:
			opt_left_bound = None
		else:
			opt_left_bound = self.left_bound
		if self.right_bound == np.inf:
			opt_right_bound = None
		else:
			opt_right_bound = self.right_bound

		optimize_result = optimize.minimize_scalar(self.neg_likelihood, bounds=(opt_left_bound, opt_right_bound))
		if not optimize_result.success:
			raise RuntimeError
		return optimize_result.x


class NormalLikelihood(LikelihoodFunction):
	def __init__(self, mu, sigma):
		"""
		Note, I could have written
		```
		likelihood = stats.norm(loc=mu, scale=sigma).pdf
		```
		because the two functions happen to have the same values.

		But this is a special case (see https://fragile-credences.github.io/bayes-normal-likelihood/),
		so I prefer to use the formula that is also true in general.
		"""
		self.mu = mu
		self.sigma = sigma
		self.domain = (-np.inf,np.inf)
		likelihood = lambda theta: stats.norm(loc=theta, scale=sigma).pdf(mu)
		super().__init__(likelihood)


class BinomialLikelihood(LikelihoodFunction):
	def __init__(self, successes: int, trials: int):
		"""
		`scipy.stats.binom` takes two arguments (n,p), where n is the number of trials
		and p is the probability of success.
		"""
		self.successes = successes
		self.trials = trials
		self.domain = (0,1)
		likelihood = lambda theta: stats.binom(trials, theta).pmf(successes)
		super().__init__(likelihood)
