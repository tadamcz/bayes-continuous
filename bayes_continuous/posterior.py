import time

import numpy as np
from scipy import integrate
from scipy import optimize
from scipy import stats
from scipy.stats._distn_infrastructure import rv_frozen
from sortedcontainers import SortedDict

from bayes_continuous import utils
from bayes_continuous.likelihood_func import LikelihoodFunction


class Posterior(stats.rv_continuous):
	def __init__(self, prior_distribution: rv_frozen, likelihood_function: LikelihoodFunction):
		"""
		We can call theta the parameter over which we want to conduct inference.
		The inputs to this __init__ are then:
		* likelihood function: relative probability of the data/event we observed, as a function of theta
		* prior over theta, with any number of parameters

		(This prior is sometimes called a hyper-prior with hyper-parameters,
		because theta is also the parameter of a distribution, the distribution Norm(θ,s²)
		used in the likelihood function below)

		Example with lognormal prior and normal likelihood:

		Suppose that a study has the point estimator B for the parameter Bigθ.
		Bigθ could be the average height of men in England.
		Possible values of Bigθ are denoted θ.

		The study results are an estimate b for B, and an estimated standard deviation sd^(B)=s.
		For example, b is the sample mean, with a sample standard deviation of s.
		To keep the problem 1-dimensional, we assume that s is the true standard deviation of B.

		The likelihood function is:
		ℒ: θ ↦ P(b∣ θ) = PDFNorm(θ,s²)(b)

		The prior over Bigθ is lognormal.
		This lognormal distribution has the following hyper-parameters: μ, σ.
		Its PDF is:
		PriorPDF: θ ↦ PDFLogNorm(μ, σ²)(θ)

		The posterior distribution over Bigθ has the following PDF:
		Posterior: θ ↦ P(θ|b) = ℒ(θ)*PriorPDF(θ) / normalization_constant
		"""
		super(Posterior, self).__init__()

		self.prior_distribution = prior_distribution
		self.likelihood_function = likelihood_function

		# Lookup table for performance
		self.cdf_lookup = SortedDict()

		'''
		Defining the support of the product pdf is important
		because, when we use a numerical equation solver on the CDF,
		it will only need to look for solutions in the support, instead
		of on the entire real line.
		'''
		a1, b1 = prior_distribution.support()
		a2, b2 = likelihood_function.left_bound, likelihood_function.right_bound

		# SciPy calls the support (a,b)
		self.a, self.b = utils.intersect_intervals([(a1, b1), (a2, b2)])

		self._mode = None
		self.integral_splitpoint = self.mode()

		self.normalization_constant = utils.split_integral(function_to_integrate=self.unnormalized_pdf,
														   splitpoint=self.integral_splitpoint,
														   integrate_to=self.b,
														   support=self.support())

	def mode(self):
		if self._mode is None:
			neg_pdf = lambda x: -self.unnormalized_pdf(x)
			left_bound, right_bound = self.support()
			if np.isfinite(left_bound) and np.isfinite(right_bound):
				method = 'bounded'
			else:
				method = 'brent'

			if left_bound == -np.inf:
				left_bound = None
			if right_bound == np.inf:
				right_bound = None

			optimize_result = optimize.minimize_scalar(neg_pdf, bounds=(left_bound, right_bound), method=method)
			if not optimize_result.success:
				raise RuntimeError
			else:
				self._mode = optimize_result.x

		return self._mode

	def unnormalized_pdf(self, x):
		return self.prior_distribution.pdf(x) * self.likelihood_function.function(x)

	def _pdf(self, x):
		return self.unnormalized_pdf(x) / self.normalization_constant

	def _cdf(self, x):

		# Todo: a better way to deal with ndarray inputs vs scalars
		if isinstance(x, np.ndarray):
			if len(x) == 1 and x[0] in self.cdf_lookup:
				return self.cdf_lookup[x[0]]

		# We can consider the cdf to be 1 forevermore once it reaches values close to 1 (todo: do this for very small values too)
		# Find the greatest key less than x
		index = self.cdf_lookup.bisect_left(x) - 1
		if index >= 0:
			cache_key, cache_val = self.cdf_lookup.peekitem(index)
			if x < cache_key:  # todo replace this with a test
				raise RuntimeError
			if x > cache_key and np.isclose(cache_val, 1, rtol=1e-7, atol=0):
				return 1


		# A form of memoization:
		# Check lookup table for largest integral already computed below x.
		# Only integrate the remaining bit.
		# Same number of integrations, but the integrations are over a much smaller interval.

		# Find the greatest key less than x
		index = self.cdf_lookup.bisect_left(x) - 1
		if index >= 0:
			cache_key, cache_val = self.cdf_lookup.peekitem(index)
			cdf_value = cache_val + integrate.quad(self.pdf, cache_key, x)[0]  # integrate the smaller interval
			if x < cache_key:  # todo replace this with a test
				raise RuntimeError
			self.cdf_lookup[float(x)] = cdf_value
			return cdf_value

		cdf_value = utils.split_integral(
			function_to_integrate=self.pdf,
			splitpoint=self.integral_splitpoint,
			integrate_to=x,
			support=self.support())

		self.cdf_lookup[float(x)] = cdf_value
		return cdf_value

	def ppf_with_bounds(self, quantile, leftbound, rightbound):
		"""
		wraps scipy ppf function
		https://github.com/scipy/scipy/blob/4c0fd79391e3b2ec2738bf85bb5dab366dcd12e4/scipy/stats/_distn_infrastructure.py#L1681-L1699
		"""

		factor = 10.
		left, right = self._get_support()

		if np.isinf(left):
			left = min(-factor, right)
			while self._ppf_to_solve(left, quantile) > 0.:
				left, right = left * factor, left
		# left is now such that cdf(left) <= q
		# if right has changed, then cdf(right) > q

		if np.isinf(right):
			right = max(factor, left)
			while self._ppf_to_solve(right, quantile) < 0.:
				left, right = right, right * factor
		# right is now such that cdf(right) >= q

		# This is where we add the bounds to ppf copied from above
		if leftbound is not None:
			left = leftbound
		if rightbound is not None:
			right = rightbound

		return optimize.brentq(self._ppf_to_solve, left, right, args=quantile, xtol=self.xtol, full_output=False)

	def compute_percentiles(self, percentiles_list):
		result = {}

		start = time.time()
		# print('Running compute_percentiles_exact. Support: ', self.support())

		percentiles_list.sort()
		percentiles_reordered = sum(zip(percentiles_list,reversed(percentiles_list)), ())[:len(percentiles_list)] #https://stackoverflow.com/a/17436999/8010877

		def get_bounds_on_ppf(existing_results, percentile):
			keys = list(existing_results.keys())
			keys.append(percentile)
			keys.sort()
			i = keys.index(percentile)
			if i != 0:
				leftbound = existing_results[keys[i - 1]]
			else:
				leftbound = None
			if i != len(keys) - 1:
				rightbound = existing_results[keys[i + 1]]
			else:
				rightbound = None
			return leftbound, rightbound

		for p in percentiles_reordered:
			# print("trying to compute the", p, "th percentile")
			try:
				leftbound, rightbound = get_bounds_on_ppf(result, p)
				res = self.ppf_with_bounds(p, leftbound, rightbound)
				result[p] = res
			except RuntimeError as e:
				result[p] = e

		sorted_result = {key: value for key, value in sorted(result.items())}

		end = time.time()
		description_string = 'Computed in ' + str(np.around(end - start, 1)) + ' seconds'

		return {'result': sorted_result, 'runtime': description_string}


class CustomFromPDF(stats.rv_continuous):
	def __init__(self, pdf_callable, a=-np.inf, b=np.inf):
		super(CustomFromPDF, self).__init__()
		self.pdf_callable = pdf_callable
		self.a = a
		self.b = b

	def _pdf(self, x):
		return self.pdf_callable(x)
