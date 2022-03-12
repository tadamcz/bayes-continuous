import numpy as np
from scipy import integrate, stats
from scipy import optimize


def is_frozen_normal(distribution):
	if isinstance(distribution, stats._distn_infrastructure.rv_frozen):
		if isinstance(distribution.dist, stats._continuous_distns.norm_gen):
			return True
	return False


def mode_of_distribution(distribution: stats._distn_infrastructure.rv_frozen):
	if is_frozen_normal(distribution):
		args, kwds = distribution.args, distribution.kwds
		shapes, loc, scale = distribution.dist._parse_args(*args, **kwds)
		return loc

	neg_pdf = lambda x: -distribution.pdf(x)
	left_bound, right_bound = distribution.support()
	if left_bound == -np.inf:
		left_bound = None
	if right_bound == np.inf:
		right_bound = None
	optimize_result = optimize.minimize_scalar(neg_pdf, bounds=(left_bound, right_bound))
	if not optimize_result.success:
		raise RuntimeError
	return optimize_result.x


def intersect_intervals(two_tuples):
	interval1, interval2 = two_tuples

	interval1_left, interval1_right = interval1
	interval2_left, interval2_right = interval2

	if interval1_right < interval2_left or interval2_right < interval2_left:
		raise ValueError("the distributions have no overlap")

	intersect_left, intersect_right = max(interval1_left, interval2_left), min(interval1_right, interval2_right)

	return intersect_left, intersect_right


def extremities_intervals(two_tuples):
	interval1, interval2 = two_tuples

	interval1_left, interval1_right = interval1
	interval2_left, interval2_right = interval2

	extreme_left = min(interval1_left, interval1_right, interval2_left, interval2_right)
	extreme_right = max(interval1_left, interval1_right, interval2_left, interval2_right)

	return extreme_left, extreme_right


def split_integral(function_to_integrate, splitpoint, integrate_to, support=(-np.inf, np.inf)):
	"""
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
	"""
	support_left, support_right = support

	if integrate_to < splitpoint:
		# just return the integral normally
		return integrate.quad(function_to_integrate, support_left, integrate_to)[
			0]  # only return the answer, first element in tuple. Same below.

	else:
		integral_left = integrate.quad(function_to_integrate, support_left, splitpoint)[0]
		integral_right = integrate.quad(function_to_integrate, splitpoint, integrate_to)[0]
		return integral_left + integral_right


def normal_parameters(x1, p1, x2, p2):
	"""
	Find parameters for a normal random variable X so that P(X < x1) = p1 and P(X < x2) = p2.
	"""
	denom = stats.norm.ppf(p2) - stats.norm.ppf(p1)
	sigma = (x2 - x1) / denom
	mu = (x1 * stats.norm.ppf(p2) - x2 * stats.norm.ppf(p1)) / denom
	return (mu, sigma)


def normal_normal_closed_form(mu_1, sigma_1, mu_2, sigma_2):
	"""
	Returns a pair (posterior_mu, posterior_sigma)
	"""
	if sigma_1 < 0 or sigma_2 < 0:
		raise ValueError

	numerator = mu_1 * sigma_1 ** -2 + mu_2 * sigma_2 ** -2
	denominator = sigma_1 ** -2 + sigma_2 ** -2

	posterior_mu = numerator / denominator
	posterior_sigma = (sigma_1 ** -2 + sigma_2 ** -2) ** (-1 / 2)

	return posterior_mu, posterior_sigma


def beta_binomial_closed_form(prior_alpha, prior_beta, likelihood_successes, likelihood_trials):
	"""
	Returns a pair (posterior_alpha, posterior_beta)
	"""
	if not float(likelihood_trials).is_integer():
		raise ValueError
	if not float(likelihood_successes).is_integer():
		raise ValueError

	likelihood_failures = likelihood_trials - likelihood_successes

	posterior_alpha = prior_alpha + likelihood_successes
	posterior_beta = prior_beta + likelihood_failures
	return posterior_alpha, posterior_beta
