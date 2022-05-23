import numpy as np
import pytest
from scipy import stats

from bayes_continuous import utils
from bayes_continuous.likelihood_func import BinomialLikelihood
from bayes_continuous.posterior import Posterior

prior_alpha = 1
prior_beta = 3

likelihood_successes = 4
likelihood_trials = 10
likelihood_failures = likelihood_trials - likelihood_successes

prior = stats.beta(prior_alpha, prior_beta)
likelihood = BinomialLikelihood(successes=likelihood_successes, trials=likelihood_trials)

posterior = Posterior(prior, likelihood)

posterior_alpha_closed_form, posterior_beta_closed_form = utils.beta_binomial_closed_form(
	prior_alpha=prior_alpha,
	prior_beta=prior_beta,
	likelihood_successes=likelihood_successes,
	likelihood_trials=likelihood_trials
)
posterior_closed_form = stats.beta(posterior_alpha_closed_form, posterior_beta_closed_form)


def test_ev():
	ev_closed_form = posterior_alpha_closed_form / (posterior_alpha_closed_form + posterior_beta_closed_form)
	assert posterior.expect() == pytest.approx(ev_closed_form)


def test_ppf():
	for p in (0.001, 0.01, 0.1, .5, .9, .99, .999):
		assert posterior.ppf(p) == pytest.approx(posterior_closed_form.ppf(p))


def test_pdf():
	p_0001 = posterior_closed_form.ppf(0.001)
	p_999 = posterior_closed_form.ppf(0.999)

	for x in np.linspace(p_0001, p_999, 100):
		assert posterior.pdf(x) == pytest.approx(posterior_closed_form.pdf(x))


def test_variance():
	numerator = posterior_alpha_closed_form * posterior_beta_closed_form
	denominator = (posterior_alpha_closed_form + posterior_beta_closed_form) ** 2 * (
			posterior_alpha_closed_form + posterior_beta_closed_form + 1)
	var_closed_form = numerator / denominator
	assert posterior.var() == pytest.approx(var_closed_form)
