import numpy as np
import pytest
from scipy import stats

from bayes_continuous.likelihood_func import NormalLikelihood
from bayes_continuous.posterior import Posterior

prior_mu = 5.45
prior_sigma = 6.789

likelihood_mu = 1.23
likelihood_sigma = 3

prior = stats.norm(loc=prior_mu, scale=prior_sigma)
likelihood = NormalLikelihood(likelihood_mu, likelihood_sigma)

posterior = Posterior(prior, likelihood)

numerator = prior_mu * prior_sigma ** -2 + likelihood_mu * likelihood_sigma ** -2
denominator = prior_sigma ** -2 + likelihood_sigma ** -2

posterior_mu_closed_form = numerator / denominator
posterior_sigma_closed_form = (prior_sigma ** -2 + likelihood_sigma ** -2) ** (-1 / 2)
posterior_closed_form = stats.norm(loc=posterior_mu_closed_form, scale=posterior_sigma_closed_form)


def test_ev_mean_mode():
	assert posterior.expect() == pytest.approx(posterior_mu_closed_form)
	assert posterior.mean() == pytest.approx(posterior_mu_closed_form)
	assert posterior.mode() == pytest.approx(posterior_mu_closed_form)


def test_ppf():
	for p in (0.001, 0.01, 0.1, .5, .9, .99, .999):
		assert posterior.ppf(p) == pytest.approx(posterior_closed_form.ppf(p))


def test_pdf():
	p_0001 = posterior_closed_form.ppf(0.001)
	p_999 = posterior_closed_form.ppf(0.999)

	for x in np.linspace(p_0001, p_999, 100):
		assert posterior.pdf(x) == pytest.approx(posterior_closed_form.pdf(x))
