import pytest
import posterior
from scipy import stats
import math

def do_analytic_math(mu_1,sigma_1,mu_2,sigma_2):
    mu_posterior = (mu_1 * sigma_1 ** -2 + mu_2 * sigma_2 ** -2) / (sigma_1 ** -2 + sigma_2 ** -2)
    sigma_posterior = (sigma_1 ** -2 + sigma_2 ** -2) ** (-1 / 2)
    return {'mu':mu_posterior,'sigma':sigma_posterior}

@pytest.mark.parametrize("mu_1", [1,10])
@pytest.mark.parametrize("sigma_1", [1,10])
@pytest.mark.parametrize("mu_2", [1,10])
@pytest.mark.parametrize("sigma_2", [1,10])
@pytest.mark.parametrize("x", [1,10])
@pytest.mark.parametrize("p", [0.99])
def test_basic(mu_1,sigma_1,mu_2,sigma_2,x,p):
    mu_posterior = do_analytic_math(mu_1,sigma_1,mu_2,sigma_2)['mu']
    sigma_posterior = do_analytic_math(mu_1,sigma_1,mu_2,sigma_2)['sigma']
    posterior_analytic = stats.norm(mu_posterior,sigma_posterior)

    prior_scipy = stats.norm(mu_1, sigma_1)
    likelihood_scipy = stats.norm(mu_2, sigma_2)
    posterior_numerical = posterior.Posterior(prior_scipy, likelihood_scipy)

    relative_tolerance = 1e-3
    assert pytest.approx(posterior_numerical.expect(), rel=relative_tolerance) == posterior_analytic.expect(), "Expected Value using expect()"
    assert pytest.approx(posterior_numerical.expect(), rel=relative_tolerance) == mu_posterior, "Expected value using mu"
    assert pytest.approx(posterior_numerical.cdf(x), rel=relative_tolerance) == posterior_analytic.cdf(x), "CDF"
    assert pytest.approx(posterior_numerical.ppf(p), rel=relative_tolerance) == posterior_analytic.ppf(p), "PPF"

#lognormal case below -- still figuring this out

# @pytest.mark.parametrize("mu_1", [1])
# @pytest.mark.parametrize("sigma_1", [1])
# @pytest.mark.parametrize("mu_2", [1])
# @pytest.mark.parametrize("sigma_2", [1])
# @pytest.mark.parametrize("x", [1])
# @pytest.mark.parametrize("p", [0.99])
# def test_basic_2(mu_1,sigma_1,mu_2,sigma_2,x,p):
#     mu_posterior = do_analytic_math(mu_1,sigma_1,mu_2,sigma_2)['mu']
#     sigma_posterior = do_analytic_math(mu_1,sigma_1,mu_2,sigma_2)['sigma']
#     posterior_analytic = stats.lognorm(scale=math.exp(mu_posterior),s=sigma_posterior)
#
#     prior_scipy = stats.lognorm(scale=math.exp(mu_1), s=sigma_1)
#     likelihood_scipy = stats.lognorm(scale=math.exp(mu_2), s=sigma_2)
#     posterior_numerical = bayes.Posterior_scipyrv(prior_scipy, likelihood_scipy)
#
#     relative_tolerance = 1e-3
#     assert pytest.approx(posterior_numerical.expect(), rel=relative_tolerance) == posterior_analytic.expect(), "Expected Value using expect()"
#     assert pytest.approx(posterior_numerical.median(), rel=relative_tolerance) == mu_posterior, "Median value using mu"
#     assert pytest.approx(posterior_numerical.cdf(x), rel=relative_tolerance) == posterior_analytic.cdf(x), "CDF"
#     assert pytest.approx(posterior_numerical.ppf(p), rel=relative_tolerance) == posterior_analytic.ppf(p), "PPF"
