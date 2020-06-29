from bayes import *

prior = stats.lognorm(1,1)
likelihood = stats.norm(20,50)
posterior = Posterior_scipyrv(prior,likelihood)

s = time.time()
for i in range(100):
	print(1-i/100,posterior.ppf(1-i/100))
e = time.time()
print(e-s,'secs')