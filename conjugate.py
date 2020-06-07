mu_1 = 1
sigma_1 =1 

mu_2 = 3
sigma_2 = 4

mu_posterior = (mu_1*sigma_1**-2 + mu_2*sigma_2**-2)/(sigma_1**-2 + sigma_2**-2)
sigma_posterior = (sigma_1**-2 + sigma_2**-2)**(-1/2)

print(mu_posterior)
print(sigma_posterior)
print(mu_posterior+2*sigma_posterior)