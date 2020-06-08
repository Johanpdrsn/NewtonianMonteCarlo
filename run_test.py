from jax.random import PRNGKey
import jax.numpy as np
from numpyro import sample
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from mixture_model import NormalMixture
from NMC import NMC

# Model key
rng = PRNGKey(0)

#Mixture model
def mix_model(data):
    w = sample("w", dist.Dirichlet((1/3) * np.ones(3)), rng_key=rng)
    mu = sample("mu", dist.Normal(np.zeros(3), np.ones(3)), rng_key=rng)
    std = sample("std", dist.Gamma(np.ones(3), np.ones(3)), rng_key=rng)
    sample("obs", NormalMixture(w, mu, std), rng_key=rng, obs=data)

#Data for mixture model
data_test1 = sample("norm1", dist.Normal(10,1), rng_key=PRNGKey(0), sample_shape=(1000,))
data_test2 = sample("norm2", dist.Normal(0,1), rng_key=PRNGKey(0), sample_shape=(1000,))
data_test3 = sample("norm3", dist.Normal(-10,1), rng_key=PRNGKey(0), sample_shape=(1000,))
test = [data_test1,data_test2,data_test3]
data_mix = np.array(test).T

# Instantiate model
nmc = NMC(mix_model,data_mix)

#Run inference
nmc.run(1000)
for key in nmc.acc_trace:
    print(key)
    print(np.mean(np.array(nmc.acc_trace[key])))
    print(np.std(np.array(nmc.acc_trace[key])))
    print(np.median(np.array(nmc.acc_trace[key])))


print(nmc.nmc_status)

kernel = NUTS(mix_model)
mcmc = MCMC(kernel, 0, 1000)
mcmc.run(rng, data_mix)
mcmc.print_summary()