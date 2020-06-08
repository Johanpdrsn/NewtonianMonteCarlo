from collections import namedtuple

from jax.random import PRNGKey, split
import jax.numpy as np
from jax import partial, grad, hessian

from numpyro.infer.util import log_density
import numpyro.distributions as dist
from numpyro import sample
from numpyro.handlers import trace, seed, substitute, replay


#Constants for reconstructing sigma, and dampening the gradient
UNCONSTRAINED_RECONSTRUCTION = 1e-5
STD_CORRECTION = 1.3
W_CORRECTION = 1
MU_CORRECTION = 0.1


#Tracks state of the NMC algorithm
NMC_STATUS = namedtuple("NMC_status", ["i", "params", "log_likelihood", "accept_prob",
                                   "rng_key"])


class NMC:
    #Class containing all of the NMC functionality
    """
    The class containing all NMC functionality. This includes the main sampling loop,
    determination of support, individual proposal distributions and functions for running the algorithm
    """
    # Init the classe with a model and the data
    def __init__(self, model, *model_args, rng_key=PRNGKey(0), **model_kwargs):
        self.model = model
        self.model_args = model_args
        self.rng_key = rng_key
        self.model_kwargs = model_kwargs            

        tr = trace(model).get_trace(model_args)
        log_likelihood= log_density(self.model, self.model_args, self.model_kwargs, self.get_params(tr))[0]

        self.nmc_status = NMC_STATUS(i=0, params=self.get_params(tr),log_likelihood=log_likelihood, accept_prob=0., rng_key=rng_key) 

        self.props = {}
        self.acc_trace = {}
        self.init_trace()


    #Initialize the accepted trace proposal objects
    def init_trace(self):
        for name in self.nmc_status.params:
            dim = len(self.nmc_status.params[name])
            for i in range(dim):
                self.props[name+str(i)] = []
                self.acc_trace[name+str(i)] = []
    

    # Return key,value parameter pair from trace. Excludes observed variables
    def get_params(self, tr):
        return {name: site["value"] for name, site in tr.items() if not site["is_observed"]}


    # The core sampler functions. Running a a single site inferece MH algorithm
    def sample(self):
        rng_key, rng_key_sample, rng_key_accept = split(self.nmc_status.rng_key, 3)
        params = self.nmc_status.params

        for site in params.keys():
            #Collect accepted trace
            for i in range(len(params[site])):
                self.acc_trace[site+str(i)].append(params[site][i])

            tr_current = trace(substitute(self.model,params)).get_trace(*self.model_args, **self.model_kwargs)
            ll_current = self.nmc_status.log_likelihood

            val_current = tr_current[site]["value"]
            dist_curr = tr_current[site]["fn"]

            log_den_fun = lambda var: partial(log_density, self.model, self.model_args, self.model_kwargs)(var)[0]
    
            val_proposal, dist_proposal = self.proposal(site, log_den_fun, self.get_params(tr_current),
                    dist_curr, rng_key_sample)

            tr_proposal = self.retrace(site, tr_current, dist_proposal, val_proposal, self.model_args, self.model_kwargs)
            ll_proposal = log_density(self.model, self.model_args, self.model_kwargs, self.get_params(tr_proposal))[0]

            ll_proposal_val = dist_proposal.log_prob(val_current).sum()
            ll_current_val = dist_curr.log_prob(val_proposal).sum()

            hastings_ratio = (ll_proposal + ll_proposal_val) - (ll_current + ll_current_val) 

            accept_prob = np.minimum(1, np.exp(hastings_ratio))
            u = sample("u", dist.Uniform(0,1), rng_key=rng_key_accept)

            if u <= accept_prob:
                params, ll_current = self.get_params(tr_proposal), ll_proposal
            else:
                params, ll_current = self.get_params(tr_current), ll_current

        iter = self.nmc_status.i + 1
        mean_accept_prob = self.nmc_status.accept_prob + (accept_prob - self.nmc_status.accept_prob) / iter

        return NMC_STATUS(iter, params, ll_current, mean_accept_prob, rng_key)


    # Computes the gradient and hessian to use in the specific proposal function
    def proposal(self, name, log_den_fun, params, dist_curr, rng_key):
        grad_fn = grad(log_den_fun)
        hess_fn = hessian(log_den_fun)

        grad_ = grad_fn(params)[name]
        hess = hess_fn(params)[name][name]

        proposal_ = self.match_proposal_dist(dist_curr)
        value, dist_ = proposal_(rng_key, params[name], grad_, hess)

        #Collect proposals
        dim = len(value)
        for i in range(dim):
            var_name = name+str(i)
            self.props[var_name].append(value[i])

        return value, dist_


    #Matches to target distribution to a proposal distribution
    def match_proposal_dist(self,dist_):
        support = dist_.support
        if isinstance(support,(dist.constraints._Real, dist.constraints._RealVector)):
            return self.unconstrained_proposal
        elif isinstance(support, (dist.constraints._GreaterThan)):
            return self.halfspace_proposal
        elif isinstance(support, (dist.constraints._Simplex)):
            return self.simplex_proposal
        else:
            raise Exception("No proposal for given variable support")


    # Proposal for the unconstrained space. Uses rules from the article, and reconstructs if sigma is not postive definite
    def unconstrained_proposal(self,rng_key, x, grad_, hess_):
        ndim = np.ndim(x)
        if ndim == 0:
            inv_hess = 1 / hess_
            dist_type = dist.Normal
        else:
            inv_hess = np.linalg.inv(hess_)
            dist_type = dist.MultivariateNormal

        loc = x - np.dot(inv_hess, grad_)
        sigma = -inv_hess

        #Reconstruct sigma if not positive definite
        if not ndim==0 and not np.all(np.linalg.eigvals(sigma)>0):
            lam, vec = np.linalg.eigh(sigma)
            sigma = vec @ np.diag(np.maximum(lam, UNCONSTRAINED_RECONSTRUCTION)) @ vec.T

        dist_ = dist_type(loc, sigma+MU_CORRECTION)

        return dist_.sample(rng_key).reshape(x.shape), dist_


    # Halspace proposer. If alpha or beta values are negative they are set to $HALF_SPACE_RECONSTRUCTION$
    def halfspace_proposal(self,rng_key, x, grad_, hess_):
        alpha = (1 - np.dot(x ** 2, hess_))
        beta = -np.dot(x, hess_) - grad_

        dist_ = dist.continuous.Gamma(concentration=alpha*STD_CORRECTION, rate=beta)
        
        return dist_.sample(rng_key).reshape(x.shape), dist_


    # Simpex proposer following the article
    def simplex_proposal(self,rng_key, x, grad_, hess_):
        max_non_diag_hess = np.max(hess_[np.logical_not(np.eye(hess_.shape[0], dtype=bool))].reshape(hess_.shape[0], -1), axis=1)
        concentration = 1 - x**2 * (np.diag(hess_) - max_non_diag_hess)

        dist_ = dist.Dirichlet(concentration=concentration+W_CORRECTION)

        return dist_.sample(rng_key).reshape(x.shape), dist_

    # Reruns a trace with the new proposed value and distribution
    def retrace(self,name, tr, dist_proposal, val_proposal, model_args, model_kwargs):
        fn_current = tr[name]["fn"]
        val_current = tr[name]["value"]

        tr[name]["fn"] = dist_proposal
        tr[name]["value"] = val_proposal

        second_trace = trace(replay(self.model, tr)).get_trace(*model_args, **model_kwargs)

        tr[name]["fn"] = fn_current
        tr[name]["value"] = val_current

        return second_trace
    

    # Run the inferece with number of iterations
    def run(self, iterations=1000):
        while self.nmc_status.i < iterations:
            self.nmc_status = self.sample()

        #Collect last trace
        for site in self.nmc_status.params.keys():
            for i in range(len(self.nmc_status.params[site])):
                self.acc_trace[site+str(i)].append(self.nmc_status.params[site][i])

        return self.nmc_status
