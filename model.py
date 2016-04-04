from pymc import Bernoulli, Beta, deterministic, MCMC, Deterministic
import numpy as np



def estimate_failures(samples, #samples from noisy labelers
                      n_samples=10000, #number of samples to run MCMC for
                      burn=None, #burn-in. Defaults to n_samples/2
                      thin=10, #thinning rate. Sample every k samples from markov chain 
                      alpha_p=1, beta_p=1, #beta parameters for true positive rate
                      alpha_e=1, beta_e=10 #beta parameters for noise rates
                      ):

  if burn is None:
    burn = n_samples / 2

  S,N = samples.shape
  p = Beta('p', alpha=alpha_p, beta=beta_p) #prior on true label
  l = Bernoulli('l', p=p, size=S)
  e_pos = Beta('e_pos', alpha_e, beta_e, size=N) # error rate if label = 1
  e_neg = Beta('e_neg', alpha_e, beta_e, size=N) # error rate if label = 0

  @deterministic(plot=False)
  def noise_rate(l=l, e_pos=e_pos, e_neg=e_neg):
    #probability that a noisy labeler puts a label 1
    return np.outer(l, 1-e_pos) + np.outer(1-l, e_neg)

  noisy_label = Bernoulli('noisy_label', p=noise_rate, size=samples.shape, value=samples, observed=True)
  variables = [l, e_pos, e_neg, p, noisy_label, noise_rate]
  model = MCMC(variables, verbose=3)
  model.sample(iter=n_samples, burn=burn, thin=thin)
  return model
