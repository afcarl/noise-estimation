from pymc import Bernoulli, Beta, deterministic, MCMC, Deterministic, Multinomial
import numpy as np
import itertools

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
  model.write_csv('out.csv', ['p', 'e_pos', 'e_neg'])
  p = np.median(model.trace('p')[:])
  e_pos = np.median(model.trace('e_pos')[:],0)
  e_neg = np.median(model.trace('e_neg')[:],0)
  return p, e_pos, e_neg

def estimate_failures_from_counts(counts, #samples from noisy labelers
                      n_samples=10000, #number of samples to run MCMC for
                      burn=None, #burn-in. Defaults to n_samples/2
                      thin=10, #thinning rate. Sample every k samples from markov chain 
                      alpha_p=1, beta_p=1, #beta parameters for true positive rate
                      alpha_e=1, beta_e=10 #beta parameters for noise rates
                      ):

  if burn is None:
    burn = n_samples / 2

  S = counts.sum()
  N = len(counts.shape)

  p_label = Beta('p_label', alpha=alpha_p, beta=beta_p) #prior on true label
  e_pos = Beta('e_pos', alpha_e, beta_e, size=N) # error rate if label = 1
  e_neg = Beta('e_neg', alpha_e, beta_e, size=N) # error rate if label = 0

  print counts
  @deterministic(plot=False)
  def patterns(p_label=p_label, e_pos=e_pos, e_neg=e_neg):
    #probability that the noisy labelers output pattern p
    P = np.zeros((2,)*N)
    for pat in itertools.product([0,1], repeat=N):
      P[pat] = p_label*np.product([1-e_pos[i] if pat[i]==1 else e_pos[i] for i in xrange(N)])
      P[pat] += (1-p_label)*np.product([e_neg[i] if pat[i]==1 else 1-e_neg[i] for i in xrange(N)])
    assert np.abs(P.sum() - 1) < 1e-6
    return P.ravel()
    
  pattern_counts = Multinomial('pattern_counts',n=S, p=patterns, value=counts.ravel(), observed=True)
  variables = [p_label, e_pos, e_neg, patterns]
  model = MCMC(variables, verbose=3)
  model.sample(iter=n_samples, burn=burn, thin=thin)
  model.write_csv('out.csv', ['p_label', 'e_pos', 'e_neg'])
  p = np.median(model.trace('p_label')[:])
  e_pos = np.median(model.trace('e_pos')[:],0)
  e_neg = np.median(model.trace('e_neg')[:],0)
  return p, e_pos, e_neg
