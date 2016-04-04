from pymc3 import *
import numpy as np
import theano.tensor as T 

def estimate_failures(observed_labels, samples=100000, alpha_p=1, beta_p=1, alpha_e=1, beta_e=10, burn=100, init = {}):
  S,N = observed_labels.shape
  models = {}
  with Model() as models[0]:
    p = Beta('p', alpha=alpha_p, beta=beta_p, observed=0.3) #prior on true label
    l = Bernoulli('l', p=p, shape=S) #true label
    e_pos = Beta('e_pos', alpha_e, beta_e, shape=N) # error rate if label = 1
    e_neg = Beta('e_neg', alpha_e, beta_e, shape=N) # error rate if label = 0
    r = Deterministic('r',T.outer(1-l,e_neg)  + T.outer(l, 1-e_pos))  # P(noisy_label = 1) = e_neg if l == 0; 1-e_pos if l == 1
    noisy_label = Bernoulli('noisy_label', p = r, shape=(S,N), observed=observed_labels)

    start=find_MAP()
    for k,v in init.items():
      start[k] = v
    #print start
    step = Metropolis()
    trace = sample(draws=samples, step=step, progressbar=True)

  return models, trace
