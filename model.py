from pymc3 import *
import numpy as np
import theano.tensor as T 
import matplotlib.pyplot as plt

if __name__ == "__main__":
  S = 3
  N = 3
  alpha_p = 1
  beta_p = 1
  alpha_e = 1
  beta_e = 10
  observed_labels = np.array(np.random.rand(S,N) > 0.5, dtype=int)

  with Model() as model:
    p = Beta('p', alpha=alpha_p, beta=beta_p) #prior on true label
    l = Bernoulli('l', p=p, shape=S) #true label
    e_pos = Beta('e_pos', alpha_e, beta_e, shape=N) # error rate if label = 1
    e_neg = Beta('e_neg', alpha_e, beta_e, shape=N) # error rate if label = 0
    r = Deterministic('r',T.outer(1-l,e_neg)  + T.outer(l, 1-e_pos))  # P(noisy_label = 1) = e_neg if l == 0; 1-e_pos if l == 1
    noisy_label = Bernoulli('noisy_label', p = r, shape=(S,N), observed=observed_labels)

    start = find_MAP()
    step = Metropolis()
    trace = sample(2000, step=step, start=start, progressbar=True)
    traceplot(trace)
    plt.show()
