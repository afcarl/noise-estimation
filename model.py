from pymc3 import Bernoulli, Beta, Deterministic, Model, Metropolis, sample, find_MAP
import numpy as np
import cPickle as pickle
import itertools
import theano.tensor as T 
from theano.compile.ops import as_op

def flatten(l):
    return list(itertools.chain.from_iterable(l))
            
# p(noisy-label = 1)

if __name__ == "__main__":
    S = 10
    N = 10
    alpha_p = 1
    beta_p = 1
    alpha_e = 1
    beta_e = 10
    observed_labels = np.array(np.random.rand(S,N) > 0.5, dtype=int)

    @as_op(itypes=[T.lvector, T.dvector, T.dvector], otypes=[T.dmatrix])
    def p_pos(l, e_pos, e_neg):
        return np.outer(1-l,e_neg)  + np.outer(l, 1-e_pos)

    with Model() as model:
            p = Beta('p', alpha=alpha_p, beta=beta_p) #prior on true label
            l = Bernoulli('l', p=p, shape=S) #true label
            e_pos = Beta('e_pos', alpha_e, beta_e, shape=N) # error rate if label = 1
            e_neg = Beta('e_neg', alpha_e, beta_e, shape=N) # error rate if label = 0

            #r = Deterministic('r', p_pos(l, e_pos, e_neg))
            r = Deterministic('r',T.outer(1-l,e_neg)  + T.outer(l, 1-e_pos)) 
            
            #noisy label
            noisy_label = Bernoulli('noisy_label', p = r, shape=(S,N), observed=observed_labels)
            start = find_MAP()
            step = Metropolis()
            trace = sample(2000, step=step, start=start, progressbar=True)
            traceplot(trace)

