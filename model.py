from pymc import Bernoulli, Beta, deterministic, MCMC, Deterministic
import numpy as np
import cPickle as pickle
import itertools

def flatten(l):
    return list(itertools.chain.from_iterable(l))
            
# p(noisy-label = 1)
def p_pos(l, e_pos, e_neg):
    if l == 1:
        return 1-e_pos #don't make a mistake 
    else:
        return e_neg   #make a mistake

if __name__ == "__main__":
    S = 10
    N = 10
    alpha_p = 1
    beta_p = 1
    alpha_e = 1
    beta_e = 10
    observed_f = np.array(np.random.rand(S,N) > 0.5, dtype=int)

    p = Beta('p', alpha=alpha_p, beta=beta_p) #prior on true label
    l = [Bernoulli('l_'+str(i), p=p) for i in xrange(S)] #true label
    e_pos = [Beta('e_pos'+str(j), alpha_e, beta_e) for j in xrange(N)] # error rate if label = 1
    e_neg = [Beta('e_neg'+str(j), alpha_e, beta_e) for j in xrange(N)] # error rate if label = 0

    rate = [[] for _ in xrange(S)]
    for i in xrange(S):
        for j in xrange(N):
            r = Deterministic(eval = p_pos,
                                    name = 'p_pos_'+str(i)+'m'+str(j),
                                    parents = {'l': l[i],
                                               'e_pos': e_pos[j],
                                               'e_neg': e_neg[j]},
                                    doc = 'The rate of errors.',
                                    trace = True,
                                    verbose = 0,
                                    dtype=float,
                                    plot=False,
                                    cache_depth = 2)

            rate[i].append(r)
    
    #noisy label
    noisy_label = np.array([[Bernoulli('f_'+str(i)+','+str(j), p = rate[i][j],
                             value=observed_f[i,j]) for j in xrange(N)] for i in xrange(S)],
                          dtype=object) 

    variables = set(l) | set([p]) | set(e_pos) | set(e_neg) | set(flatten(rate))
    model = MCMC(variables)
    model.sample(iter=100, burn=10, thin=2)
    model.write_csv("out.csv", variables=["e_pos"+str(j) for j in xrange(N)]+["e_neg"+str(j) for j in xrange(N)])