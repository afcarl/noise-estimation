import numpy as np


def solve_quadratic(a,b,c):
    q = (-b + np.lib.scimath.sqrt(b**2 - 4*a*c))/(2*a)
    r = (-b - np.lib.scimath.sqrt(b**2 - 4*a*c))/(2*a)
    
    if b**2 - 4*a*c < 0:
        #print "COMPLEX"
        q = np.real(q) + np.random.rand()
        r = np.real(r) + np.random.rand()

#        print q,r
    return q, r

def decompose(m, method=None):
    if method == "SVD":
        return np.linalg.svd(m)
    u = np.zeros((2,2))
    v = np.zeros((2,2))

    u[0,0] = 1 #set the scaling
    v[0,0] = m[0,0]
    u[1,0] = m[1,0]/m[0,0]
    v[0,1] = m[0,1]

    return (np.matrix(u), [1,1], np.matrix(v))

def learnParams_decompose(P, fix_values=0, include_mixture=0):

    X1 = np.matrix(P[0,:,:])
    X2 = np.matrix(P[1,:,:])

    Y2 = X2*np.linalg.pinv(X1)

    a = 1
    b = -np.trace(Y2)
    c = np.linalg.det(Y2)

    l1, l2 = solve_quadratic(a,b,c)

    (u1,e,v1) = decompose((l1 - l2)**(-1) * (X2 - l2*X1))
    (u2,f,v2) = decompose(-(l1 - l2)**(-1) * (X2 - l1*X1))

    u1 = (u1*np.diag(e))[:,0]
    u2 = (u2*np.diag(f))[:,0]
    
    v1 = v1[0,:]
    v2 = v2[0,:]
    
    if fix_values:

        u1 = np.clip(u1, 10**(-6), float("inf"))
        u2 = np.clip(u2, 10**(-6), float("inf"))
        v1 = np.clip(v1, 10**(-6), float("inf"))
        v2 = np.clip(v2, 10**(-6), float("inf"))
        l1 = np.clip(l1, 10**(-6), float("inf"))
        l2 = np.clip(l2, 10**(-6), float("inf"))

    U_coeff = u1.sum()*v1.sum()*(1+l1)
    V_coeff = u2.sum()*v2.sum()*(1+l2)

    u1 /= u1.sum()
    u2 /= u2.sum()
    v1 /= v1.sum()
    v2 /= v2.sum()
    l1 /= 1+l1
    l2 /= 1+l2

    A = np.hstack((u1,u2))
    B = np.vstack((v1,v2))

    U = U_coeff*np.array([A*np.diag([1-l1,0])*B, A*np.diag([l1,0])*B])
    V = V_coeff*np.array([A*np.diag([0,1-l2])*B, A*np.diag([0,l2])*B])

    prior_est= U_coeff/(U_coeff+V_coeff)
    noise_est = [1-l2, u2[0,0], v2[0,0]]
    failure_est = [(1-l1)/noise_est[0], u1[0,0]/noise_est[1], v1[0,0]/noise_est[2]]

    return prior_est, failure_est, noise_est

def estimate_failures(samples):
  #build cooccurence matrix
  S,N = samples.shape
  assert N == 3
  P = np.zeros((2,)*N)
  for s in samples:
    P[tuple(s)] += 1

  return estimate_failures_from_tensor(P)

def estimate_failures_from_tensor(P):
  prior, failures, noise = learnParams_decompose(P, fix_values=1)
  err_pos = [f*n for f,n in zip(failures, noise)]
  err_neg = [1-n for n in noise]

  return prior, err_pos, err_neg
