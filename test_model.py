import numpy as np
import pymc3 as pm
from model import estimate_failures
import matplotlib.pyplot as plt
from pymc.Matplot import plot

def logodds(x):
  return np.log(x/(1-x))

p = 0.3
K = 3
train_samples=100
generate_samples=1000

noisy_label_rates_p = np.array([0, 0, 0])
noisy_label_rates_n = np.array([0, 0, 0])

samples = []
true_labels = []

for _ in xrange(train_samples):
  #generate sample
  true_label = int(np.random.rand() < p)
  if true_label == 1:
    noisy_label = np.array(np.random.rand(K) > noisy_label_rates_p, dtype=int)
  else:
    noisy_label = np.array(np.random.rand(K) < noisy_label_rates_n, dtype=int)
  samples.append(noisy_label)
  true_labels.append(true_label)

print samples
model = estimate_failures(samples=np.vstack(samples), n_samples=generate_samples, burn=100)
plot(model.trace('p'))

