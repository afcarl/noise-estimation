import numpy as np
from model import estimate_failures
import matplotlib.pyplot as plt
from pymc.Matplot import plot

p = 0.3 #probability of true label being 1
K = 3 #number of noisy classifiers
train_samples=1000
generate_samples=10000
burn=1000

noisy_label_rates_p = np.array([0.1, 0.1, 0.1])
noisy_label_rates_n = np.array([0.1, 0.1, 0.1])

samples = []
true_labels = []

for _ in xrange(train_samples):
  #generate a sample
  true_label = int(np.random.rand() < p)
  if true_label == 1: #noisy label is 1 if no error occurs
    noisy_label = np.array(np.random.rand(K) > noisy_label_rates_p, dtype=int)
  else: #noisy label is 1 if an error occurs
    noisy_label = np.array(np.random.rand(K) < noisy_label_rates_n, dtype=int)
  samples.append(noisy_label)
  true_labels.append(true_label)

model = estimate_failures(samples=np.vstack(samples), n_samples=generate_samples, burn=burn)
plot(model.trace('p'))
