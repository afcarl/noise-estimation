import numpy as np
import model
import tensor_decomp
import matplotlib.pyplot as plt
from pymc.Matplot import plot

p = 0.3 #probability of true label being 1
K = 3 #number of noisy classifiers
train_samples= 500
generate_samples= 50000

noisy_label_rates_p = np.array([0.1, 0.2, 0.3])
noisy_label_rates_n = np.array([0.2, 0.2, 0.1])

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
samples = np.vstack(samples)

counts = np.zeros((2,)*K)
for s in samples:
  counts[tuple(s)] += 1

print 'true values'
print p, noisy_label_rates_p, noisy_label_rates_n
print model.estimate_failures_from_counts(counts=counts, n_samples=generate_samples)
print model.estimate_failures(samples=samples, n_samples=generate_samples)
print tensor_decomp.estimate_failures(samples=samples)
print tensor_decomp.estimate_failures_from_counts(counts=counts)
