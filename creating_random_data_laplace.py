# creating_random_data_laplace.py
"""
This module creates a tensor with random data
following a Laplace distribution
"""
#
# 2022-11-22
#
# sources:
#   https://pytorch.org/docs/stable/distributions.html
#   https://pytorch.org/docs/stable/distributions.html#laplace
#
#
# test on Ubuntu 22.04.1 LTS, Python 3.10.6, torch 1.12.1+cpu: OK
#


import torch
import matplotlib.pyplot as plt
from torch.distributions import Laplace

LOC=0.0  # mean of the distribution
SCALE=1.0  # scale of the distribution
SAMPLES=1000  # number of samples from the distribution

l = Laplace(torch.tensor([LOC]), torch.tensor([SCALE]))
s = l.sample()
print('\ns sample = \n', s)


u = torch.full((1,SAMPLES), l.rsample().item())
print('\nu tensor = \n', u)


t = torch.empty(SAMPLES)
for i in range(t.size(0)):
    print(i)
    t[i] = l.sample()

print('\nt tensor = \n', t)



fig = plt.figure()
plt.hist(t, bins = 50)

plt.title('A Laplace distribution with mean 0.0 and scale 1.0')

fig.savefig('creating_random_data_laplace.jpg')
plt.show()


# end of creating_random_data_laplace.py