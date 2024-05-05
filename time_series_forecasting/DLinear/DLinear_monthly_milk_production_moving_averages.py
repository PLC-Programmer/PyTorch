# DLinear_monthly_milk_production_moving_averages.py
"""
small utility to check what kernel size of a simple moving average
causes the lowest error metric with a strongly seasonal time series
"""
#
# 2024-05-04/05
#
# idea from here:
#   Forecasting with moving averages, Robert Nau, Fuqua School of Business, Duke University, August 2014
#
#
# run on Ubuntu: $ python3 DLinear_monthly_milk_production_moving_averages.py
#
#
# test on Ubuntu 22.04.4 LTS, Python 3.10.12, torch 2.1.0: OK
#
#
# to-do:
#   -


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from sklearn.metrics import mean_squared_error
import sys
import statsmodels  # for ACF

import torch as T


TITLE = "Simple Moving Averages (SMAs) of a seasonal time series"
TITLE2 = "(last year is reserved for testing)"
NAME1 = "Monthly Milk Production (in pounds)"
FILENAME1 = 'SMAs_monthly_milk_production.png'

milk = pd.read_csv('monthly-milk-production-pounds-p.csv', index_col=0, parse_dates=True)
# print("Head of milk:\n", milk.head())

# total time series length: 14 full years
TOTAL_MONTHS      = len(milk)
LOOKBACK_WINDOW   = 12 # this represents the full year 1975 --> indices 156..167

# LOSS_FUNCTION     = 'LogCosh'
LOSS_FUNCTION     = 'mse'  # Mean-Squared-Error

N_MA              = 3  # SMA's with different kernel sizes:
KERNEL_SIZE       = [12,15,18]

# some constants:
DOUBLE  = T.as_tensor(2.0)
LOG_TWO = T.as_tensor(np.log(2.0))



##################################################################
#
# data preparations
#
milk = milk.values  # np array for simplicity


##################################################################
#
# define the model
#
class SMANet(T.nn.Module):
    '''
    simple moving average
    '''
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size
        self.sma = T.nn.AvgPool1d(kernel_size=kernel_size, stride=1)

    def forward(self, x):
        # the SMA's need some padding on the LHS of the time series:
        # https://pytorch.org/docs/stable/generated/torch.nn.AvgPool1d.html
        pad_lhs = T.nn.ConstantPad1d((self.kernel_size-1,0), (x[0,0]))
        x_padded = pad_lhs(x)
        sma = self.sma(x_padded)
        return sma
##################################################################



##################################################################
#
def myCustomLoss(my_outputs, my_labels):
    '''
    my loss function: logcosh
    https://neptune.ai/blog/pytorch-loss-functions
    https://jiafulow.github.io/blog/2021/01/26/huber-and-logcosh-loss-functions/
    '''
    zeros = T.zeros_like(my_outputs)
    error = T.sub(my_outputs, my_labels)

    sp = T.nn.Softplus()
    t1a = sp(-T.multiply(DOUBLE, error))
    t1b = sp(T.multiply(DOUBLE, error))
    t2 = T.add(error,LOG_TWO)

    positive_branch = T.add(t1a, t2)
    negative_branch = T.sub(t1b, t2)

    my_outputs = T.mean(T.where(error < zeros, negative_branch, positive_branch))

    return my_outputs
#
##################################################################



##################################################################
#
# do for a bunch of MA's
# - initializing the specific SMA model
# - forward the specific SMA
# - calculating a forecast error metric
# - plotting and saving a common plot

x_train = milk[:-LOOKBACK_WINDOW].reshape(-1)  # 13 full years

# convert to tensors:
x_train_tensor  = T.from_numpy(x_train).reshape(1,len(x_train))  # at least a 2-dim tensor is needed

sma_error_metric = np.empty([N_MA])
sma_var          = np.empty([N_MA])
sma_values       = [T.from_numpy(np.empty(len(x_train_tensor.reshape(-1)))).type(T.FloatTensor) for i in range(N_MA)]


for h in range(N_MA):
    print(f'\nSMA with kernel size {KERNEL_SIZE[h]} starts...')

    model = SMANet(KERNEL_SIZE[h])  # initialize the specific SMA model

    input = x_train_tensor
    sma = model(input)  # forward
    sma_values[h] = sma

    # calculating the error metric:
    # err = myCustomLoss(sma, input)
    err = mean_squared_error(sma, input)
    print(f'  error of SMA = {err:.1f}')
    sma_error_metric[h] = err
    sma_var[h]          = T.var(sma.type(T.FloatTensor))



# plotting the results:
fig = plt.figure(figsize=(16, 9))
plt.suptitle(TITLE, fontsize=18)
plt.title(TITLE2, fontsize=14)

plt.plot(milk, label=NAME1, color='black', linewidth=0.75)

for h in range(N_MA):
    if h >= 0:  # just a curve on/off control
        plt.plot(sma_values[h].reshape(-1), label=f'SMA with kernel size {KERNEL_SIZE[h]}: error metric={sma_error_metric[h]:.1f}')
        plt.plot([],[], ' ', label=f'  variance of SMA = {sma_var[h]:.1f}')

# extra labels:
plt.plot([], [], ' ', label=f'loss function={LOSS_FUNCTION}')
plt.legend(fontsize=13)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.grid()
plt.tight_layout()
# plt.show() <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
plt.savefig(FILENAME1)
plt.close()



##################################################################
#
# ACF (autocorrelation function) + power spectrum of milk time series
fig = plt.figure(figsize=(16, 9))
plt.suptitle(NAME1, fontsize=18)
plt.title('autocorrelation function (ACF)', fontsize=18)

milk_acf = statsmodels.tsa.stattools.acf(milk, nlags=60)

plt.plot(milk_acf)
plt.xticks(np.arange(0, 60, step=1))
plt.xlabel('month')
plt.grid()
plt.tight_layout()
plt.show()


breakpoint()


# end of DLinear_monthly_milk_production_moving_averages.py
