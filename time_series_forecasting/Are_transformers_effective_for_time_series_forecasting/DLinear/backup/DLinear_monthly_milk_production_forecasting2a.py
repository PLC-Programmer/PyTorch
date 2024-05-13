# DLinear_monthly_milk_production_forecasting2a.py
"""
Using an additive linear model for deep learning based
time series forecasting.
However, the decomposition is done like this: the trend,
based on a simple moving average, is only calculated
once for the complete training period and then its
independent predictions are added to the predictions
of the trained remainder component.
(in sample): forecast the value of one time step ahead
"""
#
# 2024-05-04/06
#
#
# sources:
#   PyTorch:
#     Linear (y = xA + b): https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
#     Neural Network Training with Modules: https://pytorch.org/docs/stable/notes/modules.html#neural-network-training-with-modules
#   The univariate time series is from here:
#   https://github.com/philipperemy/keras-tcn/blob/master/tasks/time_series_forecasting.py
#
#
# run on Ubuntu: $ python3 DLinear_monthly_milk_production_forecasting2a.py
#
#
# test on Ubuntu 22.04.4 LTS, Python 3.10.12, torch 2.1.0: OKish, not doing much harm
#
#
# test results:
#  - this model is not doing better than the simple Linear model with this time series
#
#
# to-do:
#   -
#
#
# pip install plotly
# pip install -U kaleido  # to save static images of your plots


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from sklearn.metrics import mean_squared_error

import torch as T

import optuna  # hyperparameter (hp) tuning of PyTorch models with Optuna
import plotly  # for optuna plots


TUNE_ON = False   # default: False
TUNE_TRIALS = 10  # 10 is enough


TITLE = "Decomposition (trend + seasons) linear model (two additive one-layers) for time series forecasting (v2)"
NAME1 = "Monthly Milk Production (in pounds)"
FILENAME1    = 'monthly_milk_production_forecasting2_'
FILENAME2    = 'monthly_milk_production_forecasting_stats2.txt'

milk = pd.read_csv('monthly-milk-production-pounds-p.csv', index_col=0, parse_dates=True)
# print("Head of milk:\n", milk.head())

# total time series length: 14 full years
TOTAL_MONTHS      = len(milk)
LOOKBACK_WINDOW   = 12 # this represents the full year 1975 --> indices 156..167

KERNEL_SIZE       = 12  # length of look-back window for moving average; org: 12

LEARNING_RATE_TREND     = 0.01   # to be tuned
LEARNING_RATE_REMAINDER = 0.005  # to be tuned

EPOCHS            = 20*LOOKBACK_WINDOW+1  # org 10*LOOKBACK_WINDOW
OPTIMIZER         = 'Adam'
LOSS_FUNCTION     = 'LogCosh'
LOSS_LIMIT        = 1.0  # for a mean epochs target of roughly 58 with 10 different models

N_PRED            = 1  # org 10 prediction curves; 50 is enough for manual parameter testing

# some constants:
DOUBLE  = T.as_tensor(2.0)
LOG_TWO = T.as_tensor(np.log(2.0))


##################################################################
#
# data preparations
#
milk = milk.values  # np array for simplicity


# for simple training:
def split_dataset2a(data):
    '''
    split a univariate dataset into:
      [training period]] + [forecasting period, in sample]] = [total data]
    '''
    train_x, train_y = [], []
    for i in range(LOOKBACK_WINDOW,TOTAL_MONTHS-LOOKBACK_WINDOW,1):  # range() stops before a specified stop number
        train_x.append(data[i-LOOKBACK_WINDOW:i])
        train_y.append(data[i])
    # train_x = np.array(train_x)
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    return train_x, train_y
##################################################################



##################################################################
#
# define the models
#
class SMANet(T.nn.Module):
    '''
    calculate a simple moving average
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


class Trend_Net(T.nn.Module):
    '''
    this linear layer only takes care of
    the trend component
    '''
    def __init__(self):
        super().__init__()
        self.lin = T.nn.Linear(LOOKBACK_WINDOW, 1)

    def forward(self, x):
        z = self.lin(x)
        return z


class Remainder_Net(T.nn.Module):
    '''
    this linear layer only takes care of
    the remainder (seasonal) component
    '''
    def __init__(self):
        super().__init__()
        self.lin = T.nn.Linear(LOOKBACK_WINDOW, 1)

    def forward(self, x):
        z = self.lin(x)
        return z
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
# do for a couple of models:
# - build the SMA
# - calculate the x input data for the remainder linear layer = x,raw - x trend data
# - train both linear layers for the trend and  remainder (seasonal) component
# - do independent predictions for both the trend and remainder component
# - calculate a total forecast error metric
# - plot and save the model plot
#
#
# build the SMA for the complete training period:
x_train_sma = milk.reshape(-1)

# convert to tensors:
x_train_sma_tensor  = T.from_numpy(x_train_sma).reshape(1,len(x_train_sma))  # at least a 2-dim tensor is needed
sma_values          = T.from_numpy(np.empty(len(x_train_sma_tensor.reshape(-1)))).type(T.FloatTensor)

model_sma  = SMANet(KERNEL_SIZE)  # initialize the SMA model
sma_values = model_sma(x_train_sma_tensor)  # sma_values: same full length as milk

# make a training data set for later trend (moving average) prediction:
x_train_trend, y_train_trend = split_dataset2a(sma_values[0].numpy())
# convert to tensors:
x_train_trend_tensor = T.from_numpy(x_train_trend).type(T.FloatTensor)
y_train_trend_tensor = T.from_numpy(y_train_trend).type(T.FloatTensor)

# calculate the detrended x input data for the linear layer:
milk_remainder = x_train_sma - sma_values[0].numpy()  # milk_remainder: same full length as milk

x_train_remainder, y_train_remainder = split_dataset2a(milk_remainder)
# convert to tensors:
x_train_remainder_tensor = T.from_numpy(x_train_remainder).type(T.FloatTensor)
y_train_remainder_tensor = T.from_numpy(y_train_remainder).type(T.FloatTensor)


model_trend     = Trend_Net()  # initialize the trend model
model_remainder = Remainder_Net()  # initialize the remainder model

optimizer_trend = T.optim.Adam(model_trend.parameters(), lr=LEARNING_RATE_TREND)
optimizer_remainder = T.optim.Adam(model_remainder.parameters(), lr=LEARNING_RATE_REMAINDER)


pred_error_metric = np.empty([N_PRED])
best_error_metric = 1e20
best_error_model = 0
loss_limit_epoch_trend = list(range(N_PRED))
loss_limit_epoch_remainder = list(range(N_PRED))

# save some stats to a text file:
stat_header = f'model # and {LOSS_FUNCTION} value:'
print("\n" + stat_header)
f2 = open(FILENAME2, 'w')
f2.write(stat_header + '\n')


for h in range(N_PRED):

    ################################################################
    # (A) train the trend (moving average) model
    print(f'\n\n  trend (moving average) model #{h} training starts...')

    # ini's:
    loss_limit_epoch_trend[h] = EPOCHS
    model_trend.train()  # switch the module to training mode == default mode

    for i in range(EPOCHS):

        for j in range(len(x_train_trend_tensor)):
            y_pred = model_trend(x_train_trend_tensor[j])  # forward

            loss = myCustomLoss(y_pred, y_train_trend_tensor[j])

            model_trend.zero_grad()
            loss.backward()
            optimizer_trend.step()

        loss = loss.item()
        print(f'  epoch = {i} of {EPOCHS} epochs: loss = {loss:.2f}')
        if loss < LOSS_LIMIT:
            loss_limit_epoch_trend[h] = i+1  # i starts with 0
            print(f'  loss limit reached after epoch #{i+1}')
            break

    # do a trend prediction
    #
    # after training, switch the module to eval mode to do inference, compute performance metrics, etc:
    model_trend.eval()

    # do some initializations first:
    x_test_tensor = T.empty(2*LOOKBACK_WINDOW)  # past of length LOOKBACK_WINDOW + future (in sample) of length LOOKBACK_WINDOW
    x_test_ini = sma_values[0][-2*LOOKBACK_WINDOW:-LOOKBACK_WINDOW]
    x_test_tensor[0:LOOKBACK_WINDOW] = x_test_ini

    pred_value_trend = np.empty(LOOKBACK_WINDOW)

    for i in range(LOOKBACK_WINDOW,2*LOOKBACK_WINDOW):
        # at least a 2-dim tensor is needed:
        input = x_test_tensor[i-LOOKBACK_WINDOW:i].reshape(1,LOOKBACK_WINDOW)

        output = model_trend(input)

        pred_value_trend[i-LOOKBACK_WINDOW] = output.detach().numpy()[0,0]  # tensor to numpy scalar

        # add predicted value to the test tensor!
        # so, no update of test data with in-sample-data:
        x_test_tensor[i] = output[0].detach()[0]

    pred_value_trend_total = np.empty(TOTAL_MONTHS)
    pred_value_trend_total[:] = np.nan
    pred_value_trend_total[-LOOKBACK_WINDOW:] = pred_value_trend


    ################################################################
    # (B) train the remainder (seasonal) model
    print(f'\n  remainder (seasonal) model #{h} training starts...')

    # ini's:
    loss_limit_epoch_remainder[h] = EPOCHS
    model_remainder.train()  # switch the module to training mode == default mode

    for i in range(EPOCHS):

        for j in range(len(x_train_remainder_tensor)):
            y_pred = model_remainder(x_train_remainder_tensor[j])  # forward

            loss = myCustomLoss(y_pred, y_train_remainder_tensor[j])

            model_remainder.zero_grad()
            loss.backward()
            optimizer_remainder.step()

        loss = loss.item()
        print(f'  epoch = {i} of {EPOCHS} epochs: loss = {loss:.2f}')
        if loss < LOSS_LIMIT:
            loss_limit_epoch_remainder[h] = i+1  # i starts with 0
            print(f'  loss limit reached after epoch #{i+1}')
            break

    # do a remainder (seasonal) prediction
    #
    # after training, switch the module to eval mode to do inference, compute performance metrics, etc:
    model_remainder.eval()

    # do some initializations first:
    x_test_ini = T.from_numpy(milk_remainder[-2*LOOKBACK_WINDOW:-LOOKBACK_WINDOW])
    x_test_tensor[0:LOOKBACK_WINDOW] = x_test_ini

    pred_value_remainder = np.empty(LOOKBACK_WINDOW)

    for i in range(LOOKBACK_WINDOW,2*LOOKBACK_WINDOW):
        # at least a 2-dim tensor is needed:
        input = x_test_tensor[i-LOOKBACK_WINDOW:i].reshape(1,LOOKBACK_WINDOW)

        output = model_remainder(input)

        pred_value_remainder[i-LOOKBACK_WINDOW] = output.detach().numpy()[0,0]  # tensor to numpy scalar

        # add predicted value to the test tensor!
        # so, no update of test data with in-sample-data:
        x_test_tensor[i] = output[0].detach()[0]

    pred_value_remainder_total = np.empty(TOTAL_MONTHS)
    pred_value_remainder_total[:] = np.nan
    pred_value_remainder_total[-LOOKBACK_WINDOW:] = pred_value_remainder


    ################################################################
    # (C) add outputs of both models:
    pred_value_total = pred_value_trend_total + pred_value_remainder_total


    ################################################################
    # calculating a forecast error metric:
    err = myCustomLoss(T.tensor(milk[-LOOKBACK_WINDOW:]), T.tensor(pred_value_total[-LOOKBACK_WINDOW:]))
    print(f'\n  error of prediction = {err:.1f}')
    pred_error_metric[h] = err
    if err < best_error_metric:
        best_error_metric = err
        best_error_model  = h


    ################################################################
    # plotting this model:
    fig = plt.figure(figsize=(16, 9))

    plt.suptitle(TITLE, fontsize=18)

    plt.plot(milk, label=NAME1, color='black', linewidth=0.75)
    # plt.plot(pred_value_total, color='orange', label='1 month prediction, re-using predicted values')


    plt.plot(sma_values[0], color='blue', label=f'simple moving average (trend) with {KERNEL_SIZE} months kernel size', linewidth=0.75)
    # plt.plot(pred_value_trend_total, color='blue', label='trend prediction', marker='o', linestyle='')


    # extra labels:
    # plt.plot([], [], ' ', label=f'*** model #{h+1} out of {N_PRED} models ***')
    # plt.plot([], [], ' ', label=f'error metric={pred_error_metric[h]:.1f}')
    # plt.plot([], [], ' ', label=f'lookback window={LOOKBACK_WINDOW} months')
    #
    # plt.plot([], [], ' ', label=f'epochs={EPOCHS}, early stopping at < {LOSS_LIMIT} loss')
    # plt.plot([], [], ' ', label=f'  trend: stopped early after {loss_limit_epoch_trend[h]} epochs')
    # plt.plot([], [], ' ', label=f'  remainder: stopped early after {loss_limit_epoch_remainder[h]} epochs')
    #
    # plt.plot([], [], ' ', label=f'optimizer={OPTIMIZER}, loss function={LOSS_FUNCTION}, learning rate={LEARNING_RATE_TREND}')
    # plt.plot([], [], ' ', label=f'optimizer={OPTIMIZER}, loss function={LOSS_FUNCTION}, learning rate={LEARNING_RATE_REMAINDER}')
    # plt.plot([], [], ' ', label=f'training period: {TOTAL_MONTHS-LOOKBACK_WINDOW} months on the LHS')
    plt.legend(fontsize=13)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.grid()
    plt.tight_layout()
    # plt.show()
    FILENAME1a = FILENAME1 + f'{h}'.zfill(2) + 'a.png'
    plt.savefig(FILENAME1a)
    plt.close()

    # save some stats to a text file:
    stat_row = f'  model #{h}: error metric = {pred_error_metric[h]:.1f}'
    f2.write(stat_row + '\n')


# finish the stats file:
f2.write('---------------' + '\n')
f2.write(f'TOTAL_MONTHS = {TOTAL_MONTHS}' + '\n')
f2.write(f'LOOKBACK_WINDOW = {LOOKBACK_WINDOW}' + '\n')
f2.write(f'KERNEL_SIZE = {KERNEL_SIZE}' + '\n')
f2.write(f'training cutoff = {TOTAL_MONTHS-LOOKBACK_WINDOW-1}' + '\n')
f2.write('---------------' + '\n')
f2.write(f'OPTIMIZER = {OPTIMIZER}' + '\n')
f2.write(f'LOSS_FUNCTION = {LOSS_FUNCTION}' + '\n')
f2.write(f'LOSS_LIMIT = {LOSS_LIMIT}' + '\n')
f2.write(f'LEARNING_RATE_TREND = {LEARNING_RATE_TREND}' + '\n')
f2.write(f'LEARNING_RATE_REMAINDER = {LEARNING_RATE_REMAINDER}' + '\n')
f2.write(f'TUNE_ON = {TUNE_ON}' + '\n')
f2.write('---------------' + '\n')
f2.write(f'EPOCHS (trend training) = {EPOCHS}' + '\n')
f2.write(f'  mean epochs = {np.mean(loss_limit_epoch_trend):.1f}' + '\n')
f2.write(f'  max epochs = {np.max(loss_limit_epoch_trend):.1f}' + '\n')
f2.write(f'  min epochs = {np.min(loss_limit_epoch_trend):.1f}' + '\n')
f2.write(f'EPOCHS (remainder (seasonal) training) = {EPOCHS}' + '\n')
f2.write(f'  mean epochs = {np.mean(loss_limit_epoch_remainder):.1f}' + '\n')
f2.write(f'  max epochs = {np.max(loss_limit_epoch_remainder):.1f}' + '\n')
f2.write(f'  min epochs = {np.min(loss_limit_epoch_remainder):.1f}' + '\n')
f2.write('---------------' + '\n')
f2.write(f'Mean of the errors of all {N_PRED} models = {np.mean(pred_error_metric):.1f}' + '\n')
f2.write(f'Variance of the errors of all {N_PRED} models = {np.var(pred_error_metric):.1f}' + '\n')
f2.close()


print(f'\nMean of the errors of all {N_PRED} models = {np.mean(pred_error_metric):.1f}')
print(f'Variance of the errors of all {N_PRED} models = {np.var(pred_error_metric):.1f}')
print(f'\n=> best model is #{best_error_model} with an error of {best_error_metric:.1f}')

print(f'\nTrend model:')
print(f'  mean of epoch when the loss limit has been reached = {np.mean(loss_limit_epoch_trend):.1f}')
print(f'  maximum epoch when the loss limit has been reached = {np.max(loss_limit_epoch_trend):.1f}')
print(f'  minimum epoch when the loss limit has been reached = {np.min(loss_limit_epoch_trend):.1f}')

print(f'Remainder (seasonal) model:')
print(f'  mean of epoch when the loss limit has been reached = {np.mean(loss_limit_epoch_remainder):.1f}')
print(f'  maximum epoch when the loss limit has been reached = {np.max(loss_limit_epoch_remainder):.1f}')
print(f'  minimum epoch when the loss limit has been reached = {np.min(loss_limit_epoch_remainder):.1f}')


breakpoint()


# end of DLinear_monthly_milk_production_forecasting2a.py
