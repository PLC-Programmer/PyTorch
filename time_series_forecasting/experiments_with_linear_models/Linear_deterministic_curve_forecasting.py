# Linear_deterministic_curve_forecasting.py
"""
Using linear layers for deep learning based time series forecasting.
Lookback window size is one full period of the deterministic curve.
In sample: forecast the value of one time step, that is one datapoint, ahead (recursive strategy).
"""
#
# 2024-05-09/10/11
#
#
# source:
#   https://github.com/tgchomia/ts/blob/main/Example.txt
#
#
# run on ubuntu-med3: $ python3 Linear_deterministic_curve_forecasting.py
#
#
# test on Ubuntu 22.04.4 LTS, Python 3.10.12, torch 2.1.0: OK (Lin+ReLU+Lin)
#
# test results:
#   one Linear layer: smooth prediction curve, but wild swings!
#     - a double LOOKBACK_WINDOW of 2 periods doesn't help => going back to one period
#   Linear layer + ReLU +  Linear layer: bomb!! --> pretty good forecasts!
#   - from: LOOKBACK_WINDOW, PREDICTION_LENGTH|ReLU|PREDICTION_LENGTH,1
#     to:   LOOKBACK_WINDOW, LOOKBACK_WINDOW|ReLU|LOOKBACK_WINDOW,1 yields the same result with 20 runs & LOSS_LIMIT 0.15
#           but needs more parameters
#
#
#
# to-do:
#   -
#


import datetime
import pandas as pd
import math
import numpy as np

import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

import torch as T
from collections import OrderedDict

import optuna  # hyperparameter (hp) tuning of PyTorch models with Optuna
import plotly  # for optuna plots

import secrets

from prettytable import PrettyTable


TUNE_ON = False  # default: False
TUNE_TRIALS = 5 # 10 is enough

NOISE_ON = False  # default: True; put some Gaussian noise to the basic, harmonic signal
NOISE_FACTOR = 0.5  # 50% noise level
NOISE_STANDARD_T = True  # Student's t distribution with degrees of freedom
NOISE_STANDARD_T_DF = 5.0
NOISE_TYPE = 't -distribution'   # gaussian, t -distribution


NAME1 = "cos(2*pi*t/3) + 0.75*sin(2*pi*t/5)"

TITLE     = "Experiments with linear layer(s) for univariate time series forecasting"
FILENAME2 = 'Linear_deterministic_curve_forecasting--'
FILENAME3 = 'Linear_deterministic_curve_forecasting_stats.txt'

N_DATAPOINTS = 700
# TRAIN_RATIO  = 0.85  # 600 from 700 datapoints
LOOKBACK_WINDOW   = 150  # 1 periods1 150 = 3 * 5 *10
PREDICTION_LENGTH = 100  #


# define model and op's parameters:
LEARNING_RATE = 0.001  # somehow tuned --> see optuna plots
EPOCHS        = LOOKBACK_WINDOW  #
OPTIMIZER     = 'Adam'
LOSS_FUNCTION = 'LogCosh'
LOSS_LIMIT    = 0.01  # LogCosh, Linear+reLU+Linear

N_PRED        = 3  # org 10 prediction curves; 10 is enough for manual parameter testing

# some constants:
DOUBLE  = T.as_tensor(2.0)
LOG_TWO = T.as_tensor(np.log(2.0))



##################################################################
#
# generate a deterministic curve
t = np.linspace(0.1, 70, num=N_DATAPOINTS, endpoint=True)

y = np.cos(2*np.pi*t/3) + 0.75*np.sin(2*np.pi*t/5)
y_max = np.max(y)

y_clean = y.copy()

if NOISE_ON:
    # np.random.seed(0)
    if NOISE_STANDARD_T:
        y_noise = np.random.default_rng().standard_t(NOISE_STANDARD_T_DF, size=len(y)) / y_max * NOISE_FACTOR  # std.dev = 1

    else:
        y_noise = np.random.randn(len(y)) / y_max * NOISE_FACTOR  # std.dev = 1

    y += y_noise

#
##################################################################



##################################################################
#
# data preparations

VALIDATION_CUTOFF = N_DATAPOINTS - PREDICTION_LENGTH
TRAINING_CUTOFF   = N_DATAPOINTS - PREDICTION_LENGTH - PREDICTION_LENGTH
total_datapoints  = len(y)


# for tuning:
def split_dataset3(data):
    '''
    split a univariate dataset into:
      [training period]] + [validation period]] + [forecasting period, in sample]] = [total data]
    '''
    # forecast = data[-PREDICTION_LENGTH:]
    # forecast = np.array(forecast)

    train_x, train_y = [], []
    for i in range(LOOKBACK_WINDOW,TRAINING_CUTOFF,1):
        train_x.append(data[i-LOOKBACK_WINDOW:i])
        train_y.append(data[i])
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    validation_x, validation_y = [], []
    for i in range(TRAINING_CUTOFF,VALIDATION_CUTOFF,1):
        validation_x.append(data[i-LOOKBACK_WINDOW:i])
        validation_y.append(data[i])
    validation_x = np.array(validation_x)
    validation_y = np.array(validation_y)

    return train_x, train_y, validation_x, validation_y  #  , forecast


# for simple training:
def split_dataset2a(data):
    '''
    split a univariate dataset into:
      [training period]] + [forecasting period, in sample]] = [total data]
    '''
    train_x, train_y = [], []
    for i in range(LOOKBACK_WINDOW,total_datapoints-PREDICTION_LENGTH ,1):  # range() stops before a specified stop number
        train_x.append(data[i-LOOKBACK_WINDOW:i])
        train_y.append(data[i])
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    return train_x, train_y
#
##################################################################


##################################################################
#
# define the model
#
class Net(T.nn.Module):
    '''
    1D convolution
    '''
    def __init__(self):
        super().__init__()

        self.layers = T.nn.Sequential(
                        OrderedDict([

                          # OK version:
                          ('lin1', T.nn.Linear(LOOKBACK_WINDOW, PREDICTION_LENGTH)),
                          ('relu', T.nn.ReLU()),
                          ('lin2', T.nn.Linear(PREDICTION_LENGTH, 1))

                          # experiments:
                          # ('lin1', T.nn.Linear(LOOKBACK_WINDOW, 1)),
                          #
                          # ('lin1', T.nn.Linear(LOOKBACK_WINDOW, LOOKBACK_WINDOW)),
                          # ('relu', T.nn.ReLU()),
                          # ('lin2', T.nn.Linear(LOOKBACK_WINDOW, 1)),

                        ]))

    def forward(self, x):
        z = self.layers(x)
        return z
#
##################################################################


##################################################################
#
def myCustomLoss(my_outputs, my_labels, type):
    '''
    my loss function: logcosh
    https://neptune.ai/blog/pytorch-loss-functions
    https://jiafulow.github.io/blog/2021/01/26/huber-and-logcosh-loss-functions/
    '''
    if type == 'logcosh':
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

    else:
        error = T.sub(my_outputs, my_labels)
        sp = T.multiply(error,error)
        my_outputs = T.mean(sp)

        return my_outputs
#
##################################################################


##################################################################
#
def count_parameters(model):
    '''
    DO THIS ONLY AFTER ALL MODEL HANDLING!!
    (this seems to mess up a model under operation)
    save structural data of the model to the stats file
    https://medium.com/the-owl/how-to-get-model-summary-in-pytorch-57db7824d1e3
    https://pypi.org/project/prettytable/
    '''
    table = PrettyTable(['Modules', 'Parameters'])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    mystring = table.get_string()
    return mystring, total_params
#
##################################################################



##################################################################
#
def tune_model(trial):
    '''
    find a tuned learning rate
    '''
    model = Net()  # initialize the model
    model.train()

    # hp/hyperparameter tuning of the learning rate:
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    optimizer = getattr(T.optim, OPTIMIZER)(model.parameters(), lr=lr)

    # get the dataset:
    x_train, y_train, x_val, y_val  = split_dataset3(y)
    # convert to tensors:
    x_train_tensor  = T.from_numpy(x_train).type(T.FloatTensor)
    y_train_tensor  = T.from_numpy(y_train).type(T.FloatTensor)
    x_val_tensor    = T.from_numpy(x_val).type(T.FloatTensor)
    y_val_tensor    = T.from_numpy(y_val).type(T.FloatTensor)

    for i in range(EPOCHS):
        # train the model:
        for j in range(len(x_train_tensor)):
            optimizer.zero_grad()

            input = x_train_tensor[j]
            y_pred = model(input)
            loss = myCustomLoss(y_pred, y_train_tensor[j], LOSS_FUNCTION)
            loss.backward()
            optimizer.step()

        # validate the model:
        model.eval()

        with T.no_grad():
            val_loss_batch = 0.0

            for j in range(len(x_val_tensor)):
                input = x_val_tensor[j].reshape(-1)
                y_pred = model(input)
                val_loss_batch += myCustomLoss(y_pred, y_val_tensor[j], LOSS_FUNCTION)

        val_loss_epoch = val_loss_batch / len(y_val_tensor)

        trial.report(val_loss_epoch, i)

        # handle pruning based on the intermediate value:
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss_epoch


if TUNE_ON:
    print(f'\nmodel tuning starts with {TUNE_TRIALS} trials......')

    study = optuna.create_study(direction="minimize")
    study.optimize(tune_model, n_trials=TUNE_TRIALS)

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
        #

    # LEARNING_RATE = trial.params['lr']

    fig1 = optuna.visualization.plot_optimization_history(study)  # needs plotly
    fig1.write_image("fig1_optim_history.png")

    fig2 = optuna.visualization.plot_param_importances(study) # this is important to figure out which hp is important
    fig2.write_image("fig2_optim_param_importance.png")

    fig3 = optuna.visualization.plot_slice(study)  # this gives a clear picture
    fig3.write_image("fig3_optim_slice.png")

    fig4 = optuna.visualization.plot_parallel_coordinate(study)
    fig4.write_image("fig4_optim_parallel.png")

    print("model tuning finished. Optimization plots saved.\n")

# end of tuning
#
##################################################################



##################################################################
#
# do for a couple of models:
# - more data preparations
# - building
# - training
# - prediction
# - calculating a forecast error metric
# - plotting and saving the model plot

# breakpoint()
x_train, y_train = split_dataset2a(y)

# convert to tensors:
x_train_tensor  = T.from_numpy(x_train).type(T.FloatTensor)
y_train_tensor  = T.from_numpy(y_train).type(T.FloatTensor)


pred_error_metric = np.empty([N_PRED])
best_error_metric = 1e20
best_error_model = 0
loss_limit_epoch = list(range(N_PRED))

# save some stats and model structure to a text file:
stat_header = f'model # and {LOSS_FUNCTION} value:'
f2 = open(FILENAME3, 'w')
f2.write(stat_header + '\n')


print()
for h in range(N_PRED):
    print(f'\nmodel #{h} training starts...')

    # ini's:
    loss_limit_epoch[h] = EPOCHS

    seed = secrets.randbits(32)  # 128 too high here
    T.manual_seed(seed)
    model = Net()  # initialize the model
    model.train()  # switch the module to training mode == default mode
    optimizer = T.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # train the model:
    for i in range(EPOCHS):

        for j in range(len(x_train_tensor)):
            input = x_train_tensor[j]

            y_pred = model(input)  # forward

            loss = myCustomLoss(y_pred, y_train_tensor[j], LOSS_FUNCTION)
            model.zero_grad()
            loss.backward()
            optimizer.step()

            # if not j % 2500:
            #     print("  loss = ", loss)

        loss = loss.item()
        print(f'  epoch = {i} of {EPOCHS} epochs: loss = {loss:.2f}')
        if loss < LOSS_LIMIT:
            loss_limit_epoch[h] = i+1  # i starts with 0
            print(f'  loss limit reached after epoch #{i+1}')
            break

    # after training, switch the module to eval mode to do inference, compute performance metrics, etc:
    model.eval()


    # do a prediction
    # do some initializations first:
    x_test_tensor = T.empty(LOOKBACK_WINDOW+PREDICTION_LENGTH)  # past of length LOOKBACK_WINDOW + future (in sample) of length PREDICTION_LENGTH
    x_test_ini = T.from_numpy(y[-LOOKBACK_WINDOW-PREDICTION_LENGTH:-PREDICTION_LENGTH])
    x_test_tensor[0:LOOKBACK_WINDOW] = x_test_ini

    pred_value = np.empty(PREDICTION_LENGTH)

    for i in range(LOOKBACK_WINDOW,LOOKBACK_WINDOW+PREDICTION_LENGTH):

        input = x_test_tensor[i-LOOKBACK_WINDOW:i]

        output = model(input)

        pred_value[i-LOOKBACK_WINDOW] = output.detach().numpy()[0]  # tensor to numpy scalar

        # add predicted value to the test tensor!
        # so, no update of test data with in-sample-data:
        x_test_tensor[i] = output

    pred_value_total = np.empty(total_datapoints)
    pred_value_total[:] = np.nan
    pred_value_total[-PREDICTION_LENGTH:] = pred_value

    # calculating a forecast error metric:
    err = myCustomLoss(T.tensor(y[-PREDICTION_LENGTH:]), T.tensor(pred_value), LOSS_FUNCTION)
    print(f'  error of prediction = {err:.2f}')
    pred_error_metric[h] = err
    if err < best_error_metric:
        best_error_metric = err
        best_error_model  = h


    # plotting this model:
    fig = plt.figure(figsize=(16, 9))

    plt.title(TITLE, fontsize=18)

    plt.plot(t, y, label=NAME1, color='black', linewidth=0.75)

    if NOISE_ON:
        plt.plot(t, y_noise, label=f'added noise of type {NOISE_TYPE}', color='red', linewidth=0.75)

    plt.plot(t, pred_value_total, color='orange', label=f'{PREDICTION_LENGTH} datapoints prediction')

    # extra labels:
    plt.plot([], [], ' ', label=f'*** model #{h+1} out of {N_PRED} models ***')
    plt.plot([], [], ' ', label=f'error metric={pred_error_metric[h]:.2f}')
    plt.plot([], [], ' ', label=f'lookback window={LOOKBACK_WINDOW} datapoints')

    plt.plot([], [], ' ', label=f'epochs={EPOCHS}, early stopping at < {LOSS_LIMIT} loss')
    plt.plot([], [], ' ', label=f'  stopped early after {loss_limit_epoch[h]} epochs')

    plt.plot([], [], ' ', label=f'optimizer={OPTIMIZER}, loss function={LOSS_FUNCTION}, learning rate={LEARNING_RATE}')
    plt.legend(fontsize=13)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.grid()
    plt.tight_layout()
    # plt.show()
    FILENAME2a = FILENAME2 + f'{h}'.zfill(2) + '.png'
    plt.savefig(FILENAME2a)
    plt.close()


    if NOISE_ON:
        # plot only the prediction:
        fig = plt.figure(figsize=(16, 9))

        plt.suptitle(TITLE, fontsize=18)
        plt.title('(only prediction shown here)', fontsize=18)

        plt.plot(t, y_clean, label='ground truth', color='blue', linewidth=0.6)

        plt.plot(t, pred_value_total, color='orange', label=f'{PREDICTION_LENGTH} datapoints prediction')

        # extra labels:
        plt.plot([], [], ' ', label=f'*** model #{h+1} out of {N_PRED} models ***')
        plt.plot([], [], ' ', label=f'error metric={pred_error_metric[h]:.2f}')
        plt.plot([], [], ' ', label=f'lookback window={LOOKBACK_WINDOW} datapoints')

        plt.plot([], [], ' ', label=f'epochs={EPOCHS}, early stopping at < {LOSS_LIMIT} loss')
        plt.plot([], [], ' ', label=f'  stopped early after {loss_limit_epoch[h]} epochs')

        plt.plot([], [], ' ', label=f'optimizer={OPTIMIZER}, loss function={LOSS_FUNCTION}, learning rate={LEARNING_RATE}')

        if NOISE_TYPE == 'gaussian':
            plt.plot([], [], ' ', label=f'noise type={NOISE_TYPE}')
        else:
            plt.plot([], [], ' ', label=f'noise type={NOISE_TYPE}, degrees of freedom: nu={NOISE_STANDARD_T_DF:.1f}')


        plt.legend(fontsize=13)

        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)

        plt.grid()
        plt.tight_layout()
        # plt.show()
        FILENAME2a = FILENAME2 + f'{h}'.zfill(2) + 'a.png'
        plt.savefig(FILENAME2a)
        plt.close()


    # save some stats to a text file:
    stat_row = f'  model #{h}: error metric = {pred_error_metric[h]:.2f}'
    f2.write(stat_row + '\n')


# finish the stats file:
f2.write('---------------' + '\n')
f2.write(f'total_datapoints = {total_datapoints}' + '\n')
f2.write(f'LOOKBACK_WINDOW = {LOOKBACK_WINDOW}' + '\n')
f2.write(f'TRAINING_CUTOFF = {TRAINING_CUTOFF}' + '\n')
f2.write(f'VALIDATION_CUTOFF = {VALIDATION_CUTOFF}' + '\n')
f2.write(f'# of prediction steps (model output) = 1 datapoint' + '\n')
f2.write(f'prediction horizon = {PREDICTION_LENGTH}' + '\n')
f2.write('---------------' + '\n')
f2.write(f'OPTIMIZER = {OPTIMIZER}' + '\n')
f2.write(f'LOSS_FUNCTION = {LOSS_FUNCTION}' + '\n')
f2.write(f'LOSS_LIMIT = {LOSS_LIMIT}' + '\n')
f2.write(f'LEARNING_RATE = {LEARNING_RATE}' + '\n')
f2.write('---------------' + '\n')
f2.write(f'TUNE_ON = {TUNE_ON}' + '\n')
f2.write('---------------' + '\n')
f2.write(f'NOISE_ON = {NOISE_ON}' + '\n')
f2.write(f'NOISE_FACTOR = {NOISE_FACTOR}' + '\n')
f2.write(f'NOISE_STANDARD_T = {NOISE_STANDARD_T}' + '\n')
f2.write(f'NOISE_STANDARD_T_DF = {NOISE_STANDARD_T_DF}' + '\n')
f2.write('---------------' + '\n')
f2.write(f'EPOCHS (training) = {EPOCHS}' + '\n')
f2.write(f'  mean epochs = {np.mean(loss_limit_epoch):.1f}' + '\n')
f2.write(f'  max epochs = {np.max(loss_limit_epoch):.1f}' + '\n')
f2.write(f'  min epochs = {np.min(loss_limit_epoch):.1f}' + '\n')
f2.write('---------------' + '\n')
f2.write(f'Mean of the errors of all {N_PRED} models = {np.mean(pred_error_metric):.2f}' + '\n')
f2.write(f'Variance of the errors of all {N_PRED} models = {np.var(pred_error_metric):.1f}' + '\n')

f2.write('\n')
stat_model, tot_pars = count_parameters(model)
f2.write(stat_model + '\n')
f2.write(f'Total trainable parameters: {tot_pars}' + '\n')

f2.close()


print(f'\nMean of the errors of all {N_PRED} models = {np.mean(pred_error_metric):.2f}')
print(f'Variance of the errors of all {N_PRED} models = {np.var(pred_error_metric):.1f}')
print(f'\n=> best model is #{best_error_model} with an error of {best_error_metric:.2f}')

print(f'\nMean of epoch when the loss limit has been reached = {np.mean(loss_limit_epoch):.1f}')
print(f'Maximum epoch when the loss limit has been reached = {np.max(loss_limit_epoch):.1f}')
print(f'Minimum epoch when the loss limit has been reached = {np.min(loss_limit_epoch):.1f}')


breakpoint()

exit(0)


plt.suptitle('original curve:')
plt.title(NAME1)
plt.plot(t, y_clean)
plt.grid()
plt.tight_layout()
plt.show()
plt.close()

# end of Linear_deterministic_curve_forecasting.py
