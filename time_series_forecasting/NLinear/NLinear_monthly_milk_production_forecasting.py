# NLinear_monthly_milk_production_forecasting.py
"""
Using a linear model for deep learning based
time series forecasting which uses a simple normalization
for the input sequence ("Normalization-Linear")
(in sample): forecast the value of one time step ahead
"""
#
# 2024-05-03/04
#
#
# sources:
#   2022: Are Transformers Effective for Time Series Forecasting?
#         https://arxiv.org/abs/2205.13504
#   PyTorch:
#     Linear (y = xA + b): https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
#     Neural Network Training with Modules: https://pytorch.org/docs/stable/notes/modules.html#neural-network-training-with-modules
#   The univariate time series is from here:
#   https://github.com/philipperemy/keras-tcn/blob/master/tasks/time_series_forecasting.py
#
#
# run on Ubuntu: $ python3 NLinear_monthly_milk_production_forecasting.py
#
#
# test on Ubuntu 22.04.4 LTS, Python 3.10.12, torch 2.1.0: only programmatically OK
#
#
# test results:
#   - normalization doesn't help here: the same LOSS_LIMIT value of 1.0 can't be reached with a single model (out of 10)
#     for the same 20*LOOKBACK_WINDOW+1 epochs as with the original "Linear" model!
#   - the LOSS_LIMIT has been increased from 1.0 to 3.8 to bring down the mean epochs value roughly near the mean epochs value of the original "Linear" model
#   - with 10 models the mean of the prediction errors is at a similar value but the variance of the prediction errors is almost nonexistent with a (rounded) value of 0.0
#     (being for example 0.020351...)
#   - the "visual variability" of a set of 10 models is neglectable!
#
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
TUNE_TRIALS = 10 # 10 is enough


TITLE = "Normalized one-layer linear model for time series forecasting"
NAME1 = "Monthly Milk Production (in pounds)"
FILENAME1    = 'monthly_milk_production_forecasting_'
FILENAME2    = 'monthly_milk_production_forecasting_stats.txt'
FILENAME3    = 'normalized_input_sequences.png'
 
milk = pd.read_csv('monthly-milk-production-pounds-p.csv', index_col=0, parse_dates=True)
# print("Head of milk:\n", milk.head())

# total time series length: 14 full years
TOTAL_MONTHS      = len(milk)
LOOKBACK_WINDOW   = 12 # this represents the full year 1975 --> indices 156..167
TRAINING_CUTOFF   = 11*12-1  # 85% ~11 years --> 0..131
VALIDATION_CUTOFF = TOTAL_MONTHS - LOOKBACK_WINDOW - 1  # --> 132..155 = 2 years

LEARNING_RATE     = 0.003  # 0.00276 was tuned with Adam and LogCosh
EPOCHS            = 20*LOOKBACK_WINDOW+1  # org 20*LOOKBACK_WINDOW+1
OPTIMIZER         = 'Adam'
# OPTIMIZER         = 'AdamW'
# LOSS_FUNCTION     = 'mse'  # Mean-Squared-Error -- org
# LOSS_LIMIT        = 1.0  # good for mse
LOSS_FUNCTION     = 'LogCosh'
LOSS_LIMIT        = 3.8  # for a mean epochs target of roughly 58 with 10 different models

N_PRED            = 10  # org 10 prediction curves; 50 is enough for manual parameter testing

# some constants:
DOUBLE  = T.as_tensor(2.0)
LOG_TWO = T.as_tensor(np.log(2.0))



##################################################################
#
# data preparations
#
milk = milk.values  # np array for simplicity

# for tuning:
def split_dataset3(data):
    '''
    split a univariate dataset into:
      [training period]] + [validation period]] + [forecasting period, in sample]] = [total data]
    '''
    forecast = data[-LOOKBACK_WINDOW:]
    forecast = np.array(forecast)

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

    return train_x, train_y, validation_x, validation_y, forecast


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
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    return train_x, train_y
##################################################################



##################################################################
#
# define the model
#
class NNet(T.nn.Module):
    '''
    a one-layer linear model which employs
    a simple normalization on the input sequence
    by first subtracting the last value from the input sequence
    and adding it back on the model output
    '''
    def __init__(self):
        super().__init__()
        self.lin = T.nn.Linear(LOOKBACK_WINDOW, 1)

    def forward(self, x):
        last_t = x[-1]
        x = x - last_t  # subtract last element from all input sequence
        z = self.lin(x)
        z = z + last_t  # add back last element to all output
        return z, x
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
def tune_model(trial):
    '''
    find a tuned learning rate
    https://medium.com/swlh/optuna-hyperparameter-optimization-in-pytorch-9ab5a5a39e77
    https://github.com/shaktiwadekar9/Optuna_PyTorch_Hyperparameter_Search/blob/main/optuna_CNN_LearningRateTune_minLoss.ipynb
    '''
    model = NNet()  # initialize the model
    model.train()

    # hp/hyperparameter tuning of the learning rate:
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

    optimizer = getattr(T.optim, OPTIMIZER)(model.parameters(), lr=lr)

    # get the dataset:
    x_train, y_train, x_val, y_val, forecast = split_dataset3(milk)

    # convert to tensors:
    x_train_tensor  = T.from_numpy(x_train).type(T.FloatTensor)
    y_train_tensor  = T.from_numpy(y_train).type(T.FloatTensor)
    x_val_tensor    = T.from_numpy(x_val).type(T.FloatTensor)
    y_val_tensor    = T.from_numpy(y_val).type(T.FloatTensor)

    for i in range(EPOCHS):
        # train the model:
        for j in range(len(x_train_tensor)):
            optimizer.zero_grad()

            input = x_train_tensor[j].reshape(-1)
            y_pred = model(input)
            loss = myCustomLoss(y_pred, y_train_tensor[j])
            loss.backward()
            optimizer.step()

        # validate the model:
        model.eval()

        with T.no_grad():
            val_loss_batch = 0.0

            for j in range(len(x_val_tensor)):
                input = x_val_tensor[j].reshape(-1)
                y_pred = model(input)
                val_loss_batch += myCustomLoss(y_pred, y_val_tensor[j])

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
        # 0.00276

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
# - building
# - training
# - prediction
# - calculating a forecast error metric
# - plotting and saving the model plot

x_train, y_train = split_dataset2a(milk)

# convert to tensors:
x_train_tensor  = T.from_numpy(x_train).type(T.FloatTensor)
y_train_tensor  = T.from_numpy(y_train).type(T.FloatTensor)


model = NNet()  # initialize the model


# criterion = T.nn.MSELoss()  # define loss criterion: mean squared error

optimizer = T.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# optimizer = T.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

pred_error_metric = np.empty([N_PRED])
best_error_metric = 1e20
best_error_model = 0
loss_limit_epoch = list(range(N_PRED))

# save some stats to a text file:
stat_header = f'model # and {LOSS_FUNCTION} value:'
print("\n" + stat_header)
f2 = open(FILENAME2, 'w')
f2.write(stat_header + '\n')


print()
for h in range(N_PRED):
    # x_print = x_train_tensor[0].reshape(-1)
    print(f'\nmodel #{h} training starts...')

    # ini's:
    loss_limit_epoch[h] = EPOCHS
    model.train()  # switch the module to training mode == default mode

    # train the model:
    for i in range(EPOCHS):

        for j in range(len(x_train_tensor)):
            input = x_train_tensor[j].reshape(-1)  # tensor
            # no reshape => RuntimeError: mat1 and mat2 shapes cannot be multiplied (12x1 and 12x1)

            y_pred, Ninput_seq = model(input)  # forward
            # print('training mode output for x_train_tensor[0]: {}'.format(model(x_print)))

            # loss  = criterion(y_pred, y_train_tensor[j])
            loss = myCustomLoss(y_pred, y_train_tensor[j])

            model.zero_grad()
            loss.backward()
            optimizer.step()

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
    x_test_tensor   = T.empty(2*LOOKBACK_WINDOW)  # past of length LOOKBACK_WINDOW + future (in sample) of length LOOKBACK_WINDOW
    x_test_ini = T.from_numpy(milk[-2*LOOKBACK_WINDOW:-LOOKBACK_WINDOW])
    x_test_tensor[0:LOOKBACK_WINDOW] = T.transpose(x_test_ini,0,1)

    pred_value = np.empty(LOOKBACK_WINDOW)

    for i in range(LOOKBACK_WINDOW,2*LOOKBACK_WINDOW):

        input = x_test_tensor[i-LOOKBACK_WINDOW:i]

        output, Ninput_seq = model(input)

        pred_value[i-LOOKBACK_WINDOW] = output.detach().numpy()[0]  # tensor to numpy scalar

        # add predicted value to the test tensor!
        # so, no update of test data with in-sample-data:
        x_test_tensor[i] = output

    # print("\nactual y values =", milk[-LOOKBACK_WINDOW:].reshape(-1))
    # print("\nforecasted y values =", pred_value)

    pred_value_total = np.empty(TOTAL_MONTHS)
    pred_value_total[:] = np.nan
    pred_value_total[-LOOKBACK_WINDOW:] = pred_value

    # calculating a forecast error metric:
    # err = mean_squared_error(milk[-LOOKBACK_WINDOW:],pred_value)
    err = myCustomLoss(T.tensor(milk[-LOOKBACK_WINDOW:]), T.tensor(pred_value))
    print(f'  error of prediction = {err:.1f}')
    pred_error_metric[h] = err
    if err < best_error_metric:
        best_error_metric = err
        best_error_model  = h


    # plotting this model:
    fig = plt.figure(figsize=(16, 9))

    plt.suptitle(TITLE, fontsize=18)

    plt.plot(milk, label=NAME1, color='black', linewidth=0.75)
    plt.plot(pred_value_total, color='orange', label='1 month prediction, re-using predicted values')

    # extra labels:
    plt.plot([], [], ' ', label=f'*** model #{h+1} out of {N_PRED} models ***')
    plt.plot([], [], ' ', label=f'error metric={pred_error_metric[h]:.1f}')
    plt.plot([], [], ' ', label=f'lookback window={LOOKBACK_WINDOW} months')

    plt.plot([], [], ' ', label=f'epochs={EPOCHS}, early stopping at < {LOSS_LIMIT} loss')
    plt.plot([], [], ' ', label=f'  stopped early after {loss_limit_epoch[h]} epochs')

    plt.plot([], [], ' ', label=f'optimizer={OPTIMIZER}, loss function={LOSS_FUNCTION}, learning rate={LEARNING_RATE}')
    plt.plot([], [], ' ', label=f'training period: {TOTAL_MONTHS-LOOKBACK_WINDOW} months on the LHS')
    plt.legend(fontsize=13)

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.grid()
    plt.tight_layout()
    # plt.show()
    FILENAME1a = FILENAME1 + f'{h}'.zfill(2) + '.png'
    plt.savefig(FILENAME1a)
    plt.close()

    # save some stats to a text file:
    stat_row = f'  model #{h}: error metric = {pred_error_metric[h]:.1f}'
    f2.write(stat_row + '\n')


# finish the stats file:
f2.write('---------------' + '\n')
f2.write(f'TOTAL_MONTHS = {TOTAL_MONTHS}' + '\n')
f2.write(f'LOOKBACK_WINDOW = {LOOKBACK_WINDOW}' + '\n')
f2.write(f'TRAINING_CUTOFF = {TOTAL_MONTHS-LOOKBACK_WINDOW-1}' + '\n')
f2.write('---------------' + '\n')
f2.write(f'OPTIMIZER = {OPTIMIZER}' + '\n')
f2.write(f'LOSS_FUNCTION = {LOSS_FUNCTION}' + '\n')
f2.write(f'LOSS_LIMIT = {LOSS_LIMIT}' + '\n')
f2.write(f'LEARNING_RATE = {LEARNING_RATE}' + '\n')
f2.write(f'TUNE_ON = {TUNE_ON}' + '\n')
f2.write('---------------' + '\n')
f2.write(f'EPOCHS (training) = {EPOCHS}' + '\n')
f2.write(f'  mean epochs = {np.mean(loss_limit_epoch):.1f}' + '\n')
f2.write(f'  max epochs = {np.max(loss_limit_epoch):.1f}' + '\n')
f2.write(f'  min epochs = {np.min(loss_limit_epoch):.1f}' + '\n')
f2.write('---------------' + '\n')
f2.write(f'Mean of the errors of all {N_PRED} models = {np.mean(pred_error_metric):.1f}' + '\n')
f2.write(f'Variance of the errors of all {N_PRED} models = {np.var(pred_error_metric):.1f}' + '\n')
f2.close()


print(f'\nMean of the errors of all {N_PRED} models = {np.mean(pred_error_metric):.1f}')
print(f'Variance of the errors of all {N_PRED} models = {np.var(pred_error_metric):.1f}')
print(f'\n=> best model is #{best_error_model} with an error of {best_error_metric:.1f}')

print(f'\nMean of epoch when the loss limit has been reached = {np.mean(loss_limit_epoch):.1f}')
print(f'Maximum epoch when the loss limit has been reached = {np.max(loss_limit_epoch):.1f}')
print(f'Minimum epoch when the loss limit has been reached = {np.min(loss_limit_epoch):.1f}')



##################################################################
#
# visualize some normalized input sequences of the NLinear model:
# - first tensor, middle tensor, last tensor of the simple training dataset
#
raw_input_first  = x_train_tensor[0].reshape(-1)  # full year 1962
raw_input_middle = x_train_tensor[len(x_train_tensor) // 2].reshape(-1)  # full year 1968

# making a tensor for full year 1974, which is not trained with function split_dataset2a(),
# but a year 1973/11 - 1974/11,
# though full year 1974 is used here for equalized demonstration purposes:
x_train_last_star = milk[-LOOKBACK_WINDOW-LOOKBACK_WINDOW:-LOOKBACK_WINDOW]
raw_input_last_star = T.from_numpy(x_train_last_star).type(T.FloatTensor).reshape(-1)

y_pred_first, Ninput_seq_first = model(raw_input_first)
y_pred_middle, Ninput_seq_middle = model(raw_input_middle)
y_pred_last_star, Ninput_seq_last_star = model(raw_input_last_star)


# plotting these normalized input sequences:
fig = plt.figure()

plt.suptitle(TITLE)
plt.title('normalized input sequences of selected production years')

plt.plot(Ninput_seq_first, label='1962', color='xkcd:powder blue')
plt.plot(Ninput_seq_middle, label='1968', color='xkcd:sky blue')
plt.plot(Ninput_seq_last_star, label='1974', color='xkcd:dusk blue')

plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(FILENAME3)
plt.close()


breakpoint()


# end of NLinear_monthly_milk_production_forecasting.py
