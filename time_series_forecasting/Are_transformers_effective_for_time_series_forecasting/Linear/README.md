This **simple linear model** for deep learning based time series forecasting (in sample) for forecasting the value of one time step ahead indeed works quite well; at least on the "quite seasonal" (univariate) time series in use here.

Idea is based on this paper from 2022:

*Are Transformers Effective for Time Series Forecasting?* https://arxiv.org/abs/2205.13504

Important variables (or symbols) in this paper:
* L = input look-back window size (here 12 months)
* T = number of future time steps (forecasting; here 1 month)

"TL" is then used in these sentences in chapter "B.2. Implementation Details":

*The total parameters of a vanilla linear model and a NLinear are TL. The total parameters of the DLinear are 2TL.*

In the **NLinear** model ("Normalization-Linear": https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py) you have these configurations:
```
self.Linear = nn.Linear(self.seq_len, self.pred_len)  # L, T
```

While in the more elaborate **DLinear** model ("Decomposition-Linear": https://github.com/cure-lab/LTSF-Linear/blob/main/models/DLinear.py) you have these configurations:
```
self.Linear_Seasonal = nn.Linear(self.seq_len,self.pred_len)  # L, T
self.Linear_Trend = nn.Linear(self.seq_len,self.pred_len)  # L, T
```
=> this makes *2TL total parameters* :wink:

<br/>

The csv file with the monthly milk production data can be downloaded from here: https://github.com/philipperemy/keras-tcn/blob/master/tasks/monthly-milk-production-pounds-p.csv

<br/>

"Linear" really means: **y = xA + b**, see at: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html

<br/>

At the core of this computer program is this class definition for a neural network model with a LOOKBACK_WINDOW (of the last 12 months of production values) and one output, that is the production value of the following month:

```
class Net(T.nn.Module):
    '''
    simple one-layer linear model
    '''
    def __init__(self):
        super().__init__()
        self.lin = T.nn.Linear(LOOKBACK_WINDOW, 1)  # L, T

    def forward(self, x):
        z = self.lin(x)
        return z
```

<br/>

Test results:
* AdamW optimization is not better than Adam optimization (https://pytorch.org/docs/stable/generated/torch.optim.Adam.html) with the LogCosh loss function: variance of errors of all 50 models is higher; mean is about the same

<br/>

Notes:
* all 12 forecasted monthly production values are only based on past training data (and its own forecasted values). So, for example forecasted data point #2 does not use actual data point #1 of the test data on the left hand side of the time series. Or in other words, the prediction horizon is indeed 12 months.
* I'm a fan of the **LogCosh** loss funtion (https://jiafulow.github.io/blog/2021/01/26/huber-and-logcosh-loss-functions/). As you can see I've also left in comments the original MSE (Mean Squared Error) code as a standard loss function. It's a pitty that this elegant loss function is not officially supported in PyTorch yet.
