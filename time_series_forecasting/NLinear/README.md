This is my **NLinear** version ("Normalization-Linear") of the basic Linear model from here: https://github.com/PLC-Programmer/PyTorch/tree/main/time_series_forecasting/Linear

> **It's not working well on the monthly Monthly Milk Production time series!**

 
Normalization (++), or at least the proposed normalization (+), doesn't help here:
* the same LOSS_LIMIT value of 1.0 can't be reached with a single model out of 10 for the same 20*LOOKBACK_WINDOW+1 epochs as with the original Linear model!
* the LOSS_LIMIT has been increased from 1.0 to 3.8 to bring down the mean epochs value roughly near the mean epochs value of the original Linear model
* with 10 models the mean of the prediction errors is at a similar value but the variance of the prediction errors is almost nonexistent with a (rounded) value of 0.0 (being for example 0.020351...)

> **=> the "visual variability" of a set of 10 models is neglectable!**

<br/>

(+) *Meanwhile, to boost the performance of LTSF-Linear when there is a distribution shift in the dataset, NLinear first subtracts the input by the last value of the sequence. Then, the input goes through a linear layer, and the subtracted part is added back before making the final prediction. The subtraction and addition in NLinear are a simple normalization for the input sequence.*

again from: https://arxiv.org/abs/2205.13504

<br/>

(++)
```
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
        return z
```
