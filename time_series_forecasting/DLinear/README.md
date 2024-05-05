In preparation of the **DLinear** model ("Decomposition-Linear": https://github.com/cure-lab/LTSF-Linear/blob/main/models/DLinear.py (+)) I came across questions how to implement the technical details for the decomposition.

The original paper only states:

*Specifically, DLinear is a combination of a Decomposition scheme used in Autoformer and FEDformer with linear layers. It first decomposes a raw data input into a trend component by a <u>moving average kernel</u> and a remainder (seasonal) component. Then, two one-layer linear layers are applied to each component, and we sum up the two features to get the final prediction. By explicitly handling trend, DLinear enhances the performance of a vanilla linear when there is a clear trend in the data.*

Again from: https://arxiv.org/abs/2205.13504

So, for example, what is a "clear trend" and how to spot it for separation from other components of a time series?

Numerous methods have been established to exactly answer these questions.

<br/>

The original source code (+) apparently is using some kind of **simple moving average** (https://en.wikipedia.org/wiki/Moving_average#Simple_moving_average), however with some padding on both sides of the original time series.

In my version, I'm not doing this. I still use a simple moving average, without any precise reasoning for doing so, but only pad on the left hand side of the original time series, so that all (simple) moving averages are starting with the first data point, that is the first month of milk production:

```
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
```

<br/>

Then comes up the question for the kernel size, that is the size of the look-back window size. The original code has a fixed size of 25 data points, something which has been taken over from the Autoformer model:

*For DLinear, the moving average kernel size for decomposition is 25, which is the same as Autoformer.*

(I didn't look up why Autoformer is using exactly 25 as the size of the moving average kernel.)

So far I didn't find some reasoning why to prefer one moving average kernel size over the other.

In this paper..

&nbsp;&nbsp;Forecasting with moving averages

&nbsp;&nbsp;Robert Nau

&nbsp;&nbsp;Fuqua School of Business, Duke University

&nbsp;&nbsp;August 2014

https://people.duke.edu/~rnau/Notes_on_forecasting_with_moving_averages--Robert_Nau.pdf

..I got the idea to experiment with kernel sizes. That's why I have written this small utility for: https://github.com/PLC-Programmer/PyTorch/blob/main/time_series_forecasting/DLinear/DLinear_monthly_milk_production_moving_averages.py



