Sometimes it's easier to experiment with synthetic times series ("curves") first since it gives you more control over different setups.

I got this idea from here: https://github.com/tgchomia/ts/blob/main/Example.txt

So, this is the time series to experiment with, made of 700 datapoints (70 time units with equal 0.1 time steps):

![plot](./Linear_deterministic_curve_forecasting_org_curve.png)

This is also a "quite seasonal" (univariate) time series, like the monthly milk production (from the real world): https://github.com/PLC-Programmer/PyTorch/tree/main/time_series_forecasting/Linear, made of two harmonic oscillations.

<br/>

However, starting with a one-layer (linear) model led to rather disappointing results like these:

![plot](./one_layer_no_noise/Linear_deterministic_curve_forecasting--00.png)

![plot](./one_layer_no_noise/Linear_deterministic_curve_forecasting--01.png)

![plot](./one_layer_no_noise/Linear_deterministic_curve_forecasting--02.png)









