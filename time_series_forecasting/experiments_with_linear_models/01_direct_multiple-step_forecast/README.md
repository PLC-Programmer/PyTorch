## The forecasting strategy (part 2)

Here's a version of the program where all 100 datapoints of the prediction horizon are forecast by the model at a time (**"direct multiple-step forecast"**): https://github.com/PLC-Programmer/PyTorch/blob/main/time_series_forecasting/experiments_with_linear_models/direct_multiple-step_forecasting/Linear_deterministic_curve_forecasting_multiple_step.py

The prediction quality seems similar:

![plot](./direct_multiple-step_forecasting/outputs/Linear_deterministic_curve_forecasting--01.png)

![plot](./direct_multiple-step_forecasting/outputs/Linear_deterministic_curve_forecasting--01a.png)

<br/>

However, a direct multiple-step forecast strategy has its price because, albeit using the structurally same linear model, it employs significantly more parameters than the recursive strategy. Here it's 25,200 versus 15,201 parameters:

```
+--------------------+------------+
|      Modules       | Parameters |
+--------------------+------------+
| layers.lin1.weight |   15000    |
|  layers.lin1.bias  |    100     |
| layers.lin2.weight |   10000    |
|  layers.lin2.bias  |    100     |
+--------------------+------------+
Total trainable parameters: 25200
```

from: https://github.com/PLC-Programmer/PyTorch/blob/main/time_series_forecasting/experiments_with_linear_models/direct_multiple-step_forecasting/outputs/Linear_deterministic_curve_forecasting_stats.txt

```
+--------------------+------------+
|      Modules       | Parameters |
+--------------------+------------+
| layers.lin1.weight |   15000    |
|  layers.lin1.bias  |    100     |
| layers.lin2.weight |    100     |
|  layers.lin2.bias  |     1      |
+--------------------+------------+
Total trainable parameters: 15201
```

from: https://github.com/PLC-Programmer/PyTorch/blob/main/time_series_forecasting/experiments_with_linear_models/20percent_Gaussian_noise/Linear_deterministic_curve_forecasting_stats.txt

And this naturally leads to a higher training effort, be it with a noisey or without a noisy time series, for a comparable prediction quality.

<br/>

I haven't experimented with more advanced forecasting strategies like "Direct-recursive hybrid multi-step forecasting" or "Multiple output multi-step forecasting": https://machinelearningmastery.com/multi-step-time-series-forecasting/

<br/>

Here's some advice:

*Recursive forecasting is biased when the underlying model is nonlinear, but direct forecasting has higher variance because it uses fewer observations when estimating the model, especially for longer forecast horizons.*

from: https://www.semanticscholar.org/paper/Recursive-and-direct-multi-step-forecasting%3A-the-of-Taieb-Hyndman/432bd2365c8cfebd16577990404d3ff9d05d7e7d

(TBD)

##_end
