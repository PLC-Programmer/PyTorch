2024-05-12: one thing is already clear to me after some experiments:

> **Warning**
Trend, even a constant, linear and "soft" trend, is a much harder problem for a (two-layer) linear model than noise!

<br/>

Good news! After the first couple of experiments I can say: the **direct multiple-step forecast** strategy is clearly the better choice than the recursive strategy (by factor of 2?) when the deterministic curve has a (total, soft) **trend** (tested without noise so far), cf. https://github.com/PLC-Programmer/PyTorch/tree/main/time_series_forecasting/experiments_with_linear_models#the-forecasting-strategy

That means that further development - for now - will go into the *Linear_deterministic_curve_forecasting_multiple_step.py* program only: https://github.com/PLC-Programmer/PyTorch/blob/main/time_series_forecasting/experiments_with_linear_models/direct_multiple-step_forecasting/Linear_deterministic_curve_forecasting_multiple_step.py

---

2024-05-11:
**Experiments with linear models**: https://github.com/PLC-Programmer/PyTorch/tree/main/time_series_forecasting/experiments_with_linear_models (so far without a **trend** in the time series, but experimenting with noise)

<br/>

---

I've been experimenting with a **decomposition** of the highly seasonal **monthly milk production** time series (https://github.com/PLC-Programmer/PyTorch/tree/main/time_series_forecasting/DLinear#autocorrelation-function-acf) and came to the conclusion that my two concepts I've working on did not surpass the prediction quality of the simple linear model: https://github.com/PLC-Programmer/PyTorch/tree/main/time_series_forecasting/Linear

I did a decomposition in the spirit of the original *DLinear.py* program (though I didn't copy this concept 1:1): https://github.com/cure-lab/LTSF-Linear/blob/main/models/DLinear.py

<br/>

After disappointing results I did a concept (https://github.com/PLC-Programmer/PyTorch/blob/main/time_series_forecasting/DLinear/backup/DLinear_monthly_milk_production_forecasting2a.py) where the trend (moving average) component is completely separated from the remainder (seasonal) component, and only their independent predictions are finally added together. Also this concept was not superior to the Linear concept!

One reason for these disappointing results might be the fact that the trend of this time series is 'not clear' at its right hand side:

![plot](./DLinear/backup/monthly_milk_production_forecasting2_00a.png)
