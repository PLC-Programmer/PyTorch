2024-05-13: the whole directory **experiments_with_linear_models** (https://github.com/PLC-Programmer/PyTorch/tree/main/time_series_forecasting/experiments_with_linear_models) first needs a cleanup in chronological order before I will new content

---

2024-05-12: one thing is already clear to me after some experiments:

> **Warning**
Trend, even a constant, linear and "soft" trend, is a much harder problem for a (two-layer) linear model than noise!

<br/>

Good news! After the first couple of experiments I can say: the **direct multiple-step forecast** strategy is clearly the better choice than the recursive strategy (by factor of 2?) when the deterministic curve has a (total, soft) **trend** (tested without noise so far), cf. https://github.com/PLC-Programmer/PyTorch/tree/main/time_series_forecasting/experiments_with_linear_models#the-forecasting-strategy

That means that further development - for now - will go into the *Linear_deterministic_curve_forecasting_multiple_step.py* program only: https://github.com/PLC-Programmer/PyTorch/blob/main/time_series_forecasting/experiments_with_linear_models/direct_multiple-step_forecasting/Linear_deterministic_curve_forecasting_multiple_step.py
