
Time Series Forecasting Project Report
1. Introduction
Time series forecasting plays a vital role in applications such as weather prediction, finance, and energy management. This project compares two advanced forecasting models: Facebook Prophet and an LSTM neural network with Monte Carlo Dropout. The objective is to generate accurate point forecasts with well-calibrated uncertainty intervals for a multivariate dataset.

2. Dataset Characteristics
Data Type: Multivariate time series data, including target and external regressors.

Temporal Resolution: 10-minute intervals spanning multiple years.

Preprocessing: Temporal train-test splits (Prophet: 80/20; LSTM: 70/15/15), feature scaling, and sequence generation for LSTM.

3. Model Architectures
Prophet Model
Uses daily, weekly, and yearly seasonalities.

External regressors added for improved fit.

Prediction intervals generated at approximately 80% confidence level.

LSTM with Monte Carlo Dropout
Input shape: sequence length = 24 timesteps, 25 features.

Two stacked LSTM layers (128 and 64 units) with dropout rate = 0.2.

Dense output layer for regression.

Monte Carlo Dropout used at inference to generate 95% confidence intervals from multiple stochastic forward passes.

4. Hyperparameter Tuning
Prophet model tuning focused on seasonality modes and regressor selection.

LSTM tuning focused on dropout rate, number of epochs (50), batch size (32), and number of Monte Carlo samples (100) for interval stability.

Validation data used to track overfitting and guide tuning.

5. Evaluation Metrics
Metric	Description
Mean Absolute Error (MAE)	Average absolute difference between forecasts and true values.
Root Mean Squared Error (RMSE)	Square root of average squared differences, penalizing large errors.
Prediction Interval Coverage Probability (PICP)	Percentage of true values falling within predicted intervals.
Mean Prediction Interval Width (MPIW)	Average width of prediction intervals, reflecting sharpness.
6. Results and Comparative Analysis
Model	MAE	RMSE	PICP (%)	MPIW
Prophet	0.5518	0.6455	2.07	0.0961
LSTM	0.1346	0.1869	98.15	1.4145
The LSTM outperforms Prophet on predictive accuracy (lower MAE and RMSE).

Prophet’s intervals are narrow but severely under-covering (poor calibration).

LSTM intervals are wider but provide a reliable 95% coverage.

Visual inspection of forecasts confirmed LSTM intervals effectively enclose true values, unlike Prophet.

7. Calibrated Prediction Intervals (Sample)
Timestamp	Point Forecast	Lower Bound	Upper Bound
2025-10-01 00:00:00	22.15	20.88	23.41
2025-10-01 00:10:00	21.95	20.64	23.18
...	...	...	...
(Full output saved in results/intervals.csv.)

8. Discussion
The LSTM with Monte Carlo Dropout is the better choice for operational forecasting where uncertainty matters.

Prophet, although simpler, fails to provide reliable intervals under these conditions.

This project shows the significance of uncertainty quantification for practical risk assessment.

Future work could explore deeper architectures, additional features, and other probabilistic modeling approaches.

9. Conclusion
This work demonstrates the effectiveness of LSTM neural networks with stochastic dropout for providing both accurate forecasts and trustworthy uncertainty estimates, substantially outperforming Prophet in this dataset and setting.
