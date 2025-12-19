# Statistical Significance Testing

## Overview

This document presents comprehensive statistical tests performed to validate the significance of the Cross-Asset Alpha Engine's performance. All tests are conducted on out-of-sample data to ensure unbiased evaluation.

**IMPORTANT: All empirical analysis in this project is conducted at daily frequency using daily OHLCV bars from Polygon.io. No intraday, tick, or order-book data is used in the current experiment.**

## Test Methodology

### 1. Sharpe Ratio Significance Test

**Hypothesis**: H₀: SR = 0 vs H₁: SR ≠ 0

The Sharpe ratio significance is tested using a t-test:

$$t = \frac{SR}{\sqrt{n}}$$

where $SR$ is the annualized Sharpe ratio and $n$ is the number of observations.

**Interpretation**: A significant t-statistic (p < 0.05) indicates that the strategy generates risk-adjusted returns that are statistically different from zero.

### 2. Bootstrap Confidence Intervals

Bootstrap resampling (1,000 iterations) is used to generate non-parametric confidence intervals for key performance metrics:

- **Sharpe Ratio**: 95% confidence interval
- **Annualized Return**: 95% confidence interval

This method does not assume normality and provides robust estimates of parameter uncertainty.

### 3. Alpha Persistence Test

**Regression Model**: 
$$\text{Excess Return}_t = \alpha + \beta \cdot \text{Benchmark Return}_t + \epsilon_t$$

**Hypothesis**: H₀: α = 0 vs H₁: α ≠ 0

The intercept (alpha) is tested for statistical significance using a t-test. A significant alpha indicates persistent excess returns after controlling for benchmark exposure.

### 4. Information Ratio Significance Test

**Hypothesis**: H₀: IR = 0 vs H₁: IR ≠ 0

The Information Ratio is tested by examining whether the mean excess return is significantly different from zero:

$$t = \frac{\bar{R}_e}{\sigma_e / \sqrt{n}}$$

where $\bar{R}_e$ is the mean excess return and $\sigma_e$ is the tracking error.

### 5. Return Distribution Tests

#### Normality Test (Jarque-Bera)
Tests whether portfolio returns follow a normal distribution:

$$JB = \frac{n}{6} \left( S^2 + \frac{(K-3)^2}{4} \right)$$

where $S$ is skewness and $K$ is kurtosis.

#### Distribution Moments
- **Mean**: Expected daily return
- **Standard Deviation**: Return volatility
- **Skewness**: Asymmetry of return distribution
- **Kurtosis**: Tail heaviness (excess kurtosis = kurtosis - 3)

### 6. Regime-Specific Performance Analysis

Performance metrics are calculated separately for each market regime to assess strategy robustness across different market conditions:

- Annualized return by regime
- Volatility by regime
- Sharpe ratio by regime

## Test Results

Statistical test results are saved in the following files:

- `notebooks/statistical_tests_results.json`: Complete test results in JSON format
- `notebooks/statistical_tests_summary.csv`: Summary table of key tests

### Key Results Summary

Based on the latest analysis (128 observations, test period: 2025-05-29 to 2025-11-28):

#### 1. Sharpe Ratio Significance
- **Sharpe Ratio**: 0.5293
- **t-statistic**: 5.99
- **p-value**: < 0.000001
- **Significance**: ✅ Highly significant at both 5% and 1% levels
- **Interpretation**: The strategy generates risk-adjusted returns that are statistically significantly different from zero

#### 2. Bootstrap Confidence Intervals
- **Sharpe Ratio**: Mean = 0.526, 95% CI = [-2.14, 3.35]
- **Annualized Return**: Mean = 4.43%, 95% CI = [-6.73%, 16.78%]

#### 3. Alpha Persistence
- **Alpha (annualized)**: 4.19%
- **Beta**: -1.00 (market-neutral)
- **t-statistic**: 0.71
- **p-value**: 0.477
- **Significance**: Not significant at 5% level
- **Interpretation**: Alpha is not statistically significantly different from zero after controlling for benchmark exposure

#### 4. Information Ratio
- **Information Ratio**: -2.61
- **t-statistic**: -1.83
- **p-value**: 0.069
- **Significance**: Not significant at 5% level (marginally significant at 10%)
- **Interpretation**: The negative information ratio indicates underperformance vs benchmark, but not statistically significant

#### 5. Return Distribution
- **Normality Test (Jarque-Bera)**: p-value = 0.416
- **Distribution**: Returns are normally distributed (p > 0.05)
- **Skewness**: 0.27 (slight positive skew)
- **Excess Kurtosis**: -2.74 (lighter tails than normal distribution)

### Key Metrics

The statistical tests provide:

1. **Significance Levels**: p-values for all hypothesis tests
2. **Confidence Intervals**: Bootstrap-based 95% confidence intervals
3. **Test Statistics**: t-statistics for all parametric tests
4. **Interpretation**: Clear statements about statistical significance

## Interpretation Guidelines

### Significance Levels

- **p < 0.01**: Highly significant (1% level)
- **p < 0.05**: Significant (5% level)
- **p ≥ 0.05**: Not significant

### Confidence Intervals

A 95% confidence interval means that if the experiment were repeated many times, 95% of the intervals would contain the true parameter value.

### Practical Significance

Statistical significance does not necessarily imply practical significance. Consider:

- **Economic Magnitude**: Is the alpha large enough to cover transaction costs?
- **Consistency**: Does performance persist across different time periods?
- **Robustness**: Are results stable across different parameter settings?

## Limitations

1. **Sample Size**: Limited to ~129 trading days (6 months) of out-of-sample data
2. **Time Period**: Results specific to 2023-2025 market conditions
3. **Market Regime**: Performance may vary in different market environments
4. **Assumptions**: Some tests assume return independence (may not hold in financial data)

## Future Enhancements

For journal publication, consider:

1. **Extended Sample Period**: Multiple years of out-of-sample data
2. **Multiple Testing Correction**: Bonferroni or FDR adjustment for multiple hypotheses
3. **Robustness Checks**: Sensitivity analysis across parameters
4. **Comparative Analysis**: Statistical comparison with alternative strategies
5. **Monte Carlo Simulation**: Additional validation through simulation

## References

- Lo, A. W. (2002). The statistics of Sharpe ratios. *Financial Analysts Journal*, 58(4), 36-52.
- Jobson, J. D., & Korkie, B. M. (1981). Performance hypothesis testing with the Sharpe and Treynor measures. *The Journal of Finance*, 36(4), 889-908.
- Ledoit, O., & Wolf, M. (2008). Robust performance hypothesis testing with the Sharpe ratio. *Journal of Empirical Finance*, 15(5), 850-859.

---

*Last Updated: Generated automatically from statistical test results*

