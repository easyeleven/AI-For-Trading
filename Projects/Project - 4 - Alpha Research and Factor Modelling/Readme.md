# Project Description

In this project, you will build a statistical risk model using PCA. You’ll use this model to build a portfolio along with 5 alpha factors. You’ll create these factors, then evaluate them using factor-weighted returns, quantile analysis, sharpe ratio, and turnover analysis. At the end of the project, you’ll optimize the portfolio using the risk model and factors using multiple optimization formulations. For the dataset, we'll be using the end of day from Quotemedia and sector data from Sharadar.


1. Load large dollar volume stocks from quotemedia
##### Smart Beta by alternative weighting - dividend yield to choose the portfolio weight
2. Calculate Index Weights (dollar volume weights)
3. Calculate Portfolio Weights based on Dividend
4. Calculate Returns, Weighted Returns, Cumulative Returns
5. Tracking Error ` np.sqrt(252) * np.std(benchmark_returns_daily - etf_returns_daily, ddof=1)`
##### Portfolio Optimization - minimize the portfolio variance and closely track the index
$Minimize \left [ \sigma^2_p + \lambda \sqrt{\sum_{1}^{m}(weight_i - indexWeight_i)^2} \right  ]$ where $m$ is the number of stocks in the portfolio, and $\lambda$ is a scaling factor that you can choose.
6. Calculate the covariance of the returns `np.cov(returns.fillna(0).values, rowvar=False)`
7. Calculate optimal weights
   * Portfolio Variance: $\sigma^2_p = \mathbf{x^T} \mathbf{P} \mathbf{x}$
      * `cov_quad = cvx.quad_form(x, P)`
   * Distance from index weights: $\left \| \mathbf{x} - \mathbf{index} \right \|_2$ = $\sqrt{\sum_{1}^{n}(weight_i - indexWeight_i)^2}$
     * `index_diff = cvx.norm(x, p=2, axix=None)`
   * Objective function = $\mathbf{x^T} \mathbf{P} \mathbf{x} + \lambda \left \| \mathbf{x} - \mathbf{index} \right \|_2$
     * `cvx.Minimize(cov_quad + scale * index_diff)`
   * Constraints
     * ```
       x = cvx.Variable()
       constraints = [x >= 0, sum(x) == 1]
       ```
   * Optimization
     * ```
       problem = cvx.Problem(objective, constraints
       problem.solve()
       ```
8. Rebalance Portfolio over time
9. Portfolio Turnover
   * $ AnnualizedTurnover =\frac{SumTotalTurnover}{NumberOfRebalanceEvents} * NumberofRebalanceEventsPerYear $
   * $ SumTotalTurnover =\sum_{t,n}{\left | x_{t,n} - x_{t+1,n} \right |} $ Where $ x_{t,n} $ are the weights at time $ t $ for equity $ n $.
   * $ SumTotalTurnover $ is just a different way of writing $ \sum \left | x_{t_1,n} - x_{t_2,n} \right | $
 Minimum volatility ETF
