# Project Description

In this project, you will build a statistical risk model using PCA. You’ll use this model to build a portfolio along with 5 alpha factors. You’ll create these factors, then evaluate them using factor-weighted returns, quantile analysis, sharpe ratio, and turnover analysis. At the end of the project, you’ll optimize the portfolio using the risk model and factors using multiple optimization formulations. For the dataset, we'll be using the end of day from Quotemedia and sector data from Sharadar.


0. import cvxpy, numpy, pandas, time, matplotlib.pyplot
1. Load equitiies EOD price (zipline.data.bundles)
    * `bundles.register(bundle_name, ingest_func)`
    * `bundles.load(bundle_name)`
2. Build Pipeline Engine
    * `universe = AverageDollarVolume(window_length=120).top(500)` <- 490 Tickers
    * `engine = SimplePipelineEngine(get_loader, calendar, asset_finder)`
3. Get Returns
    * `data_portal = DataPotal()`
    * `get_pricing = data_portal.get_history_window()`
    * `returns = get_pricing().pct_change()[1:].fillna(0)` <- e.g. 5 year: 1256x490
##### Statistical Risk Model
4. Fit PCA
    * `pca = sklearn.decomposition.PCA(n_components, svd_solver='full')`
    * `pca.fit()`
    * `pca.components_` <- 20x490
5. Factor Betas
    * `pd.DataFrame(pca.components_.T, index=returns.columns.values, columns=np.arange(20))` <- 20x490
6. Factor Returns
    * `pd.DataFrame(pca.transform(returns), index=returns.index , columns=np.arange(20))` <- 490x20
7. Factor Coveriance Matrix
    * `np.diag(np.var(factor_returns, axix=0, ddof=1)*252)` <- 20x20
8. Idiosyncratic Variance Matrix
    * ```
      _common_returns = pd.DataFrame(np.dot(factor_returns, factor_betas.T), returns.index, returns.columns)
      _residuals = (returns - _common_returns)
      pd.DataFrame(np.diag(np.var(_residuals)*252), returns.columns, returns.columns) <- 490x490
      ```
9. Idiosyncratic Variance Vector
    * `# np.dot(idiosyncratic_variance_matrix, np.ones(len(idiosyncratic_variance_matrix)))`
    * `pd.DaraFrame(np.diag(idiosyncratic_variance_matrix), returns.columns)`
10. Predict Portfolio Risk using the Risk Model
    * $ \sqrt{X^{T}(BFB^{T} + S)X} $ where:
      * $ X $ is the portfolio weights
      * $ B $ is the factor betas
      * $ F $ is the factor covariance matrix
      * $ S $ is the idiosyncratic variance matrix
    * `np.sqrt(weight_df.T.dot(factor_betas.dot(factor_cov_matrix).dot(factor_betas.T) + idiosyncratic_var_matrix).dot(weight_df))`
##### Create Alpha Factors
11. Momentum 1 Year Factor
12. Mean Reversion 5 Day Sector Neutral Factor
13. Mean Reversion 5 Day Sector Neutral Smoothed Factor
14. Overnight Sentiment Factor
15. Overnight Sentiment Smoothed Factor
16. Combine the Factors to a single Pipeline
    * ```
      pipeline = Pipeline(screen=universe)
      pipeline.add(momentum_1yr(252, universe, sector), 'Momentum_1YR')
      :
      all_factors = engine.run_pipeline(pipeline, start, end)
      ```
##### Evaluate Alpha Factors
17. Get Pricing Data
    * `assets = all_factors.index.level[1].values.tolist()`
18. Format Alpha Factors and Pricing for Alphalens
    * `clean_factor_data = {factor: alphalens.get_clean_factor_and_forward_returns(factor, prices, period=[1])}`
    * `unixt_factor_data = {factor: factor_data.set_index(pd.MultiIndex.from_tuples([(x.timestamp(), y) for x, y in factor_data.index.values], names=['date', 'asset']))}`
19. Quantile Analysis
    * Factor Returns: 
      * `alphalens.performance.factor_returns(factor_data).iloc[:, 0].cumprod().plot()`
      * This should be generally move up and to the right
    * Basis Points Per Day per Quantile
      * `alphalens.performance.mean_return_by_quantile(factor_data)[0].iloc[:, 0].plot.bar()`
      * Should be monotonic, not too much on short that is not practical to implement
      * Return spread (Q1 minus Q5)*252, considering transaction cost to cut this half, should be clear that these alphas can only survive in an institutional setting and that leverage will likely need to be applied
20. Turnover Analysis
    * Light test before full backtest to see the stability of the alphas over time
    * Factor Rank Autocorrelation (FRA) should be close to 1
    * `alphalens.performance.factor_rank_autocorrelation(factor_data).plot()`
21. Sharpe Ratio of the Alphas
    * `pd.Series(data=252*factor_returns.mean()/factor_returns.std())`
22. The Combined Alpha Vector
    * ML like Random Forest to get a single score per stock
    * Simpler approach is to jsut average
##### Optimal Portfolio Constrained by Risk Model
23. Objective and Constraints
    * Objective Function:
      * CVXPY objective function that maximizes $ \alpha^T * x \\ $, where $ x $ is the portfolio weights and $ \alpha $ is the alpha vector.
      * `cvx.Minimize(-alpha_vector.values.flatten()*weights)`
    * Constraints
      * $ r \leq risk_{\text{cap}}^2 \\ $ `risk <= self.risk_cap **2`
      * $ B^T * x \preceq factor_{\text{max}} \\ $ `factor_betas.T*weights <= self.factor_max`
      * $ B^T * x \succeq factor_{\text{min}} \\ $ `factor_betas.T*weight >= self.factor_min`
      * $ x^T\mathbb{1} = 0 \\ $ `sum(weights) == 0.0`
      * $ \|x\|_1 \leq 1 \\ $ `sum(cvs.abs(weights)) <= 1.0`
      * $ x \succeq weights_{\text{min}} \\ $ `weights >= self.weights_min`
      * $ x \preceq weights_{\text{max}} $ `weights <= self.weights_max`
      * Where $ x $ is the portfolio weights, $ B $ is the factor betas, and $ r $ is the portfolio risk
    * OptimalHoldings(ABC).find()
      * ```
        weights = cvx.Variable(len(alpha_vector))
        risk = cvx.quad_form(f, X) + cvx.quad_form(weights, S)
        prob = cvx.Problem(obj, constraints)
        prob.solve(max_iter=500)
        optimal_weights = np.asarray(weights.value).flatten()
        returns pd.DataFrame(data=optimal_weights, index=alpha_vector.index)
        ```
24. Optimize with a Regularization Parameter
    * To enforce diversification, change Objective Function
    * CVXPY objective function that maximize $ \alpha^T * x + \lambda\|x\|_2\\ $, where $ x $ is the portfolio weights, $ \alpha $ is the alpha vector, and $ \lambda $ is the regularization parameter.
    * `objective = cvx.Minimize(-alpha_vector.values.flatten()*weights + self.lambda_reg*cvx.norm(weights, 2))`
25. Optimize with a Strict Factor Constrains and Target Weighting 
    * Another common constraints is to take a predefined target weighting, $x^*$ (e.g., a quantile portfolio), and solve to get as close to that portfolio while respecting portfolio-level constraints. 
    * Minimize on on $ \|x - x^*\|_2 $, where $ x $ is the portfolio weights  $ x^* $ is the target weighting
    * `objective = cvs.Minimize(cvx.norm(alpha_vector.values.flatten()-weights), 2)`
