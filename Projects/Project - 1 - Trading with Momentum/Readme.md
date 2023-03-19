# Trading with Momentum

Trading with Momentum Project
In this project, you will learn to implement a trading strategy on your own, and test to see if it has the potential to be profitable. You will be supplied with a universe of stocks and time range. You will also be provided with a textual description of how to generate a trading signal based on a momentum indicator. You will then compute the signal for the time range given and apply it to the dataset to produce projected returns. Finally, you will perform a statistical test on the mean of the returns to conclude if there is alpha in the signal. For the dataset, we'll be using the end of day from Quotemedia.


0. import pandas, numpy, helper
1. Load Quatemedia EOD Price Data
2. Resample to Month-end `close_price.resample('M').last()`
3. Compute Log Return
4. Shift Returns `returns.shift(n)`
5. Generate Trading Signal
   * Strategy tried:
        > For each month-end observation period, rank the stocks by previous returns, from the highest to the lowest. Select the top performing stocks for the long portfolio, and the bottom performing stocks for the short portfolio.
   * ```
      for i, row in prev_price:
        top_stock.loc[i] = row.nlargest(top_n)
     ```
6. Projected Return `portfolio_returns = (lookahead_returns * (df_long - df_short))/n_stocks`
7. Statistical Test
   * Annualized Rate of Return `(np.exp(portfolio_returns.T.sum().dropna().mean()*12) - 1) * 100`
   * T-Test
     * Null hypothesis (H0): Actual mean return from the signal is zero.
     * When p value < 0.05, the null hypothesis is rejected
     * One-sample, one-sided t-test `(t_value, p_value) = scipy.stats.ttest_1samp(portfolio_return, hypothesis)`

## Important Links

1. [Panda's resampling example](https://towardsdatascience.com/using-the-pandas-resample-function-a231144194c4)
2. [loc and iloc playground by DataCamp](https://campus.datacamp.com/courses/intermediate-python/dictionaries-pandas?ex=17)
