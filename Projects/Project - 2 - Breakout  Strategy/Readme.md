# Breakout Strategy

0. import pandas, numpy, helper
1. Load Quatemedia EOD Price Data
2. The Alpha Research Process
    * What feature of markets or investor behaviour would lead to a persistent anomaly that my signal will try to use?
    * Example Hypothesis:
      * Stocks oscillate in a range without news or significant interest
      * Traders seek to sell at the top of the range and buy at the bottom
      * When stocks break out of the range,
        * the liquidity traders seek to cover the losses, which magnify the move out of the range
        * the move out of the range attract other investor interst due to herd behaviour which favor continuation of the trend 
    * Process:
      1. Observ & Research
      2. Form Hypothesis
      3. Validate Hypothesis, back to #1
      4. Code Expression
      5. Evaluate in-sample
      6. Evaluate out-of-sample
3. Compute Highs and Lows in a Window
    * e.g., Rolling max/min for the past 50 days
4. Compute Long and Short Signals
    * long = close > high, short = close < low, position = long - short
5. Filter Signals (5, 10, 20 day signal window)
    * Check if there was a signal in the past window_size of days
      `has_past_signal = bool(sum(clean_signals[signal_i:signal_i+window_size]))`
    * Use the current signal if there's no past signal, else 0/False
      `clean_signal.append(not has_past_signal and current_signal)`
    * Apply the above to short (signal[signal == -1].fillna(0.astype(int))) and long, add them up
6. Lookahead Close Price
    * How many days to short or long `close_price.shift(lookahead_days*-1)`
7. Lookahead Price Return
    * Log return between lookahead_price and close_price
8. Compute the Signal Return
    * signal * lookahead_returns
9. Test for Significance
    * Plot a histogram of the signal returns
10. Check Outliers in the histogram
11. Kolmogorov-Smirnov Test (KS-Test)
    * Check which stock is causing the outlying returns
    * Run KS-Test on a normal distribution against each stock's signal returns
    * `ks_value, p_value = scipy.stats.kstest(rvs=group['signal_return'].values, cdf='norm', args=(mean_all, std_all))`
12. Find outliers
    * Symbols that pass the null hypothesis with a p-value less than 0.05
    * Symbols that with a KS value above ks_threshod(0.8)
    * Remove them by `good_tickers = list(set(close.column) - outlier_tickers)`
