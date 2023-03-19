## COmbining Signals for Enhanced Alpha

0. import numpy, pandas, tqdm, matplotlib.pyplot
1. Data Pipeline
    * `zipline.data.bundles` - register, load
    * `zipline.pipeline.Pipeline`
    * `universe = zipline.pipeline.AverageDollarVolume`
    * `zipline.utils.calendar.get_calendar('NYSE')`
    * `zipline.pipeline.loaders.USEquityPricingLoader`
    * `engine = zipline.pipeline.engine.SimplePipelineEngine`
    * `zipline.data.data_portal.DataPortal`
2. Alpha Factors
    * Momentum 1 Year Factor
      * `zipline.pipeline.factors.Returns().demean(groupby=Sector).rank().zscore()`
    * Mean Reversion 5 Day Sector Neutral Smoothed Factor
      * `unsmoothed = -Returns().demean(groupby=Sector).rank().zscore()`
      * `smoothed = zipline.pipeline.factors.SimpleMovingAverage(unsmoothed).rank().zscore()`
    * Overnight Sentiment Smoothed Factor
      * CTO(Returns), TrainingOvernightReturns(Returns)
    * Combine the three factors by pipeline.add()
3. Features and Labels
    * Universal Quant Features
      * Stock Volatility 20d, 120d: `pipeline.add(zipline.pipeline.factors.AnnualizedVolatility)`
      * Stock Dollar Volume 20d, 120d: `pipeline.add(zipline.pipeline.factors.AverageDollarVolume)`
      * Sector
    * Regime Features
      * High and low volatility 20d, 120d: `MarketVolatility(CustomFactor)`
      * High and low dispersion 20d, 120d: `SimpleMovingAverage(MarketDispersion(CustomFactor))`
    * Target
      * 1 Week Return, Quantized: `pipeline.add(Returns().quantiles(2)), pipeline.add(Returns().quantiles(25))`
    * engine.run_pipeline()
    * Date Feature
      * January, December, Weekday, Quarter, Qtr-Year, Month End, Month Start, Qtr Start, Qtr End
    * One-hot encode Sector
    * Shift Target
    * IID Check (Independent and Identically Distributed)
      * Check rolling autocorelation between 1d to 5d shifted target using `scipy.stats.speamanr`
    * Train/Validation/Test Splits
4. Random Forests
    * Visualize a Simple Tree
      * clf = `sklearn.tree.DecisionTreeClassifier()`
      * Graph: `IPython.display.display`
      * Rank features by importance `clf.feature_importances_`
    * Random Forest
      * clf = `sklearn.ensemble.RandomForestClassifier()`
      * Scores: `clf.score(), clf.oob_score_, clf.feature_importances_`
    * Model Results
      * Sharpe Ratios `sqrt(252)*factor_returns.mean()/factor_returns.std()`
      * Factor Returns `alphalens.performance.factor_returns()`
      * Factor Rank Autocorelation `alphalens.performance.factor_rank_autocorrelation()`
      * Scores: `clf.predict_proba()`
    * Check the above for Training Data and Validation Data
5. Overlapping Samples
    * Option 1) Drop Overlapping Samples
    * Option 2) Use `sklearn.ensemble.BaggingClassifier`'s max_samples with `base_clf = DecisionTreeClassifier()`
    * Option 3) Build an ensemble of non-overlapping trees
      * ```
        sklearn.ensemble.VotingClassifier
        sklearn.base.clone
        sklearn.preprocessing.LavelEncoder
        sklearn.utils.Bunch
        ```
6. Final Model
    * Re-Training Model using Training Set + Validation Set
