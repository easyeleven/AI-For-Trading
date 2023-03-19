## Project Discription

1. import nltk, numpy, pandas, pickle, pprint, tqdm.tqdm, bs4.BeautifulSoup, re
    * `nltk.download('stopwords'), nltk.download('wordnet')`
2. Get 10-k documents
    * Limit number of request per second by @limits
    * `feed = BeautifulSoup(request.get.text).feed`
    * `entries = [entry.content.find('filing-href').getText(), ... for entry in feed.find_all('entry')]`
    * Download 10-k documents
    * Extract Documents
      * `doc_start_pattern = re.compile(r'<DOCUMENT>')`
      * `doc_start_position_list = [x.end() for x in doc_start_pattern.finditer(text)]`
    * Get Document Types
      * `doc_type_pattern = re.compile(r'<TYPE>[^\n]+')`
      * `doc_type = doc_type_pattern.findall(doc)[0][len("<TYPE>"):].lower()`
3. Process the Data
    * Clean up
      * `text.lower()`
      * `BeautifulSoup(text, 'html.parser').get_text()`
    * Lemmatize
      * `nltk.stem.WordNetLemmatizer, nltk.corpus.wordnet`
    * Remove Stopwords
      * `nltk.corpus.stopwords`
4. Analysis on 10ks
    * Loughran and McDonald sentiment word list
      * Negative, Positive, Uncertainty, Litigious, Constraining, Superfluous, Modal
    * Sentiment Bag of Words (Count for each ticker, sentiment)
      * ```
        sklearn.feature_extraction.text.CountVectorizer(analyzer='word', vocabulary=sentiment)
        X = vectorizer.fit_transform(docs)
        features = vectorizer.get_feature_names()
        ```
    * Jaccard Similarity
      * `sklearn.metrics.jaccard_similarity_score(u, v)`
      * Get the similarity between neighboring bag of words
    * TF-IDF
      * `sklearn.feature_extraction.text.TfidfVectorizer(analyzer='word', vocabulary=sentiments)`
    * Cosine Similarity
      * `sklearn.metrics.pairwise.cosine_similarity(u, v)`
      * Get the similarity between neighboring IFIDF vectors
5. Evaluate Alpha Factors
    * Use yearly pricing to match with 10K frequency of annual production
    * Turn the sentiment dictionary into a dataframe so that alphalens can read
    * Alphalens Format
      * `data = alphalens.utils.get_clean_factor_and_forward_return(df.stack(), pricing, quantiles=5, bins=None, period=[1])`
    * Alphalens Format with Unix Timestamp
      * `{factor: data.set_index(pd.MultiIndex.from_tuples([(x.timestamp(), y) for x, y in data.index.values], names=['date', 'asset'])) for factor, data in factor_data.items()}`
    * Factor Returns
      * `alphalens.performance.factor_returns(data)`
      * Should move up and to the right
    * Basis Points Per Day per Quantile
      * `alphalens.performance.mean_return_by_quantile(data)`
      * Should be monotonic in quantiles
    * Turnover Analysis
      * Factor Rank Autocorrelation (FRA) to measure the stability without full backtest
      * `alphalens.factor_rank_autocorrelation(data)`
    * Sharpe Ratio of the Alphas
      * Should be 1 or higher
      * `np.sqrt(252)*factor_returns.mean() / factor_returns.std()`
