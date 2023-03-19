# Sentiment Analysis on Financial Statements

n this project, you'll build your own deep learning model to classify the sentiment of messages from StockTwits, a social network for investors and traders. Your model will be able to predict if any particular message is positive or negative. From this, you'll be able to generate a signal of the public sentiment for various ticker symbols.


0. import json, nltk, os, random, re, torch, torch.nn, torch.optim, torch.nn.functional, numpy
1. Import Twits
    * json.load()
2. Preprocessing the Data
    * Pre-Processing
      * ```
        nltk.download('wordnet')
        nltk.download('stopwords')
        text = message.lower()
        text = re.sub('https?:\/\/[a-zA-Z0-9@:%._\/+~#=?&;-]*', ' ', text)
        text = re.sub('\$[a-zA-Z0-9]*', ' ', text)
        text = re.sub('\@[a-zA-Z0-9]*', ' ', text)
        text = re.sub('[^a-zA-Z]', ' ', text)
        tokens = text.split()
        wnl = nltk.stem.WordNetLemmatizer()
        tokens = [wnl.lemmatize(wnl.lemmatize(word, 'n'), 'v') for word in tokens]
    * Bag of Words
      * `bow = sorted(Counter(all_words), key=counts.get, reverse=True)`
    * Remove most common words such as 'the, 'and' by high_cutoff=20, rare words by low_cutoff=1e-6
    * Create Dictionaries
      * ```
        vocab = {word: ii for ii, word in enumarate(filtered_words, 1)}
        id2vodab = {v: k for k, v in vocab.items()}
        filtered = [[word for word in message if word in vocab] for message in tokenized]
        ```
    * Balancing the classes
      * 50% is neutral --> make it 20% by dropping some neutral twits
      * Remove messages with zero length
3. Neural Network
    * Embed -> RNN -> Dense -> Softmax
    * Text Classifier
      * ```
        class TextClassifier(nn.Module):
            def __init__(self, vocab_size, embed_size, lstm_size, output_size, lstm_layers=1, dropout=0.1):
                super().__init__()
                self.vocab_size = vocab_size
                self.embed_size = embed_size
                self.lstm_size = lstm_size
                self.output_size = output_size
                self.lstm_layers = lstm_layers
                self.dropout = dropout

                self.embedding = nn.Embedding(vodab_size, embed_size)
                self.lsfm = nn.LSTM(embed_size, lstm_size, lstm_layers, dropout=dropout, batch_first=False)
                self.dropout = nn.Dropout(-0.2)
                self.fc = nn.Linear(lstm_size, output_size)
                self.softmax = nn.LogSoftmax(dim=1)
            def init_hidden(self, batch_size):
                weight = next(self.parameters()).data
                hidden = (weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_(),
                          weight.new(self.lstm_layers, batch_size, self.lstm_size).zero_())
                return hidden
            def forward(self, nn_input, hidden_state)
                batch_size = nn_input.size(0)
                nn_input = nn_input.long()
                embeds = self.embedding(nn_input)
                lstm_out, hidden_state = self.lstm(embeds, hidden_state)
                lstm_out = lstm_out[-1,:,:] # Stack up LSMT Outputs
                out = self.dropout(lstm_out)
                out = self.fc(out)
                logps = self.softmax(out)
                return logps, hidden_state
        ```
4. Training
    * DataLoaders and Batching
      * Input Tensor shape should be (sequence_length, batch_size)
      * Left pad with zeros if a message has less tokens than sequence_length.
      * If a message has more token than sequence_length, keep the first sequence_length tokens
      * Build a DataLoader as a generator 
        ```
        def dataloader(): 
            yield batch, label_tensor # both variables are torch.tensor()
        ```
    * Training and Validation
      * Split data to training set and validation set, then check the model
        ```
        text_batch, labels = next(iter(dataloader(train_features, train_labels, sequence_length=20, batch_size=64)))
        model = TextClassifier(len(vocab)+1, 200, 128, 5, dropout=0.)
        hidden = model.init_hidden(64)
        logps, hidden = model.forward(text_batch, hidden)
        print(logps)
        ```
      * Model
        ```
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = TextClassifier(len(vocab)+1, 1024, 512, 5, lstm_layers=2, dropout=0.2)
        model.embedding.weight.data.uniform_(-1,1)
        model.to(device)
        ```
      * Train!
        ```
        epochs = 3
        batch_size = 1024
        learning_rate = 0.001
        clip = 5
        print_every = 100
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        model.train()
        for epoch in range(epochs):
            print ('Starting epoch {}'.format(epoch + 1))
            hidden = model.init_hidden(batch_size)
            steps = 0
            for text_batch, labels in dataloader(train_features, train_labels, batch_size=batch_size, sequence_length=20, shuffle=True):
                steps += 1
                if text_batch.size(1) != batch_size:
                    break
                hidden = tuple([each.data for each in hidden])
                text_batch, labels = text_batch.to(device), labels.to(device)
                for each in hidden:
                    each.to(device)
                model.zero_grad()
                output, hidden = model(text_batch, hidden)
                loss = criterion(output, labels)
                loss.backwards()
                nn.utils.clip_grad_norm_(model.parameters(), clip)
                optimizer.step() # Optimize
                if steps % print_every == 0:
                    model.eval()
                    valid_losses = []
                    accuracy = []
                    valid_hidden = model.init_hidden(batch_size)
                    for text_batch, labels in dataloader(valid_features, valid_labels, batch_size=batch_size, sequence_length=20, shuffle=False):
                        if text_batch.size(1) != batch_size:
                            break
                        valid_hidden = tuple([each.data for each in valid_hidden])
                        text_batch, lables = text_batch.to(device), labels.to(device)
                        for each in valid_hidden:
                            each.to(device)
                        valid_output, valid_hidden = model(text_batch, valid_hidden)
                        valid_loss = criterion(valid_output.squeeze(), labels)
                        valid_losses.append(valid_loss.item())
                        ps = torch.exp(valid_output)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy.append(torch.mean(equals.type(torch.FloatTensor)).item())
                    model.train()
                    print("Epoch: {}/{}...".format(epoch+1, epochs),
                          "Step: {}...".format(steps),
                          "Loss: {:.6f}...".format(loss.item()),
                          "Val Loss: {:.6f}".format(np.mean(valid_losses)),
                          "Accuracy: {:.6f}".format(np.mean(accuracy)))
        ```
5. Making Predictions
    * preprocess, filter non-vocab words, convert words to ids, add a batch dimention (`torch.tensor(tokens).view(-1,1))`
        ```
        hidden = model.init_hidden(1)
        logps, _ = model.forward(text_input, hidden)
        pred = torch.exp(logps)
        ```
6. Testing
