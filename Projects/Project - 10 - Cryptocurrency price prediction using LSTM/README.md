## Cryptocurrency price prediction using LSTM
This project aims to predict the closing prices of cryptocurrencies using a Long Short-Term Memory (LSTM) neural network. The LSTM network is trained on historical price data of two cryptocurrencies, Ethereum (ETH) and Bitcoin (BTC), and then used to predict future price trends.

### Dataset
The dataset used in this project is the historical daily price data of ETH and BTC from September 8, 2018, to June 4, 2021. The data is fetched using the get_crypto_data function from the fastquant Python library. The dataset is split into a training set (95%) and a test set (5%) for training and validation purposes.

### Preprocessing
Before feeding the data to the LSTM network, the data is preprocessed in the following way:

The low, high, close, and open columns of ETH and BTC are concatenated to form the input features for the model.
The input features and the target variable (close price of ETH) are normalized using the MinMaxScaler function from the sklearn.preprocessing library.
The normalized data is converted into PyTorch tensors for training and testing the LSTM model.
Model
The LSTM model used in this project has two LSTM layers, each with 100 nodes. The input sequence length is 2, which means that the model takes the closing prices of the previous two days to predict the closing price of the third day. The model also has a fully connected (FC) layer to compute the final output.

### Training
The LSTM model is trained for 200 epochs using the Mean Absolute Error (MAE) loss function and the Adam optimizer with a learning rate of 0.001. The training and validation loss are plotted using Matplotlib during training.

### Results
The trained LSTM model is used to predict the closing prices of ETH for the test set. The predicted prices are then denormalized and compared to the actual prices. The results are visualized using Matplotlib.

### Conclusion
This project demonstrates the use of LSTM networks for cryptocurrency price prediction. The LSTM model achieved good results in predicting the closing prices of ETH. However, it is important to note that cryptocurrency prices are highly volatile, and predicting them accurately is a challenging task.
