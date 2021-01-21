## How to use the Python programming Language for Time Series Analysis!

Click to see this work on  [My Kaggle Profile](https://www.kaggle.com/bulentsiyah) 

This work was prepared together with [Gul Bulut](https://www.kaggle.com/gulyvz) and [Bulent Siyah](https://www.kaggle.com/bulentsiyah/). **The whole study consists of two parties**

* [Time Series Forecasting and Analysis- Part 1](https://www.kaggle.com/gulyvz/time-series-forecasting-and-analysis-part-1)
* [Time Series Forecasting and Analysis- Part 2](https://www.kaggle.com/bulentsiyah/time-series-forecasting-and-analysis-part-2)

This kernel will teach you everything you need to know to use Python for forecasting time series data to predict new future data points.

![](https://iili.io/JaZxFS.png)

we'll learn about state of the art Deep Learning techniques with Recurrent Neural Networks that use deep learning to forecast future data points.

![](https://iili.io/JaZCMl.png)

This kernel even covers Facebook's Prophet library, a simple to use, yet powerful Python library developed to forecast into the future with time series data.

![](https://iili.io/JaZnP2.png)

# **Content Part 1**[](https://www.kaggle.com/bulentsiyah/time-series-forecasting-and-analysis-part-2/#Content-Part-1)

  
  
1. [How to Work with Time Series Data with Pandas](https://www.kaggle.com/gulyvz/time-series-forecasting-and-analysis-part-1#1.)
2. [Use Statsmodels to Analyze Time Series Data](https://www.kaggle.com/gulyvz/time-series-forecasting-and-analysis-part-1#2.)
3. [General Forecasting Models - ARIMA(Autoregressive Integrated Moving Average)](https://www.kaggle.com/gulyvz/time-series-forecasting-and-analysis-part-1#3.)
4. [General Forecasting Models - SARIMA(Seasonal Autoregressive Integrated Moving Average)](https://www.kaggle.com/gulyvz/time-series-forecasting-and-analysis-part-1#4.)
5. [General Forecasting Models - SARIMAX](https://www.kaggle.com/gulyvz/time-series-forecasting-and-analysis-part-1#5.)

# **Content Part 2**[](https://www.kaggle.com/bulentsiyah/time-series-forecasting-and-analysis-part-2/#Content-Part-2)

  
  
1. [Deep Learning for Time Series Forecasting - (RNN)](https://www.kaggle.com/bulentsiyah/time-series-forecasting-and-analysis-part-2/#1.)
2. [Multivariate Time Series with RNN](https://www.kaggle.com/bulentsiyah/time-series-forecasting-and-analysis-part-2/#2.)
3. [Use Facebook's Prophet Library for forecasting](https://www.kaggle.com/bulentsiyah/time-series-forecasting-and-analysis-part-2/#3.)

[](https://www.kaggle.com/bulentsiyah/time-series-forecasting-and-analysis-part-2/#1.Deep-Learning-for-Time-Series-Forecasting---(RNN))

In \[1\]:

    import pandas as pd import numpy as np %matplotlib inline import matplotlib.pyplot as plt 

In \[2\]:

    df = pd.read_csv('/kaggle/input/for-simple-exercises-time-series-forecasting/Alcohol_Sales.csv',index_col='DATE',parse_dates=True) df.index.freq = 'MS' df.head() 

Out\[2\]:

S4248SM144NCEN

DATE

1992-01-013459

1992-02-013458

1992-03-014002

1992-04-014564

1992-05-014221

In \[3\]:

    df.columns = ['Sales'] df.plot(figsize=(12,8)) 

Out\[3\]:

    <matplotlib.axes._subplots.AxesSubplot at 0x7f708742e048>

![](https://www.kaggleusercontent.com/kf/33406342/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..nbGzvutVCsrIxnly2nIyvg.DMkdh133Ww15ufatpKLuS_kFO6KdrUgMUxq9TRI-hMVtsPKFL5LlXPFLiYn5DEw3r7JSd-cbKfEWe66Sqj-2vTxupwZRtKWaVW6sofj2UPNNptW5lNO67_AkkrcCUbwx8YCq_3Gv-X_Y-KoUG08rxc2_wxYYs2mWihTrS9Yd7C8XHYeAbUWDK7xEKrc20--u8yBExSNCiA2vXegVhfeRPHfBfRaLgGn558gdLW5U4KeWM00mM_ukEth0Glwi3EmTDDiJ42a0yG6-aBeDJOBdX4pfGSCBDtpqlCGO5uYW2KG-l3WvhJw_p9olp7Awrnto7cld8re2_FFKLT7VzHmDvSewvkYoyrWX5f_f0KPi3ZtcLP6z-0QsqrubeWYFYDQsplMKi2PUyRR57nFVSuUfh9X01cBYbCE7FBFY8a7Sob4EErKr7rDg3aktdNNN7I_ekCHCQF7eN3OkqqGqve1_DfDsOGUOnT_hwvps9Bo7KLiNwrJy4ud6UbLyW-Icohmt1BHl2WRkJAfkfwNLGhPeVmQ0mN7BPPfVP6hTiBlnbaf7dgiwbbMy9o2050-ceADTJWmxKjkEbsi-7SB_zbpdc5XfXD9Bi0oH6Grb71cFxGOYzH41HM20phixjAERq6iNQCZ4PTclIeySSJscMANGCMHS2g5Z-sPz5YOxTN3sXC4.axN8ODMyWnNUZwNbkoANdQ/__results___files/__results___4_1.png)

In \[4\]:

    from statsmodels.tsa.seasonal import seasonal_decompose results = seasonal_decompose(df['Sales']) results.observed.plot(figsize=(12,2)) 

Out\[4\]:

    <matplotlib.axes._subplots.AxesSubplot at 0x7f70751227f0>

![](https://www.kaggleusercontent.com/kf/33406342/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..nbGzvutVCsrIxnly2nIyvg.DMkdh133Ww15ufatpKLuS_kFO6KdrUgMUxq9TRI-hMVtsPKFL5LlXPFLiYn5DEw3r7JSd-cbKfEWe66Sqj-2vTxupwZRtKWaVW6sofj2UPNNptW5lNO67_AkkrcCUbwx8YCq_3Gv-X_Y-KoUG08rxc2_wxYYs2mWihTrS9Yd7C8XHYeAbUWDK7xEKrc20--u8yBExSNCiA2vXegVhfeRPHfBfRaLgGn558gdLW5U4KeWM00mM_ukEth0Glwi3EmTDDiJ42a0yG6-aBeDJOBdX4pfGSCBDtpqlCGO5uYW2KG-l3WvhJw_p9olp7Awrnto7cld8re2_FFKLT7VzHmDvSewvkYoyrWX5f_f0KPi3ZtcLP6z-0QsqrubeWYFYDQsplMKi2PUyRR57nFVSuUfh9X01cBYbCE7FBFY8a7Sob4EErKr7rDg3aktdNNN7I_ekCHCQF7eN3OkqqGqve1_DfDsOGUOnT_hwvps9Bo7KLiNwrJy4ud6UbLyW-Icohmt1BHl2WRkJAfkfwNLGhPeVmQ0mN7BPPfVP6hTiBlnbaf7dgiwbbMy9o2050-ceADTJWmxKjkEbsi-7SB_zbpdc5XfXD9Bi0oH6Grb71cFxGOYzH41HM20phixjAERq6iNQCZ4PTclIeySSJscMANGCMHS2g5Z-sPz5YOxTN3sXC4.axN8ODMyWnNUZwNbkoANdQ/__results___files/__results___5_1.png)

In \[5\]:

    results.trend.plot(figsize=(12,2)) 

Out\[5\]:

    <matplotlib.axes._subplots.AxesSubplot at 0x7f7074e84550>

![](https://www.kaggleusercontent.com/kf/33406342/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..nbGzvutVCsrIxnly2nIyvg.DMkdh133Ww15ufatpKLuS_kFO6KdrUgMUxq9TRI-hMVtsPKFL5LlXPFLiYn5DEw3r7JSd-cbKfEWe66Sqj-2vTxupwZRtKWaVW6sofj2UPNNptW5lNO67_AkkrcCUbwx8YCq_3Gv-X_Y-KoUG08rxc2_wxYYs2mWihTrS9Yd7C8XHYeAbUWDK7xEKrc20--u8yBExSNCiA2vXegVhfeRPHfBfRaLgGn558gdLW5U4KeWM00mM_ukEth0Glwi3EmTDDiJ42a0yG6-aBeDJOBdX4pfGSCBDtpqlCGO5uYW2KG-l3WvhJw_p9olp7Awrnto7cld8re2_FFKLT7VzHmDvSewvkYoyrWX5f_f0KPi3ZtcLP6z-0QsqrubeWYFYDQsplMKi2PUyRR57nFVSuUfh9X01cBYbCE7FBFY8a7Sob4EErKr7rDg3aktdNNN7I_ekCHCQF7eN3OkqqGqve1_DfDsOGUOnT_hwvps9Bo7KLiNwrJy4ud6UbLyW-Icohmt1BHl2WRkJAfkfwNLGhPeVmQ0mN7BPPfVP6hTiBlnbaf7dgiwbbMy9o2050-ceADTJWmxKjkEbsi-7SB_zbpdc5XfXD9Bi0oH6Grb71cFxGOYzH41HM20phixjAERq6iNQCZ4PTclIeySSJscMANGCMHS2g5Z-sPz5YOxTN3sXC4.axN8ODMyWnNUZwNbkoANdQ/__results___files/__results___6_1.png)

In \[6\]:

    results.seasonal.plot(figsize=(12,2)) 

Out\[6\]:

    <matplotlib.axes._subplots.AxesSubplot at 0x7f7074da3f28>

![](https://www.kaggleusercontent.com/kf/33406342/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..nbGzvutVCsrIxnly2nIyvg.DMkdh133Ww15ufatpKLuS_kFO6KdrUgMUxq9TRI-hMVtsPKFL5LlXPFLiYn5DEw3r7JSd-cbKfEWe66Sqj-2vTxupwZRtKWaVW6sofj2UPNNptW5lNO67_AkkrcCUbwx8YCq_3Gv-X_Y-KoUG08rxc2_wxYYs2mWihTrS9Yd7C8XHYeAbUWDK7xEKrc20--u8yBExSNCiA2vXegVhfeRPHfBfRaLgGn558gdLW5U4KeWM00mM_ukEth0Glwi3EmTDDiJ42a0yG6-aBeDJOBdX4pfGSCBDtpqlCGO5uYW2KG-l3WvhJw_p9olp7Awrnto7cld8re2_FFKLT7VzHmDvSewvkYoyrWX5f_f0KPi3ZtcLP6z-0QsqrubeWYFYDQsplMKi2PUyRR57nFVSuUfh9X01cBYbCE7FBFY8a7Sob4EErKr7rDg3aktdNNN7I_ekCHCQF7eN3OkqqGqve1_DfDsOGUOnT_hwvps9Bo7KLiNwrJy4ud6UbLyW-Icohmt1BHl2WRkJAfkfwNLGhPeVmQ0mN7BPPfVP6hTiBlnbaf7dgiwbbMy9o2050-ceADTJWmxKjkEbsi-7SB_zbpdc5XfXD9Bi0oH6Grb71cFxGOYzH41HM20phixjAERq6iNQCZ4PTclIeySSJscMANGCMHS2g5Z-sPz5YOxTN3sXC4.axN8ODMyWnNUZwNbkoANdQ/__results___files/__results___7_1.png)

In \[7\]:

    results.resid.plot(figsize=(12,2)) 

Out\[7\]:

    <matplotlib.axes._subplots.AxesSubplot at 0x7f7074d3a390>

![](https://www.kaggleusercontent.com/kf/33406342/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..nbGzvutVCsrIxnly2nIyvg.DMkdh133Ww15ufatpKLuS_kFO6KdrUgMUxq9TRI-hMVtsPKFL5LlXPFLiYn5DEw3r7JSd-cbKfEWe66Sqj-2vTxupwZRtKWaVW6sofj2UPNNptW5lNO67_AkkrcCUbwx8YCq_3Gv-X_Y-KoUG08rxc2_wxYYs2mWihTrS9Yd7C8XHYeAbUWDK7xEKrc20--u8yBExSNCiA2vXegVhfeRPHfBfRaLgGn558gdLW5U4KeWM00mM_ukEth0Glwi3EmTDDiJ42a0yG6-aBeDJOBdX4pfGSCBDtpqlCGO5uYW2KG-l3WvhJw_p9olp7Awrnto7cld8re2_FFKLT7VzHmDvSewvkYoyrWX5f_f0KPi3ZtcLP6z-0QsqrubeWYFYDQsplMKi2PUyRR57nFVSuUfh9X01cBYbCE7FBFY8a7Sob4EErKr7rDg3aktdNNN7I_ekCHCQF7eN3OkqqGqve1_DfDsOGUOnT_hwvps9Bo7KLiNwrJy4ud6UbLyW-Icohmt1BHl2WRkJAfkfwNLGhPeVmQ0mN7BPPfVP6hTiBlnbaf7dgiwbbMy9o2050-ceADTJWmxKjkEbsi-7SB_zbpdc5XfXD9Bi0oH6Grb71cFxGOYzH41HM20phixjAERq6iNQCZ4PTclIeySSJscMANGCMHS2g5Z-sPz5YOxTN3sXC4.axN8ODMyWnNUZwNbkoANdQ/__results___files/__results___8_1.png)

## Train Test Split[](https://www.kaggle.com/bulentsiyah/time-series-forecasting-and-analysis-part-2/#Train-Test-Split)

In \[8\]:

    print("len(df)", len(df)) train = df.iloc[:313] test = df.iloc[313:] print("len(train)", len(train)) print("len(test)", len(test)) 

    len(df) 325 len(train) 313 len(test) 12 

## Scale Data[](https://www.kaggle.com/bulentsiyah/time-series-forecasting-and-analysis-part-2/#Scale-Data)

In \[9\]:

    from sklearn.preprocessing import MinMaxScaler scaler = MinMaxScaler() # IGNORE WARNING ITS JUST CONVERTING TO FLOATS # WE ONLY FIT TO TRAININ DATA, OTHERWISE WE ARE CHEATING ASSUMING INFO ABOUT TEST SET scaler.fit(train) 

Out\[9\]:

    MinMaxScaler(copy=True, feature_range=(0, 1))

In \[10\]:

    scaled_train = scaler.transform(train) scaled_test = scaler.transform(test) 

## Time Series Generator[](https://www.kaggle.com/bulentsiyah/time-series-forecasting-and-analysis-part-2/#Time-Series-Generator)

This class takes in a sequence of data-points gathered at equal intervals, along with time series parameters such as stride, length of history, etc., to produce batches for training/validation.

In \[11\]:

    from keras.preprocessing.sequence import TimeseriesGenerator scaled_train[0] 

    Using TensorFlow backend. 

Out\[11\]:

    array([0.03658432])

In \[12\]:

    # define generator n_input = 2 n_features = 1 generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1) print('len(scaled_train)',len(scaled_train)) print('len(generator)',len(generator)) # n_input = 2 

    len(scaled_train) 313 len(generator) 311 

In \[13\]:

    # What does the first batch look like? X,y = generator[0] print(f'Given the Array: \n{X.flatten()}') print(f'Predict this y: \n{y}') 

    Given the Array: [0.03658432 0.03649885] Predict this y: [[0.08299855]] 

In \[14\]:

    # Let's redefine to get 12 months back and then predict the next month out n_input = 12 generator = TimeseriesGenerator(scaled_train, scaled_train, length=n_input, batch_size=1) # What does the first batch look like? X,y = generator[0] print(f'Given the Array: \n{X.flatten()}') print(f'Predict this y: \n{y}') 

    Given the Array: [0.03658432 0.03649885 0.08299855 0.13103684 0.1017181 0.12804513 0.12266006 0.09453799 0.09359774 0.10496624 0.10334217 0.16283443] Predict this y: [[0.]] 

## Create the Model[](https://www.kaggle.com/bulentsiyah/time-series-forecasting-and-analysis-part-2/#Create-the-Model)

In \[15\]:

    from keras.models import Sequential from keras.layers import Dense from keras.layers import LSTM # define model model = Sequential() model.add(LSTM(100, activation='relu', input_shape=(n_input, n_features))) model.add(Dense(1)) model.compile(optimizer='adam', loss='mse') model.summary() 

    Model: "sequential_1" _________________________________________________________________ Layer (type) Output Shape Param # ================================================================= lstm_1 (LSTM) (None, 100) 40800 _________________________________________________________________ dense_1 (Dense) (None, 1) 101 ================================================================= Total params: 40,901 Trainable params: 40,901 Non-trainable params: 0 _________________________________________________________________ 

In \[16\]:

    # fit model model.fit_generator(generator,epochs=50) 

    Epoch 1/50 301/301 [==============================] - 2s 7ms/step - loss: 0.0170 Epoch 2/50 301/301 [==============================] - 2s 5ms/step - loss: 0.0085 Epoch 3/50 301/301 [==============================] - 2s 5ms/step - loss: 0.0088 Epoch 4/50 301/301 [==============================] - 2s 5ms/step - loss: 0.0072 Epoch 5/50 301/301 [==============================] - 2s 5ms/step - loss: 0.0061 Epoch 6/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0050 Epoch 7/50 301/301 [==============================] - 2s 5ms/step - loss: 0.0043 Epoch 8/50 301/301 [==============================] - 2s 5ms/step - loss: 0.0036 Epoch 9/50 301/301 [==============================] - 2s 5ms/step - loss: 0.0030 Epoch 10/50 301/301 [==============================] - 2s 5ms/step - loss: 0.0026 Epoch 11/50 301/301 [==============================] - 2s 5ms/step - loss: 0.0021 Epoch 12/50 301/301 [==============================] - 2s 5ms/step - loss: 0.0021 Epoch 13/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0019 Epoch 14/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0019 Epoch 15/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0020 Epoch 16/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0020 Epoch 17/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0016 Epoch 18/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0017 Epoch 19/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0014 Epoch 20/50 301/301 [==============================] - 2s 5ms/step - loss: 0.0016 Epoch 21/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0017 Epoch 22/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0016 Epoch 23/50 301/301 [==============================] - 2s 5ms/step - loss: 0.0016 Epoch 24/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0015 Epoch 25/50 301/301 [==============================] - 2s 5ms/step - loss: 0.0015 Epoch 26/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0015 Epoch 27/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0017 Epoch 28/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0015 Epoch 29/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0013 Epoch 30/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0015 Epoch 31/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0015 Epoch 32/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0014 Epoch 33/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0014 Epoch 34/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0013 Epoch 35/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0016 Epoch 36/50 301/301 [==============================] - 2s 5ms/step - loss: 0.0016 Epoch 37/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0014 Epoch 38/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0014 Epoch 39/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0014 Epoch 40/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0015 Epoch 41/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0013 Epoch 42/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0015 Epoch 43/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0013 Epoch 44/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0013 Epoch 45/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0012 Epoch 46/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0012 Epoch 47/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0012 Epoch 48/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0014 Epoch 49/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0013 Epoch 50/50 301/301 [==============================] - 2s 6ms/step - loss: 0.0013 

Out\[16\]:

    <keras.callbacks.callbacks.History at 0x7f706053e4a8>

In \[17\]:

    model.history.history.keys() loss_per_epoch = model.history.history['loss'] plt.plot(range(len(loss_per_epoch)),loss_per_epoch) 

Out\[17\]:

    [<matplotlib.lines.Line2D at 0x7f705863f7f0>]

![](https://www.kaggleusercontent.com/kf/33406342/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..nbGzvutVCsrIxnly2nIyvg.DMkdh133Ww15ufatpKLuS_kFO6KdrUgMUxq9TRI-hMVtsPKFL5LlXPFLiYn5DEw3r7JSd-cbKfEWe66Sqj-2vTxupwZRtKWaVW6sofj2UPNNptW5lNO67_AkkrcCUbwx8YCq_3Gv-X_Y-KoUG08rxc2_wxYYs2mWihTrS9Yd7C8XHYeAbUWDK7xEKrc20--u8yBExSNCiA2vXegVhfeRPHfBfRaLgGn558gdLW5U4KeWM00mM_ukEth0Glwi3EmTDDiJ42a0yG6-aBeDJOBdX4pfGSCBDtpqlCGO5uYW2KG-l3WvhJw_p9olp7Awrnto7cld8re2_FFKLT7VzHmDvSewvkYoyrWX5f_f0KPi3ZtcLP6z-0QsqrubeWYFYDQsplMKi2PUyRR57nFVSuUfh9X01cBYbCE7FBFY8a7Sob4EErKr7rDg3aktdNNN7I_ekCHCQF7eN3OkqqGqve1_DfDsOGUOnT_hwvps9Bo7KLiNwrJy4ud6UbLyW-Icohmt1BHl2WRkJAfkfwNLGhPeVmQ0mN7BPPfVP6hTiBlnbaf7dgiwbbMy9o2050-ceADTJWmxKjkEbsi-7SB_zbpdc5XfXD9Bi0oH6Grb71cFxGOYzH41HM20phixjAERq6iNQCZ4PTclIeySSJscMANGCMHS2g5Z-sPz5YOxTN3sXC4.axN8ODMyWnNUZwNbkoANdQ/__results___files/__results___22_1.png)

## Evaluate on Test Data[](https://www.kaggle.com/bulentsiyah/time-series-forecasting-and-analysis-part-2/#Evaluate-on-Test-Data)

In \[18\]:

    first_eval_batch = scaled_train[-12:] first_eval_batch 

Out\[18\]:

    array([[0.63432772], [0.80776135], [0.72313873], [0.89870929], [1. ], [0.71672793], [0.88648602], [0.75869732], [0.82742115], [0.87443371], [0.96025301], [0.5584238 ]])

In \[19\]:

    first_eval_batch = first_eval_batch.reshape((1, n_input, n_features)) model.predict(first_eval_batch) 

Out\[19\]:

    array([[0.6874868]], dtype=float32)

In \[20\]:

    scaled_test[0] 

Out\[20\]:

    array([0.63116506])

In \[21\]:

    test_predictions = [] first_eval_batch = scaled_train[-n_input:] current_batch = first_eval_batch.reshape((1, n_input, n_features)) for i in range(len(test)): # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array]) current_pred = model.predict(current_batch)[0] # store prediction test_predictions.append(current_pred) # update batch to now include prediction and drop first value current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1) test_predictions 

Out\[21\]:

    [array([0.6874868], dtype=float32), array([0.8057986], dtype=float32), array([0.7580665], dtype=float32), array([0.923397], dtype=float32), array([0.9962742], dtype=float32), array([0.75279814], dtype=float32), array([0.9047118], dtype=float32), array([0.77618504], dtype=float32), array([0.8549177], dtype=float32), array([0.8928125], dtype=float32), array([0.9670736], dtype=float32), array([0.5747522], dtype=float32)]

In \[22\]:

    scaled_test 

Out\[22\]:

    array([[0.63116506], [0.82502778], [0.75972305], [0.94939738], [0.98743482], [0.82135225], [0.95956919], [0.80049577], [0.93025045], [0.95247457], [1.0661595 ], [0.65706471]])

## Inverse Transformations and Compare[](https://www.kaggle.com/bulentsiyah/time-series-forecasting-and-analysis-part-2/#Inverse-Transformations-and-Compare)

In \[23\]:

    true_predictions = scaler.inverse_transform(test_predictions) true_predictions 

Out\[23\]:

    array([[11073.90839344], [12458.03770655], [11899.6196956 ], [13833.82155687], [14686.41155297], [11837.98544043], [13615.22314852], [12111.58873272], [13032.68223149], [13476.01332593], [14344.79427296], [ 9755.02612317]])

In \[24\]:

    test['Predictions'] = true_predictions test 

    /opt/conda/lib/python3.6/site-packages/ipykernel_launcher.py:1: SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame. Try using .loc[row_indexer,col_indexer] = value instead See the caveats in the documentation: [http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy](http://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy) """Entry point for launching an IPython kernel. 

Out\[24\]:

SalesPredictions

DATE

2018-02-011041511073.908393

2018-03-011268312458.037707

2018-04-011191911899.619696

2018-05-011413813833.821557

2018-06-011458314686.411553

2018-07-011264011837.985440

2018-08-011425713615.223149

2018-09-011239612111.588733

2018-10-011391413032.682231

2018-11-011417413476.013326

2018-12-011550414344.794273

2019-01-01107189755.026123

In \[25\]:

    test.plot(figsize=(12,8)) 

Out\[25\]:

    <matplotlib.axes._subplots.AxesSubplot at 0x7f70603e2390>

![](https://www.kaggleusercontent.com/kf/33406342/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..nbGzvutVCsrIxnly2nIyvg.DMkdh133Ww15ufatpKLuS_kFO6KdrUgMUxq9TRI-hMVtsPKFL5LlXPFLiYn5DEw3r7JSd-cbKfEWe66Sqj-2vTxupwZRtKWaVW6sofj2UPNNptW5lNO67_AkkrcCUbwx8YCq_3Gv-X_Y-KoUG08rxc2_wxYYs2mWihTrS9Yd7C8XHYeAbUWDK7xEKrc20--u8yBExSNCiA2vXegVhfeRPHfBfRaLgGn558gdLW5U4KeWM00mM_ukEth0Glwi3EmTDDiJ42a0yG6-aBeDJOBdX4pfGSCBDtpqlCGO5uYW2KG-l3WvhJw_p9olp7Awrnto7cld8re2_FFKLT7VzHmDvSewvkYoyrWX5f_f0KPi3ZtcLP6z-0QsqrubeWYFYDQsplMKi2PUyRR57nFVSuUfh9X01cBYbCE7FBFY8a7Sob4EErKr7rDg3aktdNNN7I_ekCHCQF7eN3OkqqGqve1_DfDsOGUOnT_hwvps9Bo7KLiNwrJy4ud6UbLyW-Icohmt1BHl2WRkJAfkfwNLGhPeVmQ0mN7BPPfVP6hTiBlnbaf7dgiwbbMy9o2050-ceADTJWmxKjkEbsi-7SB_zbpdc5XfXD9Bi0oH6Grb71cFxGOYzH41HM20phixjAERq6iNQCZ4PTclIeySSJscMANGCMHS2g5Z-sPz5YOxTN3sXC4.axN8ODMyWnNUZwNbkoANdQ/__results___files/__results___32_1.png)

## Saving and Loading Models[](https://www.kaggle.com/bulentsiyah/time-series-forecasting-and-analysis-part-2/#Saving-and-Loading-Models)

In \[26\]:

    model.save('my_rnn_model.h5') '''from keras.models import load_model new_model = load_model('my_rnn_model.h5')''' 

Out\[26\]:

    "from keras.models import load_model\nnew_model = load_model('my_rnn_model.h5')"

[](https://www.kaggle.com/bulentsiyah/time-series-forecasting-and-analysis-part-2/#2.Multivariate-Time-Series-with-RNN)

Experimental data used to create regression models of appliances energy use in a low energy building. Data Set Information: The data set is at 10 min for about 4.5 months. The house temperature and humidity conditions were monitored with a ZigBee wireless sensor network. Each wireless node transmitted the temperature and humidity conditions around 3.3 min. Then, the wireless data was averaged for 10 minutes periods. The energy data was logged every 10 minutes with m-bus energy meters. Weather from the nearest airport weather station (Chievres Airport, Belgium) was downloaded from a public data set from Reliable Prognosis (rp5.ru), and merged together with the experimental data sets using the date and time column. Two random variables have been included in the data set for testing the regression models and to filter out non predictive attributes (parameters).

## Data[](https://www.kaggle.com/bulentsiyah/time-series-forecasting-and-analysis-part-2/#Data)

Let's read in the data set:

In \[27\]:

    import pandas as pd import numpy as np %matplotlib inline import matplotlib.pyplot as plt df = pd.read_csv('../input/for-simple-exercises-time-series-forecasting/energydata_complete.csv',index_col='date', infer_datetime_format=True) df.head() 

Out\[27\]:

ApplianceslightsT1RH\_1T2RH\_2T3RH\_3T4RH\_4...T9RH\_9T\_outPress\_mm\_hgRH\_outWindspeedVisibilityTdewpointrv1rv2

date

2016-01-11 17:00:00603019.8947.59666719.244.79000019.7944.73000019.00000045.566667...17.03333345.536.600000733.592.07.00000063.0000005.313.27543313.275433

2016-01-11 17:10:00603019.8946.69333319.244.72250019.7944.79000019.00000045.992500...17.06666745.566.483333733.692.06.66666759.1666675.218.60619518.606195

2016-01-11 17:20:00503019.8946.30000019.244.62666719.7944.93333318.92666745.890000...17.00000045.506.366667733.792.06.33333355.3333335.128.64266828.642668

2016-01-11 17:30:00504019.8946.06666719.244.59000019.7945.00000018.89000045.723333...17.00000045.406.250000733.892.06.00000051.5000005.045.41038945.410389

2016-01-11 17:40:00604019.8946.33333319.244.53000019.7945.00000018.89000045.530000...17.00000045.406.133333733.992.05.66666747.6666674.910.08409710.084097

5 rows × 28 columns

In \[28\]:

    df.info() 

    <class 'pandas.core.frame.DataFrame'> Index: 19735 entries, 2016-01-11 17:00:00 to 2016-05-27 18:00:00 Data columns (total 28 columns): Appliances 19735 non-null int64 lights 19735 non-null int64 T1 19735 non-null float64 RH_1 19735 non-null float64 T2 19735 non-null float64 RH_2 19735 non-null float64 T3 19735 non-null float64 RH_3 19735 non-null float64 T4 19735 non-null float64 RH_4 19735 non-null float64 T5 19735 non-null float64 RH_5 19735 non-null float64 T6 19735 non-null float64 RH_6 19735 non-null float64 T7 19735 non-null float64 RH_7 19735 non-null float64 T8 19735 non-null float64 RH_8 19735 non-null float64 T9 19735 non-null float64 RH_9 19735 non-null float64 T_out 19735 non-null float64 Press_mm_hg 19735 non-null float64 RH_out 19735 non-null float64 Windspeed 19735 non-null float64 Visibility 19735 non-null float64 Tdewpoint 19735 non-null float64 rv1 19735 non-null float64 rv2 19735 non-null float64 dtypes: float64(26), int64(2) memory usage: 4.4+ MB 

In \[29\]:

    df['Windspeed'].plot(figsize=(12,8)) 

Out\[29\]:

    <matplotlib.axes._subplots.AxesSubplot at 0x7f70583a95f8>

![](https://www.kaggleusercontent.com/kf/33406342/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..nbGzvutVCsrIxnly2nIyvg.DMkdh133Ww15ufatpKLuS_kFO6KdrUgMUxq9TRI-hMVtsPKFL5LlXPFLiYn5DEw3r7JSd-cbKfEWe66Sqj-2vTxupwZRtKWaVW6sofj2UPNNptW5lNO67_AkkrcCUbwx8YCq_3Gv-X_Y-KoUG08rxc2_wxYYs2mWihTrS9Yd7C8XHYeAbUWDK7xEKrc20--u8yBExSNCiA2vXegVhfeRPHfBfRaLgGn558gdLW5U4KeWM00mM_ukEth0Glwi3EmTDDiJ42a0yG6-aBeDJOBdX4pfGSCBDtpqlCGO5uYW2KG-l3WvhJw_p9olp7Awrnto7cld8re2_FFKLT7VzHmDvSewvkYoyrWX5f_f0KPi3ZtcLP6z-0QsqrubeWYFYDQsplMKi2PUyRR57nFVSuUfh9X01cBYbCE7FBFY8a7Sob4EErKr7rDg3aktdNNN7I_ekCHCQF7eN3OkqqGqve1_DfDsOGUOnT_hwvps9Bo7KLiNwrJy4ud6UbLyW-Icohmt1BHl2WRkJAfkfwNLGhPeVmQ0mN7BPPfVP6hTiBlnbaf7dgiwbbMy9o2050-ceADTJWmxKjkEbsi-7SB_zbpdc5XfXD9Bi0oH6Grb71cFxGOYzH41HM20phixjAERq6iNQCZ4PTclIeySSJscMANGCMHS2g5Z-sPz5YOxTN3sXC4.axN8ODMyWnNUZwNbkoANdQ/__results___files/__results___40_1.png)

In \[30\]:

    df['Appliances'].plot(figsize=(12,8)) 

Out\[30\]:

    <matplotlib.axes._subplots.AxesSubplot at 0x7f705839f518>

![](https://www.kaggleusercontent.com/kf/33406342/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..nbGzvutVCsrIxnly2nIyvg.DMkdh133Ww15ufatpKLuS_kFO6KdrUgMUxq9TRI-hMVtsPKFL5LlXPFLiYn5DEw3r7JSd-cbKfEWe66Sqj-2vTxupwZRtKWaVW6sofj2UPNNptW5lNO67_AkkrcCUbwx8YCq_3Gv-X_Y-KoUG08rxc2_wxYYs2mWihTrS9Yd7C8XHYeAbUWDK7xEKrc20--u8yBExSNCiA2vXegVhfeRPHfBfRaLgGn558gdLW5U4KeWM00mM_ukEth0Glwi3EmTDDiJ42a0yG6-aBeDJOBdX4pfGSCBDtpqlCGO5uYW2KG-l3WvhJw_p9olp7Awrnto7cld8re2_FFKLT7VzHmDvSewvkYoyrWX5f_f0KPi3ZtcLP6z-0QsqrubeWYFYDQsplMKi2PUyRR57nFVSuUfh9X01cBYbCE7FBFY8a7Sob4EErKr7rDg3aktdNNN7I_ekCHCQF7eN3OkqqGqve1_DfDsOGUOnT_hwvps9Bo7KLiNwrJy4ud6UbLyW-Icohmt1BHl2WRkJAfkfwNLGhPeVmQ0mN7BPPfVP6hTiBlnbaf7dgiwbbMy9o2050-ceADTJWmxKjkEbsi-7SB_zbpdc5XfXD9Bi0oH6Grb71cFxGOYzH41HM20phixjAERq6iNQCZ4PTclIeySSJscMANGCMHS2g5Z-sPz5YOxTN3sXC4.axN8ODMyWnNUZwNbkoANdQ/__results___files/__results___41_1.png)

## Train Test Split[](https://www.kaggle.com/bulentsiyah/time-series-forecasting-and-analysis-part-2/#Train-Test-Split)

In \[31\]:

    df = df.loc['2016-05-01':] df = df.round(2) print('len(df)',len(df)) test_days = 2 test_ind = test_days*144 # 24*60/10 = 144 test_ind 

    len(df) 3853 

Out\[31\]:

    288

In \[32\]:

    train = df.iloc[:-test_ind] test = df.iloc[-test_ind:] 

## Scale Data[](https://www.kaggle.com/bulentsiyah/time-series-forecasting-and-analysis-part-2/#Scale-Data)

In \[33\]:

    from sklearn.preprocessing import MinMaxScaler scaler = MinMaxScaler() # IGNORE WARNING ITS JUST CONVERTING TO FLOATS # WE ONLY FIT TO TRAININ DATA, OTHERWISE WE ARE CHEATING ASSUMING INFO ABOUT TEST SET scaler.fit(train) 

Out\[33\]:

    MinMaxScaler(copy=True, feature_range=(0, 1))

In \[34\]:

    scaled_train = scaler.transform(train) scaled_test = scaler.transform(test) 

## Time Series Generator[](https://www.kaggle.com/bulentsiyah/time-series-forecasting-and-analysis-part-2/#Time-Series-Generator)

This class takes in a sequence of data-points gathered at equal intervals, along with time series parameters such as stride, length of history, etc., to produce batches for training/validation.

In \[35\]:

    from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator # define generator length = 144 # Length of the output sequences (in number of timesteps) batch_size = 1 #Number of timeseries samples in each batch generator = TimeseriesGenerator(scaled_train, scaled_train, length=length, batch_size=batch_size) 

In \[36\]:

    print('len(scaled_train)',len(scaled_train)) print('len(generator) ',len(generator)) X,y = generator[0] print(f'Given the Array: \n{X.flatten()}') print(f'Predict this y: \n{y}') 

    len(scaled_train) 3565 len(generator) 3421 Given the Array: [0.03896104 0. 0.13798978 ... 0.14319527 0.75185111 0.75185111] Predict this y: [[0.03896104 0. 0.30834753 0.29439421 0.16038492 0.49182278 0.0140056 0.36627907 0.24142857 0.24364791 0.12650602 0.36276002 0.12 0.28205572 0.06169297 0.15759185 0.34582624 0.39585974 0.09259259 0.39649608 0.18852459 0.96052632 0.59210526 0.1 0.58333333 0.13609467 0.4576746 0.4576746 ]] 

## Create the Model[](https://www.kaggle.com/bulentsiyah/time-series-forecasting-and-analysis-part-2/#Create-the-Model)

In \[37\]:

    from tensorflow.keras.models import Sequential from tensorflow.keras.layers import Dense,LSTM scaled_train.shape 

Out\[37\]:

    (3565, 28)

In \[38\]:

    # define model model = Sequential() # Simple RNN layer model.add(LSTM(100,input_shape=(length,scaled_train.shape[1]))) # Final Prediction (one neuron per feature) model.add(Dense(scaled_train.shape[1])) model.compile(optimizer='adam', loss='mse') model.summary() 

    Model: "sequential" _________________________________________________________________ Layer (type) Output Shape Param # ================================================================= lstm (LSTM) (None, 100) 51600 _________________________________________________________________ dense (Dense) (None, 28) 2828 ================================================================= Total params: 54,428 Trainable params: 54,428 Non-trainable params: 0 _________________________________________________________________ 

## EarlyStopping[](https://www.kaggle.com/bulentsiyah/time-series-forecasting-and-analysis-part-2/#EarlyStopping)

In \[39\]:

    from tensorflow.keras.callbacks import EarlyStopping early_stop = EarlyStopping(monitor='val_loss',patience=1) validation_generator = TimeseriesGenerator(scaled_test,scaled_test, length=length, batch_size=batch_size) model.fit_generator(generator,epochs=10, validation_data=validation_generator, callbacks=[early_stop]) 

    Train for 3421 steps, validate for 144 steps Epoch 1/10 3421/3421 [==============================] - 182s 53ms/step - loss: 0.0114 - val_loss: 0.0102 Epoch 2/10 3421/3421 [==============================] - 180s 52ms/step - loss: 0.0079 - val_loss: 0.0086 Epoch 3/10 3421/3421 [==============================] - 181s 53ms/step - loss: 0.0075 - val_loss: 0.0084 Epoch 4/10 2295/3421 [===================>..........] - ETA: 58s - loss: 0.0073

In \[40\]:

    model.history.history.keys() losses = pd.DataFrame(model.history.history) losses.plot() 

Out\[40\]:

    <matplotlib.axes._subplots.AxesSubplot at 0x7f701f602fd0>

![](https://www.kaggleusercontent.com/kf/33406342/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..nbGzvutVCsrIxnly2nIyvg.DMkdh133Ww15ufatpKLuS_kFO6KdrUgMUxq9TRI-hMVtsPKFL5LlXPFLiYn5DEw3r7JSd-cbKfEWe66Sqj-2vTxupwZRtKWaVW6sofj2UPNNptW5lNO67_AkkrcCUbwx8YCq_3Gv-X_Y-KoUG08rxc2_wxYYs2mWihTrS9Yd7C8XHYeAbUWDK7xEKrc20--u8yBExSNCiA2vXegVhfeRPHfBfRaLgGn558gdLW5U4KeWM00mM_ukEth0Glwi3EmTDDiJ42a0yG6-aBeDJOBdX4pfGSCBDtpqlCGO5uYW2KG-l3WvhJw_p9olp7Awrnto7cld8re2_FFKLT7VzHmDvSewvkYoyrWX5f_f0KPi3ZtcLP6z-0QsqrubeWYFYDQsplMKi2PUyRR57nFVSuUfh9X01cBYbCE7FBFY8a7Sob4EErKr7rDg3aktdNNN7I_ekCHCQF7eN3OkqqGqve1_DfDsOGUOnT_hwvps9Bo7KLiNwrJy4ud6UbLyW-Icohmt1BHl2WRkJAfkfwNLGhPeVmQ0mN7BPPfVP6hTiBlnbaf7dgiwbbMy9o2050-ceADTJWmxKjkEbsi-7SB_zbpdc5XfXD9Bi0oH6Grb71cFxGOYzH41HM20phixjAERq6iNQCZ4PTclIeySSJscMANGCMHS2g5Z-sPz5YOxTN3sXC4.axN8ODMyWnNUZwNbkoANdQ/__results___files/__results___56_1.png)

## Evaluate on Test Data[](https://www.kaggle.com/bulentsiyah/time-series-forecasting-and-analysis-part-2/#Evaluate-on-Test-Data)

In \[41\]:

    first_eval_batch = scaled_train[-length:] first_eval_batch 

Out\[41\]:

    array([[0.1038961 , 0. , 0.72231687, ..., 0.53550296, 0.15909546, 0.15909546], [0.11688312, 0. , 0.73424191, ..., 0.52662722, 0.40344207, 0.40344207], [0.11688312, 0. , 0.73424191, ..., 0.51775148, 0.20452271, 0.20452271], ..., [0.18181818, 0. , 0.70017036, ..., 0.50118343, 0.33340004, 0.33340004], [0.09090909, 0. , 0.70017036, ..., 0.51952663, 0.78747248, 0.78747248], [0.1038961 , 0. , 0.70017036, ..., 0.53846154, 0.77286372, 0.77286372]])

In \[42\]:

    first_eval_batch = first_eval_batch.reshape((1, length, scaled_train.shape[1])) model.predict(first_eval_batch) 

Out\[42\]:

    array([[ 0.10138211, 0.06747055, 0.7054 , 0.39806256, 0.54101586, 0.43319184, 0.4200446 , 0.4243666 , 0.7039989 , 0.40865916, 0.30799067, 0.36109492, 0.6687389 , -0.00205898, 0.6135764 , 0.42317435, 0.5408545 , 0.31506443, 0.49856254, 0.3375144 , 0.595571 , 0.53723574, 0.4301601 , 0.2183933 , 0.5878991 , 0.5141734 , 0.5099164 , 0.5046277 ]], dtype=float32)

In \[43\]:

    scaled_test[0] 

Out\[43\]:

    array([0.19480519, 0. , 0.70017036, 0.3920434 , 0.53007217, 0.41064526, 0.40616246, 0.41913319, 0.72714286, 0.4115245 , 0.30722892, 0.36445121, 0.66777778, 0. , 0.61119082, 0.39840637, 0.51618399, 0.32953105, 0.53703704, 0.34024896, 0.6057377 , 0.52631579, 0.41881579, 0.2 , 0.55283333, 0.53372781, 0.76305783, 0.76305783])

In \[44\]:

    n_features = scaled_train.shape[1] test_predictions = [] first_eval_batch = scaled_train[-length:] current_batch = first_eval_batch.reshape((1, length, n_features)) for i in range(len(test)): # get prediction 1 time stamp ahead ([0] is for grabbing just the number instead of [array]) current_pred = model.predict(current_batch)[0] # store prediction test_predictions.append(current_pred) # update batch to now include prediction and drop first value current_batch = np.append(current_batch[:,1:,:],[[current_pred]],axis=1) 

## Inverse Transformations and Compare[](https://www.kaggle.com/bulentsiyah/time-series-forecasting-and-analysis-part-2/#Inverse-Transformations-and-Compare)

In \[45\]:

    true_predictions = scaler.inverse_transform(test_predictions) true_predictions = pd.DataFrame(data=true_predictions,columns=test.columns) true_predictions 

Out\[45\]:

ApplianceslightsT1RH\_1T2RH\_2T3RH\_3T4RH\_4...T9RH\_9T\_outPress\_mm\_hgRH\_outWindspeedVisibilityTdewpointrv1rv2

098.0642282.02411724.53069838.02643024.13646835.02824125.09911836.79901624.12799237.726848...21.79223837.17068716.231932756.34897556.6921682.18393340.2739457.28953025.49052425.226246

196.7347423.03212724.56049638.27405724.12363435.22240425.04828236.75549324.05867037.757009...21.62252237.03192815.869359756.84979856.7758432.40795340.4719357.04008025.58606325.403467

2101.3171283.65487724.58193338.45278424.10727435.39524624.98594536.73257924.00212037.867709...21.46573836.95288515.582083757.27725756.8520142.60464540.1515366.77541825.65402225.435035

3106.2275944.05048924.59547238.62635724.09213535.53003924.92815936.74073523.95754638.006228...21.32441436.89626315.331191757.60079756.7831162.78454739.6247516.53359825.68174625.425436

4110.2745624.31411124.60993138.81520324.08394135.67649824.88018536.79405223.92657638.167812...21.19696736.87120415.117654757.86148956.6735632.95541339.0301696.32080525.69815725.427384

..................................................................

283-484.899577-4.31702629.40432028.52185540.0727374.64098224.97573849.08468729.04299835.858090...22.88950035.01526661.924760746.902323-55.8979162.31187867.965786-5.10675428.44246035.991550

284-484.899210-4.31705729.40432928.52184540.0727444.64098024.97575449.08473029.04300135.858127...22.88950235.01528261.924766746.902294-55.8980882.31189367.965593-5.10682328.44246635.991547

285-484.899210-4.31708429.40433628.52182340.0727564.64095624.97577149.08476629.04300935.858155...22.88950335.01529161.924771746.902270-55.8982152.31190667.965450-5.10688728.44247835.991550

286-484.899119-4.31711129.40434528.52180240.0727724.64094424.97579149.08479629.04301435.858183...22.88950435.01529761.924806746.902253-55.8983512.31191067.965364-5.10693828.44250135.991532

287-484.899164-4.31713429.40435128.52177440.0727874.64091424.97581049.08483429.04302035.858207...22.88950435.01529761.924824746.902227-55.8984682.31191267.965279-5.10698228.44250735.991526

288 rows × 28 columns

[](https://www.kaggle.com/bulentsiyah/time-series-forecasting-and-analysis-part-2/#3.Use-Facebook's-Prophet-Library-for-forecasting)

In \[46\]:

    import pandas as pd from fbprophet import Prophet 

## Load Data[](https://www.kaggle.com/bulentsiyah/time-series-forecasting-and-analysis-part-2/#Load-Data)

The input to Prophet is always a dataframe with two columns: ds and y. The ds (datestamp) column should be of a format expected by Pandas, ideally YYYY-MM-DD for a date or YYYY-MM-DD HH:MM:SS for a timestamp. The y column must be numeric, and represents the measurement we wish to forecast.

In \[47\]:

    df = pd.read_csv('../input/for-simple-exercises-time-series-forecasting/Miles_Traveled.csv') df.head() 

Out\[47\]:

DATETRFVOLUSM227NFWA

01970-01-0180173.0

11970-02-0177442.0

21970-03-0190223.0

31970-04-0189956.0

41970-05-0197972.0

In \[48\]:

    df.columns = ['ds','y'] df['ds'] = pd.to_datetime(df['ds']) df.info() 

    <class 'pandas.core.frame.DataFrame'> RangeIndex: 588 entries, 0 to 587 Data columns (total 2 columns): ds 588 non-null datetime64[ns] y 588 non-null float64 dtypes: datetime64[ns](1), float64(1) memory usage: 9.3 KB 

In \[49\]:

    pd.plotting.register_matplotlib_converters() try: df.plot(x='ds',y='y',figsize=(18,6)) except TypeError as e: figure_or_exception = str("TypeError: " + str(e)) else: figure_or_exception = df.set_index('ds').y.plot().get_figure() 

![](https://www.kaggleusercontent.com/kf/33406342/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..nbGzvutVCsrIxnly2nIyvg.DMkdh133Ww15ufatpKLuS_kFO6KdrUgMUxq9TRI-hMVtsPKFL5LlXPFLiYn5DEw3r7JSd-cbKfEWe66Sqj-2vTxupwZRtKWaVW6sofj2UPNNptW5lNO67_AkkrcCUbwx8YCq_3Gv-X_Y-KoUG08rxc2_wxYYs2mWihTrS9Yd7C8XHYeAbUWDK7xEKrc20--u8yBExSNCiA2vXegVhfeRPHfBfRaLgGn558gdLW5U4KeWM00mM_ukEth0Glwi3EmTDDiJ42a0yG6-aBeDJOBdX4pfGSCBDtpqlCGO5uYW2KG-l3WvhJw_p9olp7Awrnto7cld8re2_FFKLT7VzHmDvSewvkYoyrWX5f_f0KPi3ZtcLP6z-0QsqrubeWYFYDQsplMKi2PUyRR57nFVSuUfh9X01cBYbCE7FBFY8a7Sob4EErKr7rDg3aktdNNN7I_ekCHCQF7eN3OkqqGqve1_DfDsOGUOnT_hwvps9Bo7KLiNwrJy4ud6UbLyW-Icohmt1BHl2WRkJAfkfwNLGhPeVmQ0mN7BPPfVP6hTiBlnbaf7dgiwbbMy9o2050-ceADTJWmxKjkEbsi-7SB_zbpdc5XfXD9Bi0oH6Grb71cFxGOYzH41HM20phixjAERq6iNQCZ4PTclIeySSJscMANGCMHS2g5Z-sPz5YOxTN3sXC4.axN8ODMyWnNUZwNbkoANdQ/__results___files/__results___69_0.png)

In \[50\]:

    print('len(df)',len(df)) print('len(df) - 12 = ',len(df) - 12) 

    len(df) 588 len(df) - 12 = 576 

In \[51\]:

    train = df.iloc[:576] test = df.iloc[576:] 

## Create and Fit Model[](https://www.kaggle.com/bulentsiyah/time-series-forecasting-and-analysis-part-2/#Create-and-Fit-Model)

In \[52\]:

    # This is fitting on all the data (no train test split in this example) m = Prophet() m.fit(train) 

Out\[52\]:

    <fbprophet.forecaster.Prophet at 0x7f70730c06a0>

## Forecasting[](https://www.kaggle.com/bulentsiyah/time-series-forecasting-and-analysis-part-2/#Forecasting)

**NOTE: Prophet by default is for daily data. You need to pass a frequency for sub-daily or monthly data. Info: [https://facebook.github.io/prophet/docs/non-daily\_data.html](https://facebook.github.io/prophet/docs/non-daily_data.html)**

In \[53\]:

    future = m.make_future_dataframe(periods=12,freq='MS') forecast = m.predict(future) 

In \[54\]:

    forecast.tail() 

Out\[54\]:

dstrendyhat\_loweryhat\_uppertrend\_lowertrend\_upperadditive\_termsadditive\_terms\_loweradditive\_terms\_upperyearlyyearly\_loweryearly\_uppermultiplicative\_termsmultiplicative\_terms\_lowermultiplicative\_terms\_upperyhat

5832018-08-01263410.800604274535.269790285644.123755263342.030655263476.98157516448.01304916448.01304916448.01304916448.01304916448.01304916448.0130490.00.00.0279858.813654

5842018-09-01263552.915940256177.190765267761.256896263449.458400263643.024845-1670.418537-1670.418537-1670.418537-1670.418537-1670.418537-1670.4185370.00.00.0261882.497404

5852018-10-01263690.446911263051.398780274886.109947263541.695987263825.1358335305.5058735305.5058735305.5058735305.5058735305.5058735305.5058730.00.00.0268995.952784

5862018-11-01263832.562247249806.070474261028.788020263640.580642264000.846531-8208.986942-8208.986942-8208.986942-8208.986942-8208.986942-8208.9869420.00.00.0255623.575305

5872018-12-01263970.093217251087.538081262731.050771263724.773757264186.916157-6922.716937-6922.716937-6922.716937-6922.716937-6922.716937-6922.7169370.00.00.0257047.376280

In \[55\]:

    test.tail() 

Out\[55\]:

dsy

5832018-08-01286608.0

5842018-09-01260595.0

5852018-10-01282174.0

5862018-11-01258590.0

5872018-12-01268413.0

In \[56\]:

    forecast.columns 

Out\[56\]:

    Index(['ds', 'trend', 'yhat_lower', 'yhat_upper', 'trend_lower', 'trend_upper', 'additive_terms', 'additive_terms_lower', 'additive_terms_upper', 'yearly', 'yearly_lower', 'yearly_upper', 'multiplicative_terms', 'multiplicative_terms_lower', 'multiplicative_terms_upper', 'yhat'], dtype='object')

In \[57\]:

    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(12) 

Out\[57\]:

dsyhatyhat\_loweryhat\_upper

5762018-01-01243850.453937238143.777398249480.190740

5772018-02-01235480.588794229702.624041241029.888771

5782018-03-01262683.274392256372.318521268163.016848

5792018-04-01262886.236399257227.047581269018.659587

5802018-05-01272609.522601266952.781615278452.756472

5812018-06-01272862.615300267443.492047278588.217647

5822018-07-01279321.841101273416.105839284843.281259

5832018-08-01279858.813654274535.269790285644.123755

5842018-09-01261882.497404256177.190765267761.256896

5852018-10-01268995.952784263051.398780274886.109947

5862018-11-01255623.575305249806.070474261028.788020

5872018-12-01257047.376280251087.538081262731.050771

### Plotting Forecast[](https://www.kaggle.com/bulentsiyah/time-series-forecasting-and-analysis-part-2/#Plotting-Forecast)

We can use Prophet's own built in plotting tools

In \[58\]:

    m.plot(forecast); 

![](https://www.kaggleusercontent.com/kf/33406342/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..nbGzvutVCsrIxnly2nIyvg.DMkdh133Ww15ufatpKLuS_kFO6KdrUgMUxq9TRI-hMVtsPKFL5LlXPFLiYn5DEw3r7JSd-cbKfEWe66Sqj-2vTxupwZRtKWaVW6sofj2UPNNptW5lNO67_AkkrcCUbwx8YCq_3Gv-X_Y-KoUG08rxc2_wxYYs2mWihTrS9Yd7C8XHYeAbUWDK7xEKrc20--u8yBExSNCiA2vXegVhfeRPHfBfRaLgGn558gdLW5U4KeWM00mM_ukEth0Glwi3EmTDDiJ42a0yG6-aBeDJOBdX4pfGSCBDtpqlCGO5uYW2KG-l3WvhJw_p9olp7Awrnto7cld8re2_FFKLT7VzHmDvSewvkYoyrWX5f_f0KPi3ZtcLP6z-0QsqrubeWYFYDQsplMKi2PUyRR57nFVSuUfh9X01cBYbCE7FBFY8a7Sob4EErKr7rDg3aktdNNN7I_ekCHCQF7eN3OkqqGqve1_DfDsOGUOnT_hwvps9Bo7KLiNwrJy4ud6UbLyW-Icohmt1BHl2WRkJAfkfwNLGhPeVmQ0mN7BPPfVP6hTiBlnbaf7dgiwbbMy9o2050-ceADTJWmxKjkEbsi-7SB_zbpdc5XfXD9Bi0oH6Grb71cFxGOYzH41HM20phixjAERq6iNQCZ4PTclIeySSJscMANGCMHS2g5Z-sPz5YOxTN3sXC4.axN8ODMyWnNUZwNbkoANdQ/__results___files/__results___81_0.png)

In \[59\]:

    import matplotlib.pyplot as plt %matplotlib inline m.plot(forecast) plt.xlim(pd.to_datetime('2003-01-01'),pd.to_datetime('2007-01-01')) 

Out\[59\]:

    (731216.0, 732677.0)

![](https://www.kaggleusercontent.com/kf/33406342/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..nbGzvutVCsrIxnly2nIyvg.DMkdh133Ww15ufatpKLuS_kFO6KdrUgMUxq9TRI-hMVtsPKFL5LlXPFLiYn5DEw3r7JSd-cbKfEWe66Sqj-2vTxupwZRtKWaVW6sofj2UPNNptW5lNO67_AkkrcCUbwx8YCq_3Gv-X_Y-KoUG08rxc2_wxYYs2mWihTrS9Yd7C8XHYeAbUWDK7xEKrc20--u8yBExSNCiA2vXegVhfeRPHfBfRaLgGn558gdLW5U4KeWM00mM_ukEth0Glwi3EmTDDiJ42a0yG6-aBeDJOBdX4pfGSCBDtpqlCGO5uYW2KG-l3WvhJw_p9olp7Awrnto7cld8re2_FFKLT7VzHmDvSewvkYoyrWX5f_f0KPi3ZtcLP6z-0QsqrubeWYFYDQsplMKi2PUyRR57nFVSuUfh9X01cBYbCE7FBFY8a7Sob4EErKr7rDg3aktdNNN7I_ekCHCQF7eN3OkqqGqve1_DfDsOGUOnT_hwvps9Bo7KLiNwrJy4ud6UbLyW-Icohmt1BHl2WRkJAfkfwNLGhPeVmQ0mN7BPPfVP6hTiBlnbaf7dgiwbbMy9o2050-ceADTJWmxKjkEbsi-7SB_zbpdc5XfXD9Bi0oH6Grb71cFxGOYzH41HM20phixjAERq6iNQCZ4PTclIeySSJscMANGCMHS2g5Z-sPz5YOxTN3sXC4.axN8ODMyWnNUZwNbkoANdQ/__results___files/__results___82_1.png)

In \[60\]:

    m.plot_components(forecast); 

![](https://www.kaggleusercontent.com/kf/33406342/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..nbGzvutVCsrIxnly2nIyvg.DMkdh133Ww15ufatpKLuS_kFO6KdrUgMUxq9TRI-hMVtsPKFL5LlXPFLiYn5DEw3r7JSd-cbKfEWe66Sqj-2vTxupwZRtKWaVW6sofj2UPNNptW5lNO67_AkkrcCUbwx8YCq_3Gv-X_Y-KoUG08rxc2_wxYYs2mWihTrS9Yd7C8XHYeAbUWDK7xEKrc20--u8yBExSNCiA2vXegVhfeRPHfBfRaLgGn558gdLW5U4KeWM00mM_ukEth0Glwi3EmTDDiJ42a0yG6-aBeDJOBdX4pfGSCBDtpqlCGO5uYW2KG-l3WvhJw_p9olp7Awrnto7cld8re2_FFKLT7VzHmDvSewvkYoyrWX5f_f0KPi3ZtcLP6z-0QsqrubeWYFYDQsplMKi2PUyRR57nFVSuUfh9X01cBYbCE7FBFY8a7Sob4EErKr7rDg3aktdNNN7I_ekCHCQF7eN3OkqqGqve1_DfDsOGUOnT_hwvps9Bo7KLiNwrJy4ud6UbLyW-Icohmt1BHl2WRkJAfkfwNLGhPeVmQ0mN7BPPfVP6hTiBlnbaf7dgiwbbMy9o2050-ceADTJWmxKjkEbsi-7SB_zbpdc5XfXD9Bi0oH6Grb71cFxGOYzH41HM20phixjAERq6iNQCZ4PTclIeySSJscMANGCMHS2g5Z-sPz5YOxTN3sXC4.axN8ODMyWnNUZwNbkoANdQ/__results___files/__results___83_0.png)

In \[61\]:

    from statsmodels.tools.eval_measures import rmse predictions = forecast.iloc[-12:]['yhat'] predictions 

Out\[61\]:

    576 243850.453937 577 235480.588794 578 262683.274392 579 262886.236399 580 272609.522601 581 272862.615300 582 279321.841101 583 279858.813654 584 261882.497404 585 268995.952784 586 255623.575305 587 257047.376280 Name: yhat, dtype: float64

In \[62\]:

    test['y'] 

Out\[62\]:

    576 245695.0 577 226660.0 578 268480.0 579 272475.0 580 286164.0 581 280877.0 582 288145.0 583 286608.0 584 260595.0 585 282174.0 586 258590.0 587 268413.0 Name: y, dtype: float64

In \[63\]:

    rmse(predictions,test['y']) 

Out\[63\]:

    8618.783155559411

In \[64\]:

    test.mean() 

Out\[64\]:

    y 268739.666667 dtype: float64

## Prophet Diagnostics[](https://www.kaggle.com/bulentsiyah/time-series-forecasting-and-analysis-part-2/#Prophet-Diagnostics)

Prophet includes functionality for time series cross validation to measure forecast error using historical data. This is done by selecting cutoff points in the history, and for each of them fitting the model using data only up to that cutoff point. We can then compare the forecasted values to the actual values.

In \[65\]:

    from fbprophet.diagnostics import cross_validation,performance_metrics from fbprophet.plot import plot_cross_validation_metric len(df) len(df)/12 # Initial 5 years training period initial = 5 * 365 initial = str(initial) + ' days' # Fold every 5 years period = 5 * 365 period = str(period) + ' days' # Forecast 1 year into the future horizon = 365 horizon = str(horizon) + ' days' df_cv = cross_validation(m, initial=initial, period=period, horizon = horizon) df_cv.head() 

Out\[65\]:

dsyhatyhat\_loweryhat\_upperycutoff

01977-01-01108479.087306107041.710385109884.311419102445.01976-12-11

11977-02-01102996.111502101502.260980104430.450535102416.01976-12-11

21977-03-01118973.317944117486.531273120346.494171119960.01976-12-11

31977-04-01120612.923539119090.079896122015.351195121513.01976-12-11

41977-05-01127883.031663126371.269290129257.086719128884.01976-12-11

In \[66\]:

    df_cv.tail() 

Out\[66\]:

dsyhatyhat\_loweryhat\_upperycutoff

1032017-08-01273614.230765268044.449825279627.753972283184.02016-12-01

1042017-09-01255737.189562249987.360798261551.567277262673.02016-12-01

1052017-10-01262845.616157257365.064845268981.082903278937.02016-12-01

1062017-11-01249500.895087244208.004549255508.082651257712.02016-12-01

1072017-12-01250750.668713244667.910999256110.605457266535.02016-12-01

In \[67\]:

    performance_metrics(df_cv) 

Out\[67\]:

horizonmsermsemaemapemdapecoverage

052 days2.402227e+074901.2518924506.3843710.0276310.0235930.4

153 days2.150811e+074637.6834074238.6627320.0248630.0235930.4

254 days1.807689e+074251.6925353708.9432750.0199330.0222780.5

355 days2.298205e+074793.9601544236.2752440.0230420.0235930.4

457 days2.078937e+074559.5357843972.0872700.0213170.0222780.5

........................

94360 days1.814608e+074259.8215153750.3594830.0195960.0195650.5

95361 days1.726110e+074154.6475363473.0373390.0182120.0189570.5

96362 days3.173990e+075633.8175084404.3007290.0220340.0247930.4

97364 days2.986513e+075464.9000404229.8698600.0213780.0216290.5

98365 days5.443147e+077377.7683775621.7078030.0265240.0247930.4

99 rows × 7 columns

In \[68\]:

    plot_cross_validation_metric(df_cv, metric='rmse'); 

![](https://www.kaggleusercontent.com/kf/33406342/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..nbGzvutVCsrIxnly2nIyvg.DMkdh133Ww15ufatpKLuS_kFO6KdrUgMUxq9TRI-hMVtsPKFL5LlXPFLiYn5DEw3r7JSd-cbKfEWe66Sqj-2vTxupwZRtKWaVW6sofj2UPNNptW5lNO67_AkkrcCUbwx8YCq_3Gv-X_Y-KoUG08rxc2_wxYYs2mWihTrS9Yd7C8XHYeAbUWDK7xEKrc20--u8yBExSNCiA2vXegVhfeRPHfBfRaLgGn558gdLW5U4KeWM00mM_ukEth0Glwi3EmTDDiJ42a0yG6-aBeDJOBdX4pfGSCBDtpqlCGO5uYW2KG-l3WvhJw_p9olp7Awrnto7cld8re2_FFKLT7VzHmDvSewvkYoyrWX5f_f0KPi3ZtcLP6z-0QsqrubeWYFYDQsplMKi2PUyRR57nFVSuUfh9X01cBYbCE7FBFY8a7Sob4EErKr7rDg3aktdNNN7I_ekCHCQF7eN3OkqqGqve1_DfDsOGUOnT_hwvps9Bo7KLiNwrJy4ud6UbLyW-Icohmt1BHl2WRkJAfkfwNLGhPeVmQ0mN7BPPfVP6hTiBlnbaf7dgiwbbMy9o2050-ceADTJWmxKjkEbsi-7SB_zbpdc5XfXD9Bi0oH6Grb71cFxGOYzH41HM20phixjAERq6iNQCZ4PTclIeySSJscMANGCMHS2g5Z-sPz5YOxTN3sXC4.axN8ODMyWnNUZwNbkoANdQ/__results___files/__results___92_0.png)

In \[69\]:

    plot_cross_validation_metric(df_cv, metric='mape'); 

![](https://www.kaggleusercontent.com/kf/33406342/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..nbGzvutVCsrIxnly2nIyvg.DMkdh133Ww15ufatpKLuS_kFO6KdrUgMUxq9TRI-hMVtsPKFL5LlXPFLiYn5DEw3r7JSd-cbKfEWe66Sqj-2vTxupwZRtKWaVW6sofj2UPNNptW5lNO67_AkkrcCUbwx8YCq_3Gv-X_Y-KoUG08rxc2_wxYYs2mWihTrS9Yd7C8XHYeAbUWDK7xEKrc20--u8yBExSNCiA2vXegVhfeRPHfBfRaLgGn558gdLW5U4KeWM00mM_ukEth0Glwi3EmTDDiJ42a0yG6-aBeDJOBdX4pfGSCBDtpqlCGO5uYW2KG-l3WvhJw_p9olp7Awrnto7cld8re2_FFKLT7VzHmDvSewvkYoyrWX5f_f0KPi3ZtcLP6z-0QsqrubeWYFYDQsplMKi2PUyRR57nFVSuUfh9X01cBYbCE7FBFY8a7Sob4EErKr7rDg3aktdNNN7I_ekCHCQF7eN3OkqqGqve1_DfDsOGUOnT_hwvps9Bo7KLiNwrJy4ud6UbLyW-Icohmt1BHl2WRkJAfkfwNLGhPeVmQ0mN7BPPfVP6hTiBlnbaf7dgiwbbMy9o2050-ceADTJWmxKjkEbsi-7SB_zbpdc5XfXD9Bi0oH6Grb71cFxGOYzH41HM20phixjAERq6iNQCZ4PTclIeySSJscMANGCMHS2g5Z-sPz5YOxTN3sXC4.axN8ODMyWnNUZwNbkoANdQ/__results___files/__results___93_0.png)
