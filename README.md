# vibration_noise_detection

This repository contains the implementation of a structure for detecting differences between two images or videos (detecting damages on car bodies before and after rent)

## Data overview

* There are twelve engines in the data set, six of which have the NG label and the other six have the OK label.
* The data of each engine includes the output of two accelerometer sensors that have calculated the engine's acceleration in seven different tests performed on the engine.

<br />

<img width="350" alt="image" src="https://github.com/MiladSoleymani/vibration_noise_detection/assets/78655282/8172723f-7395-4311-998a-0c6f628caaa3">

## Removing DC frequency (High pass Filter)

* To have a stable signal and remove the bad effect of DC frequency, we apply a Butterworth high pass filter on data (result can be seen in the plot)

* The order of the filter is twenty and the cut-off frequency is 10Hz

<img width="321" alt="image" src="https://github.com/MiladSoleymani/vibration_noise_detection/assets/78655282/4b153813-367f-4180-911b-7f91a593bc4e">

* Considering that the data obtained from the sensors includes seven engine test intervals, each can contain useful data individually.

* Also, we need to chunk the signal for use in the voting algorithm.

## Power spectral density (PSD)

* The power spectrum of a time series describes the distribution of power into frequency components composing that signal.

* We calculate PSD for each chunk separately.

<img width="432" alt="image" src="https://github.com/MiladSoleymani/vibration_noise_detection/assets/78655282/212d7ce4-91de-4197-8de2-82c3e5dac803">

## Grid-search and model selection

* To get the best model, Grid-search algorithm is used.

* For this purpose, the Grid-search algorithm is performed separately with suitable parameters for each model and with the number of three folds.

* After obtaining the best parameters, the cross-validation results of each model are saved.

* Note: All seven engine data parts are used for train and evaluation.

<img width="250" alt="image" src="https://github.com/MiladSoleymani/vibration_noise_detection/assets/78655282/edc5bdaa-115e-44b5-bc0f-c07d1f2e33f3">

## Results

* Four models SVM, XGBoost, k-nearest-neighbors, and decision tree were investigated.

* The results of each fold for all models with the best parameters obtained from Grid-search, as well as the mean and variance of the cross-validation results, can be seen in the table for each model.
  
* In the table, there are sections called voting-accuracy, where the results of the voting-algorithm are written in these sections. (The algorithm is explained in this case on the next slide)

<img width="328" alt="image" src="https://github.com/MiladSoleymani/vibration_noise_detection/assets/78655282/fd1ff7ef-8a8d-4cb2-b2d8-0e9b244f7bc0">

