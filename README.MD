# HackNITP Winter '21 AI Challenge 
https://dockship.io/challenges/5fe5d4ad78da6d7f387580b8/hacknitp-winter-'21-ai-challenge/overview)

## The aim of the challenge was to classify the given audio files into positive, neutral or negative sentiment
## LB RANK : 2


### DATASET LINK : https://drive.google.com/file/d/1_6T6xWZLY35rJbp7XUTltaHrb0-yZELd/view?usp=sharing

# APPROACH 1:

Our first approach consisted of converting the audio files into text and then to treat the given problem as a multilabel classification NLP problem. For this I used Google's ```speech_recognition``` library. However, due to the excessive noise in some of the files this conversion was not possible. 

# APPROACH 2:

Our next approach had us converting the given audio files into numerical values and then treat the given problem as a tabular multilabel classification problem. For this I used ```librosa```. Librosa is widely known for its use in sound processing. Using this we were able to convert all the audio files into their correspoinding numerical values successfully. These numerical values represented the amplitude at each instant of time corresponding to the sampling rate.

## PREPROCESSING

- On the new dataset that consisted of the numerical values we did some genreal feature engineering like **mean**, **std**, **skew**, **diff**,**min** and **max**.
- We tried out various AutoML libraries to get a rough idea of which models seem to be working good on the generated dataset and the scores for the following are :

  - MLJAR-SUPERVISED - 98.18182
  - PyCaret - 96.36364

- LightGBM performed the best on the new dataset as observed in the results of the AutoML libraries.

- For the penultimate submission we used ```OPTUNA```, which is one of the widely used libraries for hyper-parameter tuning and were able to get a score of **99.09091**.

- For the final submission we used the **filename** as well in the features. The filename column had some data leakage which was possibly due to the filenames not being encrypted. We used the filename column along with the generated dataset on the LightGBM Classifier along with Optuna to tune it and were able to achieve a score of **100**.


## Technologies Used :

- ```librosa```
- ```pycaret```
- ```mljar-supervised```
- ```lightgbm```
- ```optuna```
- ```wave```
- ```auto-sklearn```
