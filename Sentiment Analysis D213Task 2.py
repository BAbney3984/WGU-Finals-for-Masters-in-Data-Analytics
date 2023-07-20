#!/usr/bin/env python
# coding: utf-8

# # NLM2 TASK 2: SENTIMENT ANALYSIS USING NEURAL NETWORKS
# Advanced Data Analytics - D213 Task 2
# Brittany Abney
# 6/20/23

# # Part I:  Research Question
# 
# ***A.  Describe the purpose of this data analysis by doing the following:***
# 
# 1.  Can I determine a positive or negative sentiment on the Amazon, Yelp, imdB data reviews using neural networks?
# 
# 2.  The objective and goals are to clean the dataset so it can be used for neural network modeling to decipher positive and negative sentiments.
# 
# 3.  I will use Sequential ANN (Artificial Neural Network) using embedding for binary classification and stronger word associations.

# # Part II:  Data Preparation
# 
#  ***Please see the code and results that follow below for each.***
# 
# •   presence of unusual characters (e.g., emojis, non-English characters, etc.)
# 
# •   vocabulary size
# 
# •   proposed word embedding length
# 
# •   statistical justification for the chosen maximum sequence length

# In[1]:


#Import libraries needed to run all analysis.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import os
import datetime
import string
import re

import warnings
warnings.filterwarnings('ignore')

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype
from pandas import Series, DataFrame
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from collections import  Counter
from collections import defaultdict
from collections import OrderedDict

import seaborn as sns
from sklearn import model_selection
from sklearn import linear_model
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import preprocessing
from sklearn.preprocessing import LabelBinarizer
from sklearn import decomposition
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.layers import LSTM, Dropout, SpatialDropout1D, GlobalAveragePooling1D
from tensorflow.keras.layers import Embedding, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from keras.backend import clear_session
from keras import backend as K
from keras import layers
from keras import regularizers

from keras import losses 
from keras import metrics 
from keras import optimizers


import argparse
import os
from keras.callbacks import ModelCheckpoint

from pylab import rcParams

import regex
from nltk.text import Text
import string, re
from keras.models import load_model

import sys
get_ipython().system('{sys.executable} -m pip install textblob')
get_ipython().system('{sys.executable} -m textblob.download_corpora')


from sklearn.feature_extraction.text import CountVectorizer
plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')
from platform import python_version
print('python version used for this analysis is:', python_version())



# In[2]:


#Load my three dataset: Amazon, IMDB, Yelp
# amazon
amazon = open("amazon_cells_labelled.txt").read()

a_labels, a_texts = [], []
for i, line in enumerate(amazon.split('\n')):
    content = line.split('\t')
    if len(content) > 1:
        a_texts.append(content[0])
        a_labels.append(content[1])

df_a = pd.DataFrame()
df_a['sentiment'] = a_labels
df_a['review'] = a_texts
df_a['source'] = 'amazon'

# imdb
imdb = open('imdb_labelled.txt').read()

i_labels, i_texts = [], []
for i, line in enumerate(imdb.split('\n')):
    content = line.split('\t')
    if len(content) > 1:
        i_texts.append(content[0])
        i_labels.append(content[1])

df_i = pd.DataFrame()
df_i['sentiment'] = i_labels
df_i['review'] = i_texts
df_i['source'] = 'imdb'

# yelp
yelp = open('yelp_labelled.txt').read()

y_labels, y_texts = [], []
for i, line in enumerate(yelp.split('\n')):
    content = line.split('\t')
    if len(content) > 1:
        y_texts.append(content[0])
        y_labels.append(content[1])

df_y = pd.DataFrame()
df_y['sentiment'] = y_labels
df_y['review'] = y_texts
df_y['source'] = 'yelp'


# In[3]:


#Preview entire dataframes and confirm sizes

display(df_a.head())
display(df_a.shape)
display(df_i.head())
display(df_i.shape)
display(df_y.head())
display(df_y.shape)


# In[4]:


#Combine all three dataframes together and confirm the concanated size.
# (C1. Chen, 2020)

df = pd.concat([df_a, df_i, df_y], ignore_index=True)
df.sentiment = df.sentiment.astype(int)
df.info()


# In[5]:


#Add sentence composition features to identify total number of characters in the sentence, total numberof words, average word length, total punctuation, upper case words and total number of title case words.
# descriptive features
import string

df['chars'] = df.review.apply(len)
df['words'] = df.review.apply(lambda x: len(x.split()))
df['avg_wlen'] = df['chars'] / df['words']
df['puncs'] = df.review.apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
df['uppers'] = df.review.apply(lambda x: len([word for word in x.split() if word.isupper()]))
df['titles'] = df.review.apply(lambda x: len([word for word in x.split() if word.istitle()]))
df.head()


# In[6]:


#Review the features for each of these statistics.
# descriptive statistics for sentence features
display(df.groupby(['source', 'sentiment']).describe().loc[:,(slice(None),['mean', 'std'])].reset_index())
display(df.groupby(['source', 'sentiment']).describe().loc[:,(slice(None),['min', 'max'])].reset_index())


# ***Amazon Results:***
# We identify that the mean counts for positive Amazon reviews is 53.6 and negative is 56.8. The mean countfor positive words is 9.9, while negative is 10.6. This is very close to equal. 
# The average word length for positive and negative is 5.7.
# For punctuation themean courts for positive show 0.41 of the reviews are less than the negative .56. The maximum count for positive reviews with upper case is .41, while negative is five times that of positive.
# The title case shows the mean counts for positive 1.3 and negative at 1.2, almost equal.
# 
# ***IMDb Results:***
# The mean count for positive characters is 87.5 and that is higher than the negative at 77.1. The maximum count for positive reviews is 479, while the negative is only 321.
# The mean count for positive words is 15.1 Negative is 13.6
# The average word length for mean was identical for negative and positive at 5.8
# For punctuation the mean count for positive at 2.7 is nearly equal to negative at 2.5
# Upper case mean counts for positive is 1.8 and negative at 1.4.
# The mean counts for title case are 1.8 for positive and 1.4 for negative.
# 
# ***Yelp Results:***
# The mean counts for positive at 55.9 are lower than the negative at 60.8
# Total words mean counts are 10.3 for positive and 11.5 for negative.
# The average word length forpositive is 5.6 and negative at 5.4
# The mean count for punctuation is 1.9 for positive and negative at 2.0.
# The upper case mean count for positive is .30 and are lower than the negative at .50.
# For the title case the mean counts was equal at 1.3.
# 
# ***What did we learn?*** Negative Amazon reviews are more likley to use upper case words. Positive IMDb reviews typically have a high character count over a negative review. Negative Yelp reviews are more likley to have high character counts. IMDb reviews also use more words and characters as expected.
# 

# In[7]:


#View the data shape
df.shape


# In[8]:


#Review a bit more 
df.describe(include= object)


# In[9]:


#Check data for null values as these can impact a model from accuracy.
df.isnull().any()


# In[10]:


#check if there are duplicates

df.duplicated().sum()


# In[11]:


duplicate = df[df.duplicated(keep=False)]
duplicate


# In[12]:


#Remove duplicates with a drop
df = df.drop_duplicates()
df


# In[13]:


#Remove #'s from data
df['review'] = df['review'].str.replace('\d+', '')


# In[14]:


# Reducing noise that is presenting in the text data is a must for a fucntioning model.
# Cleaning the data will add spaces to prevent words from merging. 
# Convert words to lowercase. 
# Remove punctuation tokens.
# Remove non-alphabetic non numeric tokens.
# Remove stop words and unnecessary words.
# Split sentences into tokens separated by whitespace.

i = 0
df['clean_text'] = ''
for row in df.review:
    # add spaces to prevent word merging
    row = row.replace('.', '. ', row.count('.')).replace(',', ', ', row.count(','))
    # split into words
    tokens = word_tokenize(row)
    # convert to lower case
    tokens = [token.lower() for token in tokens]
    # remove punctuation
    table = str.maketrans('', '', string.punctuation)
    words = [token.translate(table) for token in tokens]
    # remove non-alphabetic or numeric tokens
    words = [word for word in words if word.isalnum()]
    # filter stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if not word in stop_words]
    #print(words)
    df['clean_text'][i] = ' '.join(words)
    i += 1
df.clean_text = df.source + ' ' + df.clean_text
df.head()


# In[15]:


#Collect length of review
df['ReviewLength'] = df.review.str.len()


# In[16]:


# Create a list of the reviews and initialize variables for the vocabulary and max review length
reviews = df['review'].tolist()
vocab=[]
max_review_len = 0

#Loop throw the the reviews, build the vocabulary and keep track of max review length
for review in reviews:
    review_len = len(review.split(" "))
    if review_len > max_review_len:
        max_review_len = review_len
        
    for word in review.split(" "):
        if not word in vocab:
            vocab.append(word)
            
print("Vocab:    ", len(vocab))
print("Longest:  ", max_review_len)
print("Emb Size: ", len(vocab)**0.25)

#(C3.Brownlee, 2021)

Above shows the vocabulary size is 7,913 distinct words.
The suggested embedding length is 9.4, I will use 12 to make sure sufficient compute is used.
The longest sentence uses 73 words, that will be target when padding smaller values.
# # B2.  Describe the goals of the tokenization process, including any code generated and packages that are used to normalize text during the tokenization process.
# 
# I will use keras built-in one-hot encoding to translate words into numeric representations.

# In[17]:


encoded_reviews = [keras.preprocessing.text.one_hot(d, len(vocab)) for d in reviews]
print(encoded_reviews[0])


# # B3.  Explain the padding process used to standardize the length of sequences, including the following in your explanation: 
# 
# I am padding at the end of reviews to make all sequences the same length as the max sequence (73).

# In[18]:


#Summarized result of padding, with entirety of first review with padding.
#(C2. Tensorflow, 2023)

padded_reviews = keras.preprocessing.sequence.pad_sequences(encoded_reviews, maxlen=max_review_len, padding= 'post')
print(padded_reviews)


# In[19]:


padded_reviews[0]


# ***B4. Identify how many categories of sentiment will be used and an activation function for the final dense layer of the network.***
# 
# There are two categories of sentiment. Positive (1) and Negative (0). 
# Since there are only two I will use the Sigmoid activation function for the final dense layer. (1.Daityari, 2019)
# 
# Since the data has been encoded and padded, I split the data into training and testing sets. I will split it 80% for training and 20% for testing.

# In[20]:


X_train, X_test, y_train, y_test = train_test_split(padded_reviews, df.sentiment, test_size=0.2, random_state=7)


# In[21]:


print(np.any(np.isnan(X_test)))
print(np.any(np.isnan(y_test)))
print(np.any(np.isnan(X_train)))
print(np.any(np.isnan(y_train)))


# ***B5. The above steps were explained on how the data was prepared in B1-B4.***

# ***B6: Saving the copy of prepared data below that I will attach to submission***

# In[22]:


#Save the prepared data
df_final = pd.DataFrame(padded_reviews)
df_final['sentiment'] = df.sentiment
df_final.to_csv('Documents/cleanedD213Task2.csv')


# # Part III:  Network Architecture
# 
# ***C1.  Provide the output of the model summary of the function from TensorFlow.***

# In[23]:


print(df_final.shape); print(X_train.shape); print(X_test.shape)


# In[24]:


dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))


# In[25]:


train_dataset = dataset.shuffle(len(df_final)).batch(1)


# In[26]:


#Model Summary

model = keras.Sequential()
model.add(keras.layers.Embedding(len(vocab), 12, input_length=max_review_len))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dropout(0.2))
model.add(keras.layers.Dense(16, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=['accuracy'])

print(model.summary())

***C2.  Discuss the number of layers, the type of layers, and total number of parameters.***

I am employing 5 layers in my model. First is the embedding layer with 94k parameters, a result of the 7913 words with embedding length of 12. Second layer is the flatten layer used to accompany the embedding layer. Third is a dropout layer to prevent overfitting. Fourth layer is a dense layer with a 14k parameters, results of the 876 input values and 16 nodes. Last layer is the final layer with 17 parameters - resulting from the input values.C3.  Justify the choice of hyperparameters, including the following elements:

•   activation functions

I am using two activation functions. The first is Sigmoid. It will be used in the final layer due to the model moving towards a binary classification. The outputs of Sigmoid range are between 0 and 1. Iam also using Rectified Linear Unit. Relu works great with a sentiment analysis of 0 and 1, because it returns 0 for anything under 0 and actual value for anything above.  (2.Brownlee, 2021)

•   number of nodes per layer

The first layer is the number of words in the model and using one node to prepare for binary classification. The fourth dense layer is using 16 nodes. This is an arbitrary number between the first and last layer node counts that worked well with the model. 

•   loss function

Loss function is Binary Crossenthropy. When there are two possible output values, this is the appropriate loss function.

•   optimizer

I am using Adam for optimization. The Adam optimizer combines two primary advantages of others. It uses Root Mean Square and Adaptive Gradient Algorithm.

•   stopping criteria

I will employ early stopping during the fitting of the model. I will base this stop on the minimum loss value. If there are not consisent loss values after four epochs, I will revert back to the last eopch that had progression.

•   evaluation metric

THe primary evaluations are loss and accuracy. Accuracy is the ratio of correct predictions to the entire target and loss equates to the error I seek to to minimize.


# # Part IV:  Model Evaluation
# 
# ***D1. Discuss the impact of using stopping criteria instead of defining the number of epochs, including a screenshot showing the final training epoch.***
# 
# Stopping criteria can prevent models from overfitting. It also prevents over cycling as the model will stop processing and ignore futher eopochs when it reaches an optimum state.
# 

# In[27]:


model.layers


# In[28]:


model.compile(loss = 'mean_squared_error', optimizer = 'adam', metrics = ['accuracy']
)


# In[29]:


unrestricted = model.fit(X_train, y_train,epochs=20,validation_data=[X_test, y_test])
#(C4.Manmayi, 2023)


# In[30]:


# Now with Early Stopping
earlystopping = keras.callbacks.EarlyStopping(monitor = "val_loss", mode ="min", patience = 5, restore_best_weights = True)
restricted = model.fit(X_train, y_train, epochs=20, validation_data=[X_test, y_test], callbacks = [earlystopping])


# ***D2.  Provide visualizations of the model’s training process, including a line graph of the loss and chosen evaluation metric.***

# Loss Results

# In[31]:


plt.plot(unrestricted.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('MAE (loss)')
plt.title('Loss vs Epoch Count')
plt.show


# Accuracy Results

# In[32]:


plt.plot(unrestricted.history['accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy Versus Epoch Count')
plt.show()


# ***D3. Assess the fitness of the model and any measures taken to address overfitting.***
# 
# My model does show a bit of overfitting when comparing the training accuracy to the test accuracy. The test accuracy covered only 20% while the training was at 80%. To prevent as much overfitting as possible, I used the early stopping method.
# 
# ***D4.  Discuss the predictive accuracy of the trained network.***
# 
# The training set reached 99% accuracy on the 8th epoch. This is quite high and can be due to overfitting. The validation accuracy came in at 77%. Which is still a very high accuracy score. 

# # Part V:  Summary and Recommendations
# 
# ***E.  Provide the code used to save the trained network within the neural network.***

# In[33]:


#Save the model to JSON
model_json = model.to_json()

with open("D213_Sentiment_Model.json", "w") as json_file:
    json_file.write(model_json)

# Save model to HDF5
model.save_weights("D213_Sentiment_Model.h5")


# ***F.  Discuss the functionality of your neural network, including the impact of the network architecture.***
# 
# This model was able to determine positive or negative sentiment on all three dataframes using a neural network. The objective and goals were to clean the dataset so that it could be used for neural network modeling and to decipher the negative and positive sentiments. This was accomplished with a 99% training accuracy and 77% validation accuracy. The model was built using a sequential model with five layers. I would state that the functionality of this model was a success based off initial goals.

# ***G.  Recommend a course of action based on your results.***
# 
# With the accuracy of 99%, we can deterime the general sentiments of the reviews based off the text alone.This data can be used to review customer review sentiments very quickly. I would recommend also re-running this model with fewer layers to see if overfitting can be reduced and accuracy made higher. I also feel that by mixing the types of reviews, it can cause larger confusion. Such as Amazon products will have different review types than imdB movies or Yelp restaraunt reviews. I would recommend to provide a more similar dataset for better comparison.

# # Part VI: Reporting
# 
# ***H.Create your neural network using an industry-relevant interactive development environment (e.g., a Jupyter Notebook). Include a PDF or HTML document of your executed notebook presentation.***
# 
# This report was created using Python's Jupyter Notebook and the PDF saved is from Jupyter Notebook.

# ***I.  List the web sources used to acquire data or segments of third-party code to support the application.***

# # Code References
# 
# C1. Chen, B. (2020, October 30). Pandas CONCAT() tricks you should know to speed up your data analysis. Medium. https://towardsdatascience.com/pandas-concat-tricks-you-should-know-to-speed-up-your-data-analysis-cd3d4fdfe6dd 
# 
# C2. TensorFlow. (n.d.2023). Tf.keras.utils.pad_sequences&nbsp; :&nbsp;  tensorflow V2.12.0. TensorFlow Pad Sequences. https://www.tensorflow.org/api_docs/python/tf/keras/utils/pad_sequences 
# 
# C3. Brownlee, J. (2021, February 1). How to use word embedding layers for deep learning with keras. MachineLearningMastery.com. https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/ 
# 
# C4. manmayi. (2023, February 28). Choose optimal number of epochs to train a neural network in Keras. GeeksforGeeks. https://www.geeksforgeeks.org/choose-optimal-number-of-epochs-to-train-a-neural-network-in-keras/ 
# 

# # Sources
# 
# 1. Daityari, S. (2019, September 26). How to perform sentiment analysis in Python 3 using the Natural Language Toolkit (NLTK). DigitalOcean. https://www.digitalocean.com/community/tutorials/how-to-perform-sentiment-analysis-in-python-3-using-the-natural-language-toolkit-nltk 
# 
# 2. Brownlee, J. (2021a, January 21). How to choose an activation function for deep learning. MachineLearningMastery.com. https://machinelearningmastery.com/choose-an-activation-function-for-deep-learning/ 
# 
# 3. Chen, J. (n.d.). Sentiment analysis behind text with different ... - stanford university. Standford EDU. http://cs230.stanford.edu/projects_fall_2021/reports/102679174.pdf 
