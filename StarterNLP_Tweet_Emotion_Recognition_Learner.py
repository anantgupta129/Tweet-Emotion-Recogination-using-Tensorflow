# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] id="sp7D0ktn5eiG"
# # Tweet Emotion Recognition: Natural Language Processing with TensorFlow
#

# %% [markdown] id="cprXxkrMxIgT"
# ## 1: Setup and Imports
#

# %% id="yKFjWz6e5eiH" outputId="899b1e41-d820-43fb-a11e-5f35ecba0161" colab={"base_uri": "https://localhost:8080/"}
# %matplotlib inline

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def show_history(h):
    epochs_trained = len(h.history['loss'])
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(0, epochs_trained), h.history.get('accuracy'), label='Training')
    plt.plot(range(0, epochs_trained), h.history.get('val_accuracy'), label='Validation')
    plt.ylim([0., 1.])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(0, epochs_trained), h.history.get('loss'), label='Training')
    plt.plot(range(0, epochs_trained), h.history.get('val_loss'), label='Validation')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    
def show_confusion_matrix(y_true, y_pred, classes):
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred, normalize='true')

    plt.figure(figsize=(8, 8))
    sp = plt.subplot(1, 1, 1)
    ctx = sp.matshow(cm)
    plt.xticks(list(range(0, len(classes))), labels=classes)
    plt.yticks(list(range(0, len(classes))), labels=classes)
    plt.colorbar(ctx)
    plt.show()

    
print('Using TensorFlow version', tf.__version__)

# %% [markdown] id="7JsBpezExIga"
# ## 3: Importing Data
#
# 1. Importing the Tweet Emotion dataset
# 2. Creating train, validation and test sets
# 3. Extracting tweets and labels from the examples

# %% id="0YHOvjAu5eiL" outputId="2a267d57-b307-40dc-b718-02a336dbfdf5" colab={"base_uri": "https://localhost:8080/"}
# !pip install nlp
import nlp
dataset = nlp.load_dataset('emotion')

# %% id="2s0h541FxIgc" outputId="8858ab84-0102-4757-c5ed-89a884fd7f7c" colab={"base_uri": "https://localhost:8080/"}
dataset

# %% id="z7eCnxU25eiN"
train = dataset['train']
val = dataset['validation']
test = dataset['test']


# %% id="oDYXMfZy5eiP"
def get_tweet(data):
  tweets = [x['text'] for x in data]
  labels = [x['label'] for x in data]
  return tweets, labels


# %% id="jeq3-vSB5eiR"
tweets_train, labels_train = get_tweet(train)

# %% id="bHD3Tk0J5eiU" outputId="bd07d344-e695-40a5-c2aa-2bb8ae97b467" colab={"base_uri": "https://localhost:8080/"}
tweets_train[1:4], labels_train[1:4]

# %% [markdown] id="gcAflLv6xIgp"
# ##  4: Tokenizer
#
# Tokenizing the tweets, it is a text preprocessing module represents words as numbers

# %% id="qfX5-ResxIgq"
from keras.preprocessing.text import Tokenizer

# %% id="cckUvwBo5eif"
tokenizer = Tokenizer(num_words=10000, oov_token='<UNK>')  # any word not in vocablury is represented by oov_token
tokenizer.fit_on_texts(tweets_train)

# %% id="VzvNh4PZ7FNx" outputId="c8ee1cb8-b3e8-49b6-eba1-d91d889f06bd" colab={"base_uri": "https://localhost:8080/"}
tokenizer.texts_to_sequences([tweets_train[0]])

# %% id="jU7Krvdm7jj2" outputId="503390ab-be3a-48c5-9c2b-4d8d0e2596ae" colab={"base_uri": "https://localhost:8080/", "height": 35}
tweets_train[0]

# %% [markdown] id="i3Bqm7b2xIgu"
# ## 5: Padding and Truncating Sequences
# padding is requires as the model wants input of same length
#
# 1. Checking length of the tweets
# 2. Creating padded sequences
#
# truncating if sentence longer then maxlen
# then the ending will truncated means will not be included in list

# %% id="mLvf_WFZxIgu" outputId="eed4219a-c30f-4f5d-dbec-e8e508745d65" colab={"base_uri": "https://localhost:8080/", "height": 265}
lengths = [len(t.split()) for t in tweets_train]
plt.hist(lengths, bins=len(set(lengths)))
plt.show() 

# %% id="EOi5lIE3xIgx"
maxlen = 50
from keras.preprocessing.sequence import pad_sequences

def get_seq(tokenizer, tweets, maxlen):
  seq = tokenizer.texts_to_sequences(tweets)
  padded = pad_sequences(seq, truncating='post', padding='post', maxlen=maxlen)
  return padded


# %% id="Q9J_Iemf5eiq"
train_padded = get_seq(tokenizer, tweets_train, maxlen)

# %% id="eglH77ky5ei0" outputId="d2a81f14-f4d1-4862-c075-45fed6843db2" colab={"base_uri": "https://localhost:8080/"}
train_padded[0]

# %% [markdown] id="BURhOX_KxIg8"
# ##  6: Preparing the Labels
#
# 1. Creating classes to index and index to classes dictionaries
# 2. Converting text labels to numeric labels

# %% id="SufT2bpD5ejE" outputId="619cbb7d-d180-459a-cce8-63131dbac247" colab={"base_uri": "https://localhost:8080/"}
set(labels_train)

# %% id="rpwzL88I7YSm" outputId="9f231f05-3120-4ca4-c096-3355c92f5a2d" colab={"base_uri": "https://localhost:8080/", "height": 265}
plt.hist(labels_train, bins=12)
plt.show()

# %% id="dNLF6rXL5ejN"
class_to_index = dict((c,i) for i, c in enumerate(set(labels_train)))
index_to_class = dict((v,k) for k, v in class_to_index.items())

# %% id="_08InVyM5ejc" outputId="1ab09525-18ea-4517-d44a-43c647a68c15" colab={"base_uri": "https://localhost:8080/"}
class_to_index

# %% id="gpeDoA6gxIhE" outputId="19aac171-92f9-40fa-9e16-c40f0e3b9e43" colab={"base_uri": "https://localhost:8080/"}
index_to_class

# %% id="Jq0WJYsP5ejR"
get_labels_ids = lambda labels: np.array([class_to_index[x] for x in labels])

# %% id="v15KnrNC5ejW" outputId="3115545d-adbd-4f09-a574-bcd9ff383b3e" colab={"base_uri": "https://localhost:8080/"}
train_labels = get_labels_ids(labels_train)
print(train_labels[0])

# %% [markdown] id="c-v0Mnh8xIhP"
# ## 7: Creating the Model
#
# 1. Creating the model
# 2. Compiling the model

# %% id="OpewXxPQ5eji" outputId="9df862cf-300d-44f8-fa0a-e6405959f596" colab={"base_uri": "https://localhost:8080/"}
from keras.layers import Embedding, Bidirectional, LSTM, Dense

model = tf.keras.models.Sequential([Embedding(10000, 16, input_length=maxlen),
                                    Bidirectional(LSTM(20, return_sequences=True)),
                                    Bidirectional(LSTM(20)),
                                    Dense(6, activation='softmax')
                                  ])
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy']
              )
model.summary()

# %% [markdown] id="1HST_CHjxIhR"
# ##  8: Training the Model
#
# 1. Preparing a validation set
# 2. Training the model

# %% id="Ff7F3hCK5ejm"
val_tweets, val_labels = get_tweet(val)
val_padded = get_seq(tokenizer, val_tweets, maxlen)
val_labels = get_labels_ids(val_labels)

# %% id="hlMKaZ3H5ejr" outputId="dadfb9bb-ab58-44ee-d516-0b72c5c25f98" colab={"base_uri": "https://localhost:8080/"}
val_tweets[0], val_labels[0]

# %% id="bzBqnWQ-5ejw" outputId="3df97d92-e335-4b91-ad5c-30b5fb3cc896" colab={"base_uri": "https://localhost:8080/"}
h = model.fit(train_padded, train_labels,
              validation_data=(val_padded, val_labels),
              epochs=25)

# %% [markdown] id="EdsJyMTLxIhX"
# ##  9: Evaluating the Model
#
# 1. Visualizing training history
# 2. Prepraring a test set
# 3. A look at individual predictions on the test set
# 4. A look at all predictions on the test set

# %% id="ENCfvXeLxIhX" outputId="76962a21-1e66-4f5f-8de6-0c53c642f758" colab={"base_uri": "https://localhost:8080/", "height": 392}
show_history(h)

# %% id="kWuzoz8uxIha"
test_tweets, test_labels = get_tweet(test)
test_padded = get_seq(tokenizer, test_tweets, maxlen)
test_labels = get_labels_ids(test_labels)

# %% id="7vRVJ_2SxIhc" outputId="f4e59266-53e2-49bf-f204-0d2184804a79" colab={"base_uri": "https://localhost:8080/"}
model.evaluate(test_padded, test_labels)

# %% id="rh638vHG5ej6" outputId="e8469e20-b242-4ca5-afb4-8afec501ead7" colab={"base_uri": "https://localhost:8080/"}
import random
i = random.randint(0, len(test_labels)-1)

print('Sentence 1:', test_tweets[i])
print('org emotion 1:', index_to_class[test_labels[i]])

p = model.predict(np.expand_dims(test_padded[i], axis=0))[0]
pred_class = index_to_class[np.argmax(p).astype('uint8')]

print('Predicted Emotion 1 :', pred_class)
i = random.randint(0, len(test_labels)-1)

print('Sentence 2:', test_tweets[i])
print('org emotion 2:', index_to_class[test_labels[i]])

p = model.predict(np.expand_dims(test_padded[i], axis=0))[0]
pred_class = index_to_class[np.argmax(p).astype('uint8')]

print('Predicted Emotion 2:', pred_class)

# %% id="hHl5SVCFxIhh"
preds = model.predict(test_padded)
preds = np.argmax(preds, axis=-1)

# %% id="NC8YQ0OexIhj" outputId="f40b48f7-ab58-437f-d270-7326c14c52f3" colab={"base_uri": "https://localhost:8080/", "height": 472}
show_confusion_matrix(test_labels, preds, list(set(labels_train)))
