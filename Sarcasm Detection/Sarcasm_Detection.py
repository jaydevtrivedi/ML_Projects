# Code to authenticate with google drive
!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse
from google.colab import auth
auth.authenticate_user()
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()
import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}

# Create your drive
!mkdir -p drive
!google-drive-ocamlfuse drive

# Upload your dataset here
!ls /content/drive/Datasets/sarcasm

# Installing tensorflow 2 (+ Restart Runtime)
!pip install tensorflow-gpu==2.0.0-beta1

# Program Code

# Step 1 : Declaring Constants
file_train_balanced_sarcasm = '/content/drive/Datasets/sarcasm/train-balanced-sarcasm.csv'
file_test_balanced = '/content/drive/Datasets/sarcasm/test-balanced.csv'
file_test_unbalanced = '/content/drive/Datasets/sarcasm/test-unbalanced.csv'

stopwords = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
             "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do",
             "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having",
             "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how",
             "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself",
             "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought",
             "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should",
             "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
             "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through",
             "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were",
             "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
             "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
             "yourselves"]

TRAIN_SIZE = 0.9
RANDOM_STATE = 0
NUM_EPOCHS = 10
BATCH_SIZE = 32

vocab_size = 1000
embedding_dim = 16
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 2500
end = 3000

# Step 2 : Importing all libraries
import csv
import numpy as np
import tensorflow as tf

print(tf.__version__)
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Step 3 : Get the dataset and explore
dataset = pd.read_csv(file_train_balanced_sarcasm)
print(dataset.columns)
print(dataset.iloc[0,:].values)

# Merging parent comment and result sarcasm/no sarcasm and see how the models go
comments = dataset.pop('parent_comment') + dataset.pop('comment')
labels = dataset.pop('label')

# Step 4 : train_test_split
train_dataset, validation_dataset, train_label, validation_label = train_test_split(comments,
                                                                                    labels,
                                                                                    train_size=TRAIN_SIZE, random_state=RANDOM_STATE)

train_dataset, predict_dataset, train_label, predict_label = train_test_split(train_dataset, train_label,
                                                                              train_size=TRAIN_SIZE, random_state=RANDOM_STATE)

# Step 5 : Remove Stop words
# Clean data from stopwords
def remove_stopwords(sentence):
    for token in stopwords:
        token = " " + token + " "
        sentence = str(sentence)
        sentence = sentence.lower()
        sentence = sentence.replace(token, " ")
        sentence = sentence.replace("  ", " ")
    return sentence


train_dataset = train_dataset.apply(remove_stopwords)
validation_dataset = validation_dataset.apply(remove_stopwords)
predict_dataset = predict_dataset.apply(remove_stopwords)

# Step 6 : Tokenize the dataset
sentence_tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
sentence_tokenizer.fit_on_texts(comments.apply(str))
word_index = sentence_tokenizer.word_index
print(len(word_index))

train_dataset = sentence_tokenizer.texts_to_sequences(train_dataset)
train_dataset_padded = pad_sequences(train_dataset, padding=padding_type, maxlen=max_length, truncating=trunc_type)

validation_dataset = sentence_tokenizer.texts_to_sequences(validation_dataset)
validation_dataset_padded = pad_sequences(validation_dataset, padding=padding_type, maxlen=max_length,
                                          truncating=trunc_type)

predict_dataset = sentence_tokenizer.texts_to_sequences(predict_dataset)
predict_dataset_padded = pad_sequences(predict_dataset, padding=padding_type, maxlen=max_length, truncating=trunc_type)

# Step 7 : Declare the models
def single_layer_lstm(vocab_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def multi_layer_lstm(vocab_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 64),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def bidirectional_lstm(vocab_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def conv1D(vocab_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, 64),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def conv1d_sarc(vocab_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Conv1D(128, 5, activation='relu'),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def multilayer_gru(vocab_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.GRU(32)),
        tf.keras.layers.Dense(6, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


list_func = [single_layer_lstm, multi_layer_lstm, bidirectional_lstm, conv1D, conv1d_sarc, multilayer_gru]
dict_results = dict()

# Step 8 : Train the models
for func in list_func:
    model = func(vocab_size)
    model.fit(train_dataset_padded, train_label, epochs=NUM_EPOCHS,
                    validation_data=(validation_dataset_padded, validation_label), verbose=2)
    dict_results[func.__name__] = r2_score(predict_label, model.predict(predict_dataset_padded))

# Step 9 : Print out results
print(dict_results)