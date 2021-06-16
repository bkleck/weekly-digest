import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import tokenization
import logging

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import wget

logging.basicConfig(level=logging.INFO)

# this is used to activate GPU and prevent Blas xGEMM error
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# just run this once, for the first time
# url = 'https://raw.githubusercontent.com/tensorflow/models/master/official/nlp/bert/tokenization.py'
# filename = wget.download(url)

module_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
bert_layer = hub.KerasLayer(module_url, trainable=True)


## Load data
print('')
print("Please type today's date in the following format (110621)")
# all files will be saved with today's date at the front
# input file must also be in the same format
date = input('Please enter date: ')
total_time = datetime.datetime.now() # track run-time

input_file_1 = 'data/scrape/' + date + '_clean_google.csv'
df1 = pd.read_csv(input_file_1, index_col=0)
df1.rename(columns={'Article':'article'}, inplace=True)

input_file_2 = 'data/scrape/' + date + '_clean_baidu.csv'
df2 = pd.read_csv(input_file_2, index_col=0)
df2.rename(columns={'Article':'article'}, inplace=True)


## BERT functions

# create encodings for BERT
vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)

def bert_encode(texts, tokenizer, max_len=512):
    all_tokens = []
    all_masks = []
    all_segments = []
    
    for text in texts:
        text = tokenizer.tokenize(text)
            
        text = text[:max_len-2]
        input_sequence = ["[CLS]"] + text + ["[SEP]"]
        pad_len = max_len - len(input_sequence)
        
        tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
        pad_masks = [1] * len(input_sequence) + [0] * pad_len
        segment_ids = [0] * max_len
        
        all_tokens.append(tokens)
        all_masks.append(pad_masks)
        all_segments.append(segment_ids)
    
    return np.array(all_tokens), np.array(all_masks), np.array(all_segments)

    # build BERT model with neural network classifer
def build_model(bert_layer, max_len=512):
    input_word_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_word_ids")
    input_mask = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="input_mask")
    segment_ids = tf.keras.Input(shape=(max_len,), dtype=tf.int32, name="segment_ids")

    pooled_output, sequence_output = bert_layer([input_word_ids, input_mask, segment_ids])
    clf_output = sequence_output[:, 0, :]
    net = tf.keras.layers.Dense(64, activation='relu')(clf_output)
    net = tf.keras.layers.Dropout(0.2)(net)
    net = tf.keras.layers.Dense(32, activation='relu')(net)
    net = tf.keras.layers.Dropout(0.2)(net)
    out = tf.keras.layers.Dense(2, activation='softmax')(net)
    
    # activation change to sigmoid for multilabel
    # loss is bce loss
    # metrics accuracy

    model = tf.keras.models.Model(inputs=[input_word_ids, input_mask, segment_ids], outputs=out)
    model.compile(tf.keras.optimizers.Adam(lr=1e-5), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    return model


# encode predictions into labels
# first element is for 0
# second element is for 1
def encode(row):
  i = row[0]
  if round(i) == 0:
    val = i # changed from 1 to i to keep original logit score -- used to keep the largest signal and drop the rest to 0
  else:
    val = 0
  return val



## Run predictions

# max_len = 150 # of the sentence
# embedding = bert_encode(df.article.values, tokenizer, max_len=max_len)

# model = build_model(bert_layer, max_len=150)
# model.summary()

signals = ['revenue','product','market','partnership','mgmt','clinical','fundraising']
paths = ['models/news_rev.h5', 'models/news_pdt.h5', 'models/news_mkt.h5', 'models/news_partnership.h5', 
          'models/news_mgmt_combine.h5', 'models/news_clinical.h5', 'models/news_fundraising.h5']

def run_preds(df, signals, paths, bert_layer):
  embedding = bert_encode(df.article.values, tokenizer, max_len=150)
  model = build_model(bert_layer, max_len=150)

  for idx, signal in enumerate(signals):
    begin_time = datetime.datetime.now() # track run-time

    # get the path from the appropriate index in corresponding list
    path = paths[idx]

    # load and predict with each set of weights
    model.load_weights(path)
    pred = model.predict(embedding)

    pred_list = pred.tolist()
    # save predictions back to dataframe under the relevant column
    df[signal] = pred_list
    # encode into 1 or 0
    df[signal] = df[signal].apply(lambda x: encode(x))

    print(str(signal) + ' predictions have been completed!')
    print(datetime.datetime.now() - begin_time)

  # do a min-max normalization for results
  for signal in signals:
    
    # only do min-max if the column is not entirely 0
    na_count = (df[signal] == 0).sum()
    if na_count != len(df):
      df[signal] = (df[signal] - df[signal].min()) / (df[signal].max() - df[signal].min()) 

    # separate the numerical columns away for manipulation
    front_df = df.loc[:, 'Search Query': 'article']
    back_df = df.loc[:, 'revenue':'fundraising']

    # keep only the largest signal score
    m = np.zeros_like(back_df.values)
    m[np.arange(len(df)), back_df.values.argmax(1)] = 1
    df1 = pd.DataFrame(m, columns = back_df.columns).astype(int)

    # combine back the numerical columns
    final_df = pd.concat([front_df, df1], axis=1)
    return final_df

print(' ')
print('Google news:')
df1 = run_preds(df1, signals, paths, bert_layer)
output_file_1 = 'data/' + date + '_signal_google_predictions.csv'
df1.to_csv(output_file_1)

print(' ')
print('Baidu news:')
df2 = run_preds(df2, signals, paths, bert_layer)
output_file_2 = 'data/' + date + '_signal_baidu_predictions.csv'
df2.to_csv(output_file_2)

print(' ')
print('Both predictions have been saved to data folder!')
print('Total time taken:')
print(datetime.datetime.now() - total_time)