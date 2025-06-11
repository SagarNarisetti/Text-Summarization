
import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))
import numpy as np1 
import pandas as pd 
import matplotlib.pyplot as plt1

import string
from tensorflow.keras import Input, Model
import re as re2

import seaborn as sns
from sklearn.model_selection import train_test_split

import unicodedata
from random import randint
from wordcloud import WordCloud
import tensorflow
from tensorflow.keras.layers import TextVectorization

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# relaive imports
from src.utils import remove_punctuation, remove_punctuation_from_text, remove_number_from_text, remove_stopwords,sequence_to_summary, sequence_to_text
from src.extras import contractions
from src.utils import handle_contractions,clean_text, trim_text_and_summary, rare_words_metrics
from src.model_build import LSTM_Model_build, LSTM_inference, \
    LSTM_decode_sequence, Bidirectional_LSTM_build, Bidirectional_LSTM_inference, Bidirectional_LSTM_decode



data_path = os.path.join(os.getcwd(),'data','extracted_data')

filename1 = 'news_summary.csv'
filename2 = 'news_summary_more.csv'

df1 = pd.read_csv(os.path.join(data_path,filename1), encoding='iso-8859-1').reset_index(drop=True)
df2 = pd.read_csv(os.path.join(data_path,filename2), encoding='iso-8859-1').reset_index(drop=True)
# df1.head()
# print(df2.shape)
# df2.head()
df1=df1[['headlines','text']]
df = pd.concat([df1, df2], axis='index')
# print(df.shape)
# df.head()
df_raw=df.copy()

df.headlines = df.headlines.apply(handle_contractions)
df.text = df.text.apply(handle_contractions)
df.text = df.text.apply(clean_text)
df.headlines = df.headlines.apply(clean_text)
df.tail(5)
wc_text = WordCloud(width=600,height=300).generate(' '.join(df.text))
plt1.imshow(wc_text)
wc_summary = WordCloud(width=600,height=300).generate(' '.join(df.headlines))
plt1.imshow(wc_summary)
df.headlines = df.headlines.apply(lambda x: f'_START_ {x} _END_')
start_token = 'sostok'
end_token = 'eostok'
df.headlines = df.headlines.apply(lambda x: f'{start_token} {x} {end_token}')
text_value_count = [len(val.split()) for val in df.text]
headlines_value_count = [len(val.split()) for val in df.headlines]

pd.DataFrame({'text': text_value_count, 'headlines': headlines_value_count}).hist(bins=100, figsize=(20, 5), range=[0, 50])
plt1.show()


# maximum text and summary lengths
maximum_text_length = 43
maximum_summary_length = 13
def max_length_percentage(data,number):
    d=0
    for i1 in data:
        if(len(i1.split())<=number):
            d=d+1
    percentge=round(d/len(data),2)
    return percentge
max_length_percentage(df.text,maximum_text_length),max_length_percentage(df.headlines,maximum_summary_length)
# select the summary and text between their defined max lens respectively
df = trim_text_and_summary(df, maximum_text_length, maximum_summary_length)
print(f'Dataset size: {len(df)}')
df.sample(5)

# Splitting the training and validation sets
x_train, x_val, y_train, y_val = train_test_split(np1.array(df['text']),np1.array(df['summary']),
                                                  test_size=0.1,random_state=1,shuffle=True)
x_tk = Tokenizer()
x_tk.fit_on_texts(list(x_train))
text_word_count,text_total_word_count=rare_words_metrics(x_tk,4)
x_tk = Tokenizer(num_words=text_total_word_count-text_word_count)
x_tk.fit_on_texts(list(x_train))
# one-hot-encoding
x_train_sequence = x_tk.texts_to_sequences(x_train)
x_val_sequence = x_tk.texts_to_sequences(x_val)

# padd max length
x_train_padded = pad_sequences(x_train_sequence, maxlen=maximum_text_length, padding='post')
x_val_padded = pad_sequences(x_val_sequence, maxlen=maximum_text_length, padding='post')

# if you're not using num_words parameter in Tokenizer then use this
x_vocab_size = len(x_tk.word_index) + 1

# else use this
# x_vocab_size = x_tk.num_words + 1
x_vocab=x_vocab_size
print(x_vocab_size)
y_tk = Tokenizer()
y_tk.fit_on_texts(list(y_train))
headlines_word_count,headlines_total_word_count=rare_words_metrics(y_tk,5)
y_tk = Tokenizer(headlines_total_word_count-headlines_word_count)
y_tk.fit_on_texts(list(y_train))
# one-hot-encoding
y_train_sequence = y_tk.texts_to_sequences(y_train)
y_val_sequence = y_tk.texts_to_sequences(y_val)

# padding upto maximum_summary_length
y_train_padded = pad_sequences(y_train_sequence, maxlen=maximum_summary_length, padding='post')
y_val_padded = pad_sequences(y_val_sequence, maxlen=maximum_summary_length, padding='post')

# if you're not using num_words parameter in Tokenizer then use this
y_vocab_size = len(y_tk.word_index) + 1

# else use this
# y_vocab_size = y_tk.num_words + 1
y_vocab=y_vocab_size
print(y_vocab_size)
# removing the rows in training data where the out put sentence only contains <START> and <STOP> and no other words
remove_ind_Training=[]
for val in range(len(y_train_padded)):
    count=0
    # print(y_train_padded)
    for g in y_train_padded[val]:
        if g!=0:
            count=count+1
    if(count==2):
        remove_ind_Training.append(val)

y_train_padded=np1.delete(y_train_padded,remove_ind_Training, axis=0)
x_train_padded=np1.delete(x_train_padded,remove_ind_Training, axis=0)
# removing the rows of validation data where the output value contains only <START> and <STOP>

remove_ind_validation=[]
for val in range(len(y_val_padded)):
    count=0
    for g in y_val_padded[val]:
        if g!=0:
            count=count+1
    if(count==2):
        remove_ind_validation.append(val)

y_val_padded=np1.delete(y_val_padded,remove_ind_validation, axis=0)
x_val_padded=np1.delete(x_val_padded,remove_ind_validation, axis=0)

latent_dimension = 250
embeding_dimension = 300
num_of_epochs = 100
# LSTM
LSTM_seq2seq = LSTM_Model_build(embeding_dimension, latent_dimension, maximum_text_length,x_vocab_size, y_vocab_size)
LSTM_model = LSTM_seq2seq['model']

encoder_input = LSTM_seq2seq['inputs']['encoder']
decoder_input = LSTM_seq2seq['inputs']['decoder']

encoder_output = LSTM_seq2seq['outputs']['encoder']
decoder_output = LSTM_seq2seq['outputs']['decoder']

encoder_final_states = LSTM_seq2seq['states']['encoder']
decoder_final_states = LSTM_seq2seq['states']['decoder']

decoder_embeding_layer = LSTM_seq2seq['layers']['decoder']['embedding']
last_decoder_lstm = LSTM_seq2seq['layers']['decoder']['last_decoder_lstm']
decoder_dense = LSTM_seq2seq['layers']['decoder']['dense']
callbacks = [EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.000001, verbose=1),]
LSTM_history = LSTM_model.fit([x_train_padded, y_train_padded[:, :-1]],
                    y_train_padded.reshape(y_train_padded.shape[0], y_train_padded.shape[1], 1)[:, 1:],
                    epochs=num_of_epochs,batch_size=128,callbacks=callbacks,
                    validation_data=( [x_val_padded, y_val_padded[:, :-1]],y_val_padded.reshape(y_val_padded.shape[0], y_val_padded.shape[1], 1)[:, 1:]))

# index to word _dictionary
reverse_target_word_index = y_tk.index_word
# Accuracy
plt1.plot(LSTM_history.history['accuracy'][1:], label='train acc')
plt1.plot(LSTM_history.history['val_accuracy'], label='val')
plt1.xlabel('Epoch')
plt1.ylabel('Accuracy')
plt1.legend(loc='lower right')
# Loss
plt1.plot(LSTM_history.history['loss'][1:], label='train loss')
plt1.plot(LSTM_history.history['val_loss'], label='val')
plt1.xlabel('number of Epoch')
plt1.ylabel('losses during training')
plt1.legend(loc='lower right')

reverse_source_word_index = x_tk.index_word
target_word_index = y_tk.word_index

LSTM_encoder_model, LSTM_decoder_model = LSTM_inference(maximum_text_length, latent_dimension, encoder_input, encoder_output,
                                              encoder_final_states, decoder_input, decoder_output,
                                              decoder_embeding_layer, decoder_dense, last_decoder_lstm)
LSTM_encoder_model.summary()
LSTM_decoder_model.summary()

# Testing on validation data
for val in range(0, 2):
    print(f"# {val+1} News: ", sequence_to_text(x_val_padded[val], reverse_source_word_index))
    print("Original_Summary: ", sequence_to_summary(y_val_padded[val], target_word_index, reverse_target_word_index, start_token, end_token))
    print("Predicted_Summary: ", LSTM_decode_sequence(x_val_padded[val].reshape(1, maximum_text_length), LSTM_encoder_model,
                                                      LSTM_decoder_model, maximum_summary_length))
    print()

Biseq2seq = Bidirectional_LSTM_build(embeding_dimension, latent_dimension, maximum_text_length,x_vocab_size, y_vocab_size)
Bi_LSTM_model = Biseq2seq['model']

encoder_input = Biseq2seq['inputs']['encoder']
decoder_input = Biseq2seq['inputs']['decoder']

encoder_output = Biseq2seq['outputs']['encoder']
decoder_output = Biseq2seq['outputs']['decoder']

encoder_final_states = Biseq2seq['states']['encoder']
decoder_final_states = Biseq2seq['states']['decoder']

decoder_embeding_layer = Biseq2seq['layers']['decoder']['embedding']
last_decoder_lstm = Biseq2seq['layers']['decoder']['last_decoder_lstm']
decoder_dense = Biseq2seq['layers']['decoder']['dense']
Bi_LSTM_model.summary()
callbacks = [EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.000001, verbose=1),]
Bi_LSTM_history = Bi_LSTM_model.fit([x_train_padded, y_train_padded[:, :-1]],
                            y_train_padded.reshape(y_train_padded.shape[0], y_train_padded.shape[1], 1)[:, 1:],
                            epochs=num_of_epochs,batch_size= 64,callbacks=callbacks,
                            validation_data=([x_val_padded, y_val_padded[:, :-1]],
                                             y_val_padded.reshape(y_val_padded.shape[0], y_val_padded.shape[1], 1)[:, 1:]))


# Accuracy
plt1.plot(Bi_LSTM_history.history['accuracy'][1:], label='train acc')
plt1.plot(Bi_LSTM_history.history['val_accuracy'], label='val')
plt1.xlabel('Number of Epochs')
plt1.ylabel('Accuracy_score')
plt1.legend(loc='lower right')
# Loss
plt1.plot(Bi_LSTM_history.history['loss'][1:], label='train loss')
plt1.plot(Bi_LSTM_history.history['val_loss'], label='val')
plt1.xlabel('Number of Epochs')
plt1.ylabel('Loss_values')
plt1.legend(loc='lower right')

# Inference
BiLSTM_encoder_model, BiLSTM_decoder_model = Bidirectional_LSTM_inference(maximum_text_length, latent_dimension, encoder_input, encoder_output,
                                              encoder_final_states, decoder_input, decoder_output,
                                              decoder_embeding_layer, decoder_dense, last_decoder_lstm)
print(BiLSTM_encoder_model.summary())
print(BiLSTM_decoder_model.summary())

# Testing on validation data
for val in range(0, 3):
    print(f"# {val+1} Text: ", sequence_to_text(x_val_padded[val]))
    print("Original_summary: ", sequence_to_summary(y_val_padded[val],start_token, end_token))
    print("Predicted_summary: ", Bidirectional_LSTM_decode(x_val_padded[val].reshape(1, maximum_text_length),
                                                           BiLSTM_encoder_model,BiLSTM_decoder_model, maximum_summary_length))
    print()


# Transformers

from summarizer import Summarizer,TransformerSummarizer
bert_model = Summarizer()
GPT2_model = TransformerSummarizer(transformer_type="GPT2",transformer_model_key="gpt2-medium")
min_length_text = 40
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
rouge=Rouge()
LSTM_perdiction=[]
BiLSTM_prediction=[]
BERT_prediction=[]
GPT2_prediction=[]
ORIGINAL_text=[]
ORIGINAL_summary=[]
for i in range(0,1):
    original_text=sequence_to_text(x_val_padded[i])
    ORIGINAL_text.append(original_text)
    print('#    original text : ',original_text)

    original_summary=sequence_to_summary(y_val_padded[i],start_token, end_token)
    ORIGINAL_summary.append(original_summary)
    print('#    original summary : ',original_summary)

    gpt_2_summary=''.join(GPT2_model(sequence_to_text(x_val_padded[i]), min_length=min_length_text))
    GPT2_prediction.append(gpt_2_summary)
    print('#    GPT-2 summary : ',gpt_2_summary)

    BERT_summary=''.join(bert_model(sequence_to_text(x_val_padded[i]), min_length=min_length_text))
    BERT_prediction.append(BERT_summary)
    print('#     BERT summary : ',BERT_summary)

    LSTM_summary=LSTM_decode_sequence(x_val_padded[i].reshape(1, maximum_text_length),LSTM_encoder_model,LSTM_decoder_model, maximum_summary_length)
    LSTM_perdiction.append(LSTM_summary)
    print('#     LSTM summary : ',LSTM_summary)

    BiLSTM_summary=Bidirectional_LSTM_decode(x_val_padded[i].reshape(1,maximum_text_length),BiLSTM_encoder_model,BiLSTM_decoder_model, maximum_summary_length)
    BiLSTM_prediction.append(BiLSTM_summary)
    print('#   BiLSTM summary : ', BiLSTM_summary)
rouge_LSTM=rouge.get_scores(LSTM_perdiction,ORIGINAL_summary)
rouge_BiLSTM=rouge.get_scores(BiLSTM_prediction,ORIGINAL_summary)
rouge_BERT=rouge.get_scores(BERT_prediction,ORIGINAL_summary)
rouge_GPT2=rouge.get_scores(GPT2_prediction,ORIGINAL_summary)
print('rouge_LSTM',rouge_LSTM)
print('rouge_BiLSTM',rouge_BiLSTM)
print('rouge_BERT',rouge_BERT)
print('rouge_GPT2',rouge_GPT2)
