
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np1 
import pandas as pd 




filename1 = r'/kaggle/input/news-summary/news_summary.csv'
filename2 = r'/kaggle/input/news-summary/news_summary_more.csv'

df1 = pd.read_csv(filename1, encoding='iso-8859-1').reset_index(drop=True)
df2 = pd.read_csv(filename2, encoding='iso-8859-1').reset_index(drop=True)
df1.head()
print(df2.shape)
df2.head()
df1=df1[['headlines','text']]
df=df = pd.concat([df1, df2], axis='rows')
print(df.shape)
df.head()
df_raw=df.copy()
import matplotlib.pyplot as plt1

import string
from tensorflow.keras import Input, Model

import seaborn as sns
import re as re2
from sklearn.model_selection import train_test_split

import unicodedata
from random import randint
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Embedding, TimeDistributed
from wordcloud import WordCloud
from tensorflow.keras.preprocessing.text import Tokenizer
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Removal of the  punctuation from words
def remove_punctuation(word):
    cleaned_list = [value for value in word if value not in string.punctuation]
    return ''.join(cleaned_list)
# Removal of punctuation from the text data
def remove_punctuation_from_text(text):
    cleaned_text = [remove_punctuation(value) for value in text]
    return ''.join(cleaned_text)
# Remove numericals from the text data
def remove_number_from_text(sequence):
    sequence = re2.sub('[0-9]+', '', sequence)
    return ' '.join(sequence.split())
# handling stop words
def remove_stopwords(sequence):
    stop_words = stopwords.words('english')
    sequence = sequence.split()
    word_sequence = [value for value in sequence if value not in stop_words]
    return ' '.join(word_sequence)
contractions = {
    "ain't": "am not",  "aren't": "are not",    "can't": "can not", "can't've": "can not have", "'cause": "because",    "could've": "could have","couldn't": "could not",
    "couldn't've": "could not have","didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hadn't've": "had not have", "hasn't": "has not",
    "haven't": "have not", "he'd": "he would", "he'd've": "he would have", "he'll": "he will", "he'll've": "he will have", "he's": "he is", "how'd": "how did",
    "how'd'y": "how do you",  "how'll": "how will", "how's": "how is", "i'd": "i would", "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have",
    "i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would", "it'd've": "it would have", "it'll": "it will","it'll've": "it will have", "it's": "it is",
    "let's": "let us", "ma'am": "madam", "mayn't": "may not", "might've": "might have", "mightn't": "might not", "mightn't've": "might not have", "must've": "must have",
    "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have", "o'clock": "of the clock","oughtn't": "ought not",
    "oughtn't've": "ought not have", "shan't": "shall not","sha'n't": "shall not", "shan't've": "shall not have", "she'd": "she would","she'd've": "she would have",
    "she'll": "she will", "she'll've": "she will have", "she's": "she is","should've": "should have",    "shouldn't": "should not",    "shouldn't've": "should not have",
    "so've": "so have",    "so's": "so as",    "that'd": "that would",    "that'd've": "that would have",    "that's": "that is",    "there'd": "there would",
    "there'd've": "there would have",    "there's": "there is",    "they'd": "they would",    "they'd've": "they would have",    "they'll": "they will",    "they'll've": "they will have",
    "they're": "they are",    "they've": "they have",    "to've": "to have",    "wasn't": "was not",    "we'd": "we would",    "we'd've": "we would have",
    "we'll": "we will",    "we'll've": "we will have",    "we're": "we are",    "we've": "we have",    "weren't": "were not",    "what'll": "what will",
    "what'll've": "what will have",    "what're": "what are",    "what's": "what is",    "what've": "what have",    "when's": "when is",    "when've": "when have",
    "where'd": "where did",    "where's": "where is",   "where've": "where have",    "who'll": "who will",    "who'll've": "who will have",    "who's": "who is",
    "who've": "who have",    "why's": "why is",    "why've": "why have",    "will've": "will have",    "won't": "will not",    "won't've": "will not have",
    "would've": "would have",    "wouldn't": "would not",    "wouldn't've": "would not have",    "y'all": "you all",    "y'all'd": "you all would",
    "y'all'd've": "you all would have",    "y'all're": "you all are",    "y'all've": "you all have",    "you'd": "you would",    "you'd've": "you would have",
    "you'll": "you will",    "you'll've": "you will have",    "you're": "you are",    "you've": "you have"
    }
def handle_contractions(txt, contraction_map=contractions):
    # Using regex for getting all contracted words
    contractions_keys = '|'.join(contraction_map.keys())
    contra_patt = re2.compile(f'({contractions_keys})', flags=re2.DOTALL)

    def expd_word(contraction):
        # Getting entire matched sub-string
        match = contraction.group(0)
        es = contraction_map.get(match)
        if not es:
            print(match)
            return match
        return es

    es = contra_patt.sub(expd_word, txt)
    es = re2.sub("'", "", es)
    return es

df.headlines = df.headlines.apply(handle_contractions)
df.text = df.text.apply(handle_contractions)
# Cleaning text
def clean_text(sequence):
    sequence = sequence.lower()
    sequence = remove_punctuation_from_text(sequence)
    sequence = remove_number_from_text(sequence)
    sequence = remove_stopwords(sequence)


    # hadling special symbols like hyphens 
    sequence = re2.sub('–', '', sequence)
    sequence = ' '.join(sequence.split())  # removing `extra` white spaces

    # Removing unnecessary characters from text
    sequence = re2.sub("(\\t)", ' ', str(sequence)).lower()
    # replacing \\t values with empty spaces
    sequence = re2.sub("(\\r)", ' ', str(sequence)).lower()
    # replacing \\r values with spaces
    sequence = re2.sub("(\\n)", ' ', str(sequence)).lower()

   
    sequence = unicodedata.normalize('NFKD', sequence).encode('ascii', 'ignore').decode(
        'utf-8', 'ignore'
    )

    
    sequence = re2.sub("(--+)", ' ', str(sequence)).lower()
    sequence = re2.sub("(\s+)", ' ', str(sequence)).lower()
    sequence = re2.sub("(\.\.+)", ' ', str(sequence)).lower()
    sequence = re2.sub("(~~+)", ' ', str(sequence)).lower()
    sequence = re2.sub("([cC][mM]\d+)|([cC][hH][gG]\d+)", 'CM_NUM',
                  str(sequence)).lower()
    sequence = re2.sub("(\+\++)", ' ', str(sequence)).lower()
    sequence = re2.sub("(__+)", ' ', str(sequence)).lower()
    sequence = re2.sub("(\s+.\s+)", ' ', str(sequence)).lower()
    sequence = re2.sub(r"[<>()|&©ø\[\]\'\",;?~*!]", ' ', str(sequence)).lower()
    
    sequence = re2.sub("(mailto:)", ' ', str(sequence)).lower()
    

    sequence = re2.sub(r"(\\x9\d)", ' ', str(sequence)).lower()
    sequence = re2.sub("(\-\s+)", ' ', str(sequence)).lower()
    sequence = re2.sub("(mailto:)", ' ', str(sequence)).lower()

    sequence = re2.sub("(\.\s+)", ' ', str(sequence)).lower()
    
    sequence = re2.sub("(\:\s+)", ' ', str(sequence)).lower()
    sequence = re2.sub("([iI][nN][cC]\d+)", 'INC_NUM', str(sequence)).lower()



    try:
        url = re2.search(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', str(sequence))
        repl_url = url.group(3)
        # handling url present in the data
        sequence = re2.sub(r'((https*:\/*)([^\/\s]+))(.[^\s]+)', repl_url, str(sequence))
    except Exception as e:
        pass

    
    sequence = re2.sub("(\s+.\s+)", ' ', str(sequence)).lower()
    # handling (\s+.\s+) by replacing with blank space

    return sequence

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
def trim_text_and_summary(df, maximum_text_length, maximum_summary_length):
    ctd = np1.array(df['text'])
    cleaned_summary_data = np1.array(df['headlines'])

    mini_text = []
    mini_summary = []

    for val in range(len(ctd)):
        if len(ctd[val].split()) <= maximum_text_length and len(
            cleaned_summary_data[val].split()
        ) <= maximum_summary_length:
            mini_text.append(ctd[val])
            mini_summary.append(cleaned_summary_data[val])

    df = pd.DataFrame({'text': mini_text, 'summary': mini_summary})
    return df


df = trim_text_and_summary(df, maximum_text_length, maximum_summary_length)
print(f'Dataset size: {len(df)}')
df.sample(5)

# calculating the rare words and other metrics
def rare_words_metrics(tokenizer,threshold):
    ct=0
    freq=0
    tol_count=0
    tf=0
    
    for key,val in tokenizer.word_counts.items():
        tol_count=tol_count+1
        tf=tf+val
        if (val<threshold):
            ct=ct+1
            freq=freq+val
    percentage_of_rare_words=(ct/tol_count)*100
    total_coverage_rarewords=(freq/tf)*100
    print('   rare word metrics metrics :')
    print('                       count :',ct)
    print('                 total count :',tol_count)
    print('    percentage of rare words :',percentage_of_rare_words)
    print('  total coverage of rareword :',total_coverage_rarewords)
    print('Total frequency of rare word :',tf)
    return ct,tol_count
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
def LSTM_Model_build(embeding_dimension, latent_dimension, maximum_text_length,
                                       x_vocab_size, y_vocab_size):
  

     
      encoder_input = Input(shape=(maximum_text_length, ))

      # encoder embedding layer
      encoder_embedding = Embedding(x_vocab_size,embeding_dimension,trainable=False)(encoder_input)

      # encoder_lstmlayer_1
      encoder_lstm_layer_1 = LSTM(latent_dimension,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
      encoder_output_1, state_h_1, state_c_1 = encoder_lstm_layer_1(encoder_embedding)

      # encoder_lstmlayer_2
      encoder_lstm2 = LSTM(latent_dimension,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
      encoder_output, *encoder_final_states = encoder_lstm2(encoder_output_1)


      #  Decoder

      # setting up the decoder using the encoder initial_states

      decoder_input = Input(shape=(None, ))

      # decoder embedding layer
      decoder_embeding_layer = Embedding(y_vocab_size,embeding_dimension,trainable=True)
      decoder_embedding = decoder_embeding_layer(decoder_input)

      # decoder_lstm_layer_1
      decoder_lstm_layer = LSTM(latent_dimension,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4)
      decoder_output, *decoder_final_states = decoder_lstm_layer(decoder_embedding, initial_state=encoder_final_states)

      # dense layer
      decoder_dense = TimeDistributed(Dense(y_vocab_size, activation='softmax'))
      decoder_output = decoder_dense(decoder_output)


      #  Model

      model = Model([encoder_input, decoder_input], decoder_output)
      model.summary()

      optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
      model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])

      return {'model': model,
              'inputs': {'encoder': encoder_input,
                          'decoder': decoder_input},
          'outputs': {
              'encoder': encoder_output,
              'decoder': decoder_output},
          'states': {
              'encoder': encoder_final_states,
              'decoder': decoder_final_states},
          'layers': {
              'decoder': {
                  'embedding': decoder_embeding_layer,
                  'last_decoder_lstm': decoder_lstm,
                  'dense': decoder_dense}}
              }
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
LSTM inference
def LSTM_inference(maximum_text_length, latent_dimension, encoder_input,
                                                 encoder_output,encoder_final_states,
                                                 decoder_input, decoder_output,decoder_embeding_layer,
                                                 decoder_dense, last_decoder_lstm):
    # Encode the input sequence to get the feature vector
    encoder_model = Model(inputs=encoder_input, outputs=[encoder_output] + encoder_final_states)

    # Decoder setup
    # Below variavles will save the states of the previous step
    decoder_h_state_input = Input(shape=(latent_dimension, ))
    decoder_c_state_input_ = Input(shape=(latent_dimension, ))
    decoder_hidden_input_state = Input(shape=(maximum_text_length, latent_dimension))

    # Get the embedding values of the decoding sequence
    decoder_embedding = decoder_embeding_layer(decoder_input)

   
    # setting the initial states to previous time steps to forecast the next values in the sequence
    decoder_output, *decoder_states = last_decoder_lstm(decoder_embedding,
                                                        initial_state=[decoder_h_state_input, decoder_c_state_input_])

    # A dense layer with softmax activation  to generate probobaility distribution over the target vocablary
    decoder_output = decoder_dense(decoder_output)

    # Final model for decoder_sequence
    decoder_model = Model([decoder_input] + [decoder_hidden_input_state,
                                             decoder_h_state_input, decoder_c_state_input_],
                          [decoder_output] + decoder_states)

    return (encoder_model, decoder_model)
LSTM_encoder_model, LSTM_decoder_model = LSTM_inference(maximum_text_length, latent_dimension, encoder_input, encoder_output,
                                              encoder_final_states, decoder_input, decoder_output,
                                              decoder_embeding_layer, decoder_dense, last_decoder_lstm)
LSTM_encoder_model.summary()
LSTM_decoder_model.summary()
def LSTM_decode_sequence(input_sequence, encoder_model, decoder_model):
    # Encoding the input values as state value vectors.
    Encoder_Out_, encoder_hidden, e_c = encoder_model.predict(input_sequence)

    # Generating the null target sequence of unit length
    target_seq = np1.zeros((1, 1))

    # adding the first value of target sentence with the start word.
    target_seq[0, 0] = target_word_index[start_token]

    stop_condition = False
    decoded_sequence = ''

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [Encoder_Out_, encoder_hidden, e_c])

        # Sampling  the  token
        sample_tok_idx = np1.argmax(output_tokens[0, -1, :])
        sample_token_value = reverse_target_word_index[sample_tok_idx]

        if sample_token_value != end_token:
            decoded_sequence += ' ' + sample_token_value

        # condition for exiting while either it hits maximum lenght or find the stop word
        if (sample_token_value == end_token) or (len(decoded_sequence.split()) >= (maximum_summary_length - 1)):
            stop_condition = True

        # Update the unit value target sequence with sampled token index
        target_seq = np1.zeros((1, 1))
        target_seq[0, 0] = sample_tok_idx

        # Update hidden and cell states
        encoder_hidden, e_c = h, c

    return decoded_sequence

def sequence_to_summary(input_sequence):
    new_string = ''
    for val in input_sequence:
        if (
            (val != 0 and val != target_word_index[start_token]) and
            (val != target_word_index[end_token])
        ):
            new_string = new_string + reverse_target_word_index[val] + ' '
    return new_string
def sequence_to_text(input_sequence):
    new_string = ''
    for val in input_sequence:
        if val != 0:
            new_string = new_string + reverse_source_word_index[val] + ' '
    return new_string
# Testing on validation data
for val in range(0, 2):
    print(f"# {val+1} News: ", sequence_to_text(x_val_padded[val]))
    print("Original_Summary: ", sequence_to_summary(y_val_padded[val]))
    print("Predicted_Summary: ", LSTM_decode_sequence(x_val_padded[val].reshape(1, maximum_text_length), LSTM_encoder_model,
                                                      LSTM_decoder_model))
    print()
# Bidirectional LSTM
def Bidirectional_LSTM_build(embeding_dimension, latent_dimension, maximum_text_length, x_vocab_size, y_vocab_size):
    

      #  Encoder
      encoder_input = Input(shape=(maximum_text_length, ))

      # encoder embedding layer
      encoder_embedding = Embedding(x_vocab_size,embeding_dimension,trainable=False,name='encoder_embedding')(encoder_input)

#        Bidirectional encoder_lstm_1
      encoder_bidirectional_LSTM_1 = Bidirectional(LSTM(latent_dimension,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4,name='encoder_lstm_1'),name='encoder_bidirectional_lstm_1')
      encoder_output_1, forward_h_1, forward_c_1, backward_h_1, backward_c_1 = encoder_bidirectional_LSTM_1(encoder_embedding)
      encoder_bi_lstm1_output = [encoder_output_1, forward_h_1, forward_c_1, backward_h_1, backward_c_1]

      # Bidirectional encoder lstm_2
      encoder_bidirectional_lstm2 = Bidirectional(LSTM(latent_dimension,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4,name='encoder_lstm_2'),name='encoder_bidirectional_lstm_2')
      encoder_output_2, forward_h_2, forward_c_2, backward_h_2, backward_c_2 = encoder_bidirectional_lstm2(encoder_output_1)
      encoder_bi_lstm2_output = [encoder_output_2, forward_h_2, forward_c_2, backward_h_2, backward_c_2]

      # Bidirectional encoder lstm_3
      encoder_bi_lstm = Bidirectional(LSTM(latent_dimension,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.4,name='encoder_lstm_3'),name='encoder_bidirectional_lstm_3')
      encoder_output, *encoder_final_states = encoder_bi_lstm(encoder_output_2)


      # Decoder


      # Set up the decoder, using `encoder_states` as initial state.

      decoder_input = Input(shape=(None, ))

      # decoder embedding layer
      decoder_embeding_layer = Embedding(y_vocab_size,embeding_dimension,trainable=False,name='decoder_embedding')
      decoder_embedding = decoder_embeding_layer(decoder_input)

      decoder_bi_lstm = Bidirectional(LSTM(latent_dimension,return_sequences=True,return_state=True,dropout=0.4,recurrent_dropout=0.2,name='decoder_lstm_1'),name='decoder_bidirectional_lstm_1')
      decoder_output, *decoder_final_states = decoder_bi_lstm(decoder_embedding, initial_state=encoder_final_states
          # decoder_embedding, initial_state=encoder_final_states[:2]
      )  # taking only the forward states

      # dense and time distributed layer
      decoder_dense = TimeDistributed(Dense(y_vocab_size, activation='softmax'))
      decoder_output = decoder_dense(decoder_output)


      #  Model Built
      model = Model([encoder_input, decoder_input], decoder_output, name='seq2seq_model_with_bidirectional_lstm')
      model.summary()

      optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
      model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])

      return {
          'model': model,
          'inputs': {
              'encoder': encoder_input,
              'decoder': decoder_input
          },
          'outputs': {
              'encoder': encoder_output,
              'decoder': decoder_output
          },
          'states': {
              'encoder': encoder_final_states,
              'decoder': decoder_final_states
          },
          'layers': {
              'decoder': {
                  'embedding': decoder_embeding_layer,
                  'last_decoder_lstm': decoder_bi_lstm,
                  'dense': decoder_dense }}
          }
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
def Bidirectional_LSTM_inference(
    maximum_text_length, latent_dimension, encoder_input, encoder_output,
    encoder_final_states, decoder_input, decoder_output,
    decoder_embeding_layer, decoder_dense, last_decoder_bi_lstm):

    # input sequence encoding to get the feature_vectors
    encoder_model = Model(
        inputs=encoder_input, outputs=[encoder_output] + encoder_final_states
    )

    # Decoder_setup
    # these values store the previous time step states
    decoder_state_forward_input_h = Input(shape=(latent_dimension, ))
    decoder_state_forward_input_c = Input(shape=(latent_dimension, ))
    decoder_state_backward_input_h = Input(shape=(latent_dimension, ))
    decoder_state_backward_input_c = Input(shape=(latent_dimension, ))

    # Create the hidden_input layer two times  the latent dimension,
    # As we are using Bidirectional LSTM we will get two of the hidden and cell states
    decoder_hidden_input_state = Input(shape=(maximum_text_length, latent_dimension * 2))

    decoder_initial_state = [decoder_state_forward_input_h,
                             decoder_state_forward_input_c,
                             decoder_state_backward_input_h,
                             decoder_state_backward_input_c]

    # Get the embedding values of the decoder sequence
    decoder_embedding = decoder_embeding_layer(decoder_input)

   
    # setting the initial states from the previous step to predict the next sequence
    decoder_output, *decoder_states = last_decoder_bi_lstm(decoder_embedding, initial_state=decoder_initial_state)

    # A dense layer with softmax activation function to generate probability distribution over the targeted vocabulary
    decoder_output = decoder_dense(decoder_output)

    # Final decoder model
    decoder_model = Model([decoder_input] + [decoder_hidden_input_state] + decoder_initial_state,
                          [decoder_output] + decoder_states)

    return (encoder_model, decoder_model)
# Inference
BiLSTM_encoder_model, BiLSTM_decoder_model = Bidirectional_LSTM_inference(maximum_text_length, latent_dimension, encoder_input, encoder_output,
                                              encoder_final_states, decoder_input, decoder_output,
                                              decoder_embeding_layer, decoder_dense, last_decoder_lstm)
print(BiLSTM_encoder_model.summary())
print(BiLSTM_decoder_model.summary())
def Bidirectional_LSTM_decode(input_sequence, encoder_model, decoder_model):
    # Encoding the input_values as state vectors.
    Encoder_Out_, *state_values = encoder_model.predict(input_sequence)
    
    # generating empty target sequence of unit values.
    target_seq = np1.zeros((1, 1))

    # filling the first word of the target sequence with start_word.
    target_seq[0, 0] = target_word_index[start_token]

    stop_condition = False
    decoded_sequence = ''
    
    while not stop_condition:
        output_tokens, *decoder_states = decoder_model.predict(
            [target_seq] + [Encoder_Out_] + state_values
        )

        # Sampling a token
        sample_tok_idx = np1.argmax(output_tokens[0, -1, :]) # Greedy Search
        sample_token_value = reverse_target_word_index[sample_tok_idx + 1]
        
        if sample_token_value != end_token:
            decoded_sequence += ' ' + sample_token_value

        
        if (sample_token_value == end_token) or (len(decoded_sequence.split()) >= (maximum_summary_length - 1)):
            stop_condition = True

        # Updating the target sequence of UNIT value.
        target_seq = np1.zeros((1, 1))
        target_seq[0, 0] = sample_tok_idx

        state_values = decoder_states

    return decoded_sequence
# Testing on validation data
for val in range(0, 3):
    print(f"# {val+1} Text: ", sequence_to_text(x_val_padded[val]))
    print("Original_summary: ", sequence_to_summary(y_val_padded[val]))
    print("Predicted_summary: ", Bidirectional_LSTM_decode(x_val_padded[val].reshape(1, maximum_text_length),
                                                           BiLSTM_encoder_model,BiLSTM_decoder_model))
    print()
# Transformers
!pip install transformers==2.2.0
!pip install bert-extractive-summarizer==0.7.1
!pip install rouge
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

    original_summary=sequence_to_summary(y_val_padded[i])
    ORIGINAL_summary.append(original_summary)
    print('#    original summary : ',original_summary)

    gpt_2_summary=''.join(GPT2_model(sequence_to_text(x_val_padded[i]), min_length=min_length_text))
    GPT2_prediction.append(gpt_2_summary)
    print('#    GPT-2 summary : ',gpt_2_summary)

    BERT_summary=''.join(bert_model(sequence_to_text(x_val_padded[i]), min_length=min_length_text))
    BERT_prediction.append(BERT_summary)
    print('#     BERT summary : ',BERT_summary)

    LSTM_summary=LSTM_decode_sequence(x_val_padded[i].reshape(1, maximum_text_length),LSTM_encoder_model,LSTM_decoder_model)
    LSTM_perdiction.append(LSTM_summary)
    print('#     LSTM summary : ',LSTM_summary)

    BiLSTM_summary=Bidirectional_LSTM_decode(x_val_padded[i].reshape(1,maximum_text_length),BiLSTM_encoder_model,BiLSTM_decoder_model)
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
