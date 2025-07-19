from tensorflow.keras import Input, Model
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Embedding, TimeDistributed
import numpy as np1
import tensorflow as tf
from src.utils import extract_yml


class Lstm_class:

    def __init__(self):
        pass

    def LSTM_Model_build(self, embeding_dimension, latent_dimension, maximum_text_length,
                         x_vocab_size, y_vocab_size):

        encoder_input = Input(shape=(maximum_text_length,))

        # encoder embedding layer
        encoder_embedding = Embedding(x_vocab_size, embeding_dimension, trainable=False)(encoder_input)

        # encoder_lstmlayer_1
        encoder_lstm_layer_1 = LSTM(latent_dimension, return_sequences=True, return_state=True, dropout=0.4,
                                    recurrent_dropout=0.4)
        encoder_output_1, state_h_1, state_c_1 = encoder_lstm_layer_1(encoder_embedding)

        # encoder_lstmlayer_2
        encoder_lstm2 = LSTM(latent_dimension, return_sequences=True, return_state=True, dropout=0.4,
                             recurrent_dropout=0.4)
        encoder_output, *encoder_final_states = encoder_lstm2(encoder_output_1)

        #  Decoder
        # setting up the decoder using the encoder initial_states
        decoder_input = Input(shape=(None,))

        # decoder embedding layer
        decoder_embeding_layer = Embedding(y_vocab_size, embeding_dimension, trainable=True)
        decoder_embedding = decoder_embeding_layer(decoder_input)

        # decoder_lstm_layer_1
        decoder_lstm_layer = LSTM(latent_dimension, return_sequences=True, return_state=True, dropout=0.4,
                                  recurrent_dropout=0.4)
        decoder_output, *decoder_final_states = decoder_lstm_layer(decoder_embedding,
                                                                   initial_state=encoder_final_states)

        # dense layer
        decoder_dense = TimeDistributed(Dense(y_vocab_size, activation='softmax'))
        decoder_output = decoder_dense(decoder_output)

        #  Model
        model = Model([encoder_input, decoder_input], decoder_output)
        model.summary()

        optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

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
                        'last_decoder_lstm': decoder_lstm_layer,
                        'dense': decoder_dense}}
                }

    def LSTM_decode_sequence(self,
                             input_sequence,
                             encoder_model,
                             decoder_model,
                             reverse_target_word_index,
                             target_word_index,
                             maximum_summary_length=None
                             ):
        # Encoding the input values as state value vectors.
        Encoder_Out_, encoder_hidden, e_c = encoder_model.predict(input_sequence)

        # Generating the null target sequence of unit length
        target_seq = np1.zeros((1, 1))

        # adding the first value of target sentence with the start word.
        target_seq[0, 0] = target_word_index[extract_yml('Start_Token')]

        stop_condition = False
        decoded_sequence = ''

        while not stop_condition:
            output_tokens, h, c = decoder_model.predict([target_seq] + [Encoder_Out_, encoder_hidden, e_c])

            # Sampling  the  token
            sample_tok_idx = np1.argmax(output_tokens[0, -1, :])
            sample_token_value = reverse_target_word_index[sample_tok_idx]

            if sample_token_value != extract_yml('End_Token'):
                decoded_sequence += ' ' + sample_token_value

            # condition for exiting while either it hits maximum length or find the stop word
            if (sample_token_value == extract_yml('End_Token')) or (
                    len(decoded_sequence.split()) >= (maximum_summary_length - 1)):
                stop_condition = True

            # Update the unit value target sequence with sampled token index
            target_seq = np1.zeros((1, 1))
            target_seq[0, 0] = sample_tok_idx

            # Update hidden and cell states
            encoder_hidden, e_c = h, c

        return decoded_sequence

    # LSTM inference
    def LSTM_inference(self,
                       maximum_text_length,
                       latent_dimension,
                       encoder_input,
                       encoder_output,
                       encoder_final_states,
                       decoder_input,
                       decoder_output,
                       decoder_embeding_layer,
                       decoder_dense,
                       last_decoder_lstm
                       ):
        # Encode the input sequence to get the feature vector
        encoder_model = Model(inputs=encoder_input, outputs=[encoder_output] + encoder_final_states)

        # Decoder setup
        # Below variavles will save the states of the previous step
        decoder_h_state_input = Input(shape=(latent_dimension,))
        decoder_c_state_input_ = Input(shape=(latent_dimension,))
        decoder_hidden_input_state = Input(shape=(maximum_text_length, latent_dimension))

        # Get the embedding values of the decoding sequence
        decoder_embedding = decoder_embeding_layer(decoder_input)

        # setting the initial states to previous time steps to forecast the next values in the sequence
        decoder_output, *decoder_states = last_decoder_lstm(decoder_embedding,
                                                            initial_state=[decoder_h_state_input,
                                                                           decoder_c_state_input_])

        # A dense layer with softmax activation  to generate probobaility distribution over the target vocablary
        decoder_output = decoder_dense(decoder_output)

        # Final model for decoder_sequence
        decoder_model = Model([decoder_input] + [decoder_hidden_input_state,decoder_h_state_input, decoder_c_state_input_],
                              [decoder_output] + decoder_states)

        return (encoder_model, decoder_model)


class BidiLSTM:
    def __init__(self):
        pass
    # Bidirectional LSTM
    def Bidirectional_LSTM_build(self, embeding_dimension, latent_dimension, maximum_text_length, x_vocab_size,
                                 y_vocab_size):
        #  Encoder
        encoder_input = Input(shape=(maximum_text_length,))

        # encoder embedding layer
        encoder_embedding = Embedding(x_vocab_size, embeding_dimension, trainable=False, name='encoder_embedding')(
            encoder_input)

        #        Bidirectional encoder_lstm_1
        encoder_bidirectional_LSTM_1 = Bidirectional(
            LSTM(latent_dimension, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4,
                 name='encoder_lstm_1'), name='encoder_bidirectional_lstm_1')
        encoder_output_1, forward_h_1, forward_c_1, backward_h_1, backward_c_1 = encoder_bidirectional_LSTM_1(
            encoder_embedding)
        encoder_bi_lstm1_output = [encoder_output_1, forward_h_1, forward_c_1, backward_h_1, backward_c_1]

        # Bidirectional encoder lstm_2
        encoder_bidirectional_lstm2 = Bidirectional(
            LSTM(latent_dimension, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4,
                 name='encoder_lstm_2'), name='encoder_bidirectional_lstm_2')
        encoder_output_2, forward_h_2, forward_c_2, backward_h_2, backward_c_2 = encoder_bidirectional_lstm2(
            encoder_output_1)
        encoder_bi_lstm2_output = [encoder_output_2, forward_h_2, forward_c_2, backward_h_2, backward_c_2]

        # Bidirectional encoder lstm_3
        encoder_bi_lstm = Bidirectional(
            LSTM(latent_dimension, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.4,
                 name='encoder_lstm_3'), name='encoder_bidirectional_lstm_3')
        encoder_output, *encoder_final_states = encoder_bi_lstm(encoder_output_2)

        # Decoder
        # Set up the decoder, using `encoder_states` as initial state.
        decoder_input = Input(shape=(None,))

        # decoder embedding layer
        decoder_embeding_layer = Embedding(y_vocab_size, embeding_dimension, trainable=False, name='decoder_embedding')
        decoder_embedding = decoder_embeding_layer(decoder_input)

        decoder_bi_lstm = Bidirectional(
            LSTM(latent_dimension, return_sequences=True, return_state=True, dropout=0.4, recurrent_dropout=0.2,
                 name='decoder_lstm_1'), name='decoder_bidirectional_lstm_1')
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
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

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
                    'dense': decoder_dense}}
        }

    def Bidirectional_LSTM_inference(self,
                                     maximum_text_length,
                                     latent_dimension,
                                     encoder_input,
                                     encoder_output,
                                     encoder_final_states,
                                     decoder_input,
                                     decoder_output,
                                     decoder_embeding_layer,
                                     decoder_dense,
                                     last_decoder_bi_lstm):

        # input sequence encoding to get the feature_vectors
        encoder_model = Model(
            inputs=encoder_input, outputs=[encoder_output] + encoder_final_states
        )

        # Decoder_setup
        # these values store the previous time step states
        decoder_state_forward_input_h = Input(shape=(latent_dimension,))
        decoder_state_forward_input_c = Input(shape=(latent_dimension,))
        decoder_state_backward_input_h = Input(shape=(latent_dimension,))
        decoder_state_backward_input_c = Input(shape=(latent_dimension,))

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

    def Bidirectional_LSTM_decode(self,
                                  input_sequence,
                                  encoder_model,
                                  decoder_model,
                                  reverse_target_word_index,
                                  target_word_index,
                                  maximum_summary_length):
        # Encoding the input_values as state vectors.
        Encoder_Out_, *state_values = encoder_model.predict(input_sequence)

        # generating empty target sequence of unit values.
        target_seq = np1.zeros((1, 1))

        # filling the first word of the target sequence with start_word.
        target_seq[0, 0] = target_word_index[extract_yml('Start_Token')]

        stop_condition = False
        decoded_sequence = ''

        while not stop_condition:
            output_tokens, *decoder_states = decoder_model.predict(
                [target_seq] + [Encoder_Out_] + state_values
            )

            # Sampling a token
            sample_tok_idx = np1.argmax(output_tokens[0, -1, :])  # Greedy Search
            sample_token_value = reverse_target_word_index[sample_tok_idx + 1]

            if sample_token_value != extract_yml('End_Token'):
                decoded_sequence += ' ' + sample_token_value

            if (sample_token_value == extract_yml('End_Token')) or (
                    len(decoded_sequence.split()) >= (maximum_summary_length - 1)):
                stop_condition = True

            # Updating the target sequence of UNIT value.
            target_seq = np1.zeros((1, 1))
            target_seq[0, 0] = sample_tok_idx

            state_values = decoder_states

        return decoded_sequence
