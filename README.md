### A Comparative Study of LSTMs and BiLSTMs using Text Summarization


<!-- Abstract -->
The quantity of text data that is now available on the internet and in other digital formats
continues to expand, and as a result, people are becoming increasingly overwhelmed.
Summarising a text is an essential task in natural language processing (NLP) that tries to
produce a shorter version of a text while maintaining the text's most relevant information.
It is possible to use it to generate summaries of a variety of different types of content,
including news items, scientific papers, legal documents, and more.
Deep learning approaches such as LSTMs, Bidirectional LSTMs, and Transformers are
used to solve the aforementioned problem. These models are trained on enormous
volumes of text data and can comprehend the meaning and context of the text, allowing
them to provide more accurate and relevant summaries. Pre-trained models such as BERT
and GPT-2 have recently been fine-tuned to obtain cutting-edge performance on text
summarization tasks. These models were trained on enormous quantities of text data and
may be fine-tuned on a specific job using a smaller dataset, making them more efficient
and effective. The above models have all been built, and comparisons have been made
between them using the appropriate metrics and graphical representations. It is made very
clear that solutions would be provided for the difficulties that arose throughout the
implementation. The pretrained Transformer model simulations have generated
impressive outcomes, with BERT's performance coming out on top when compared to the
other transformer.
Keywords: NLP, Summarization, LSTM, Bidirectional, Transformers, BERT, GPT-2,
validation, encoder-decoder.
<!-- Table of Contents -->
1.1 Background
    1.2 Objective
    1.3 Structure of report 
2.1 Introduction:
    2.2 Neural Networks
    2.3 Backpropagation:
    2.4 Recurrent Neural Networks
    2.5 Long Short Term Memory
2.5 Attention
2.6 Transformers
3.1 Overview
3.2 Dataset
3.3 Proposed Models
3.4 Metrics
4.1 Data Exploration and Preparation 
4.1.1 Data Exploration: 
4.1.2 Data preprocessing
4.1.3 Adding tokens at beginning and end of summary: 32
4.1.4 Calculating Maximum Text and Summary Length 32
4.1.5 Rare Word analysis 33
4.2 Model Design 33
4.2.2 LSTM 34
4.2.2 Transformer 36
4.2.2.1 BERT 36
4.2.2.2 GPT-2 37
5.1 Training Metrics: 38
5.1.1 LSTM: 38
5.1.2 Bidirectional LSTM 39
5.2 Testing results: 41
5.2.1 Rouge-1: 41
5.2.1.1 Precision: 42
5.2.1.2 Recall: 42
5.2.1.3 F-1 Score: 43
5.2.2 Rouge-2 43
5.2.2.1 Precision: 43
5.2.2.2 Recall:. 44
5.2.2.3 F-1 Score: 44
5.2.3 Rouge - L: 45
5.2.3.1 Precision: 45
5.2.3.2 Recall: 45
5.2.3.3 F-Score: 46
6 Critical 49
7 Conclusion 50
<!-- Table of Figures -->
1 . Neural Network 13
2. Neural Network Backpropagation 15
3. Recurrent Neural Network 16
4. Long Short Term Memory 18
5. Bidirectional Long Short Term Memory 19
6. Attention 20
7. Transformer Stack 23
8. Attention at Encoder level 24
List of Abbreviations
NLP Natural Language Processing
LSTM Long Short Term Memory
BiLSTM Bidirectional Long Short Term Memory
BERT Bidirectional Encoder Representation from Transformers
ML Machine Learning
GPT-2 Generative Pre-trained Transformer 2
ROUGE Recall-Oriented Understudy for Gisting Evaluation
RNN Recurrent Neural Networks
<!-- 1 Introduction -->
<!-- 1.1 Background -->
People are overwhelmed by the plethora of information and papers available as the
Internet has expanded exponentially. Due to the increased availability of documents,
extensive research in the field of text summarization has been required (Allahyari et al.,
2017). Text summarization is the method of producing a synopsis of a certain text that
incorporates the most important data from a source document; the objective is to obtain a
summary of the document's key points (Abualigah et al., 2019). It is the process of
producing a reduced form of a text document that preserves important data as well as the
overall meaning of the original text. Automatic text summarization is becoming an
essential method for quickly and easily discovering useful information across enormous
amounts of text (Andhale and Bewoor, 2016). In the 1950s, research began in this field,
but we still lack effective techniques that can create summaries like humans or experts
(Meena and Gopalani, 2015). Text summarization is categorised into extractive and
abstractive, while extractive summarization involves selecting a subset of the sentences
found in the original source, abstractive summarization is rephrasing of the information
in the text(Erkan and Radev, 2004). Most of the summarization researched today is
extractive summarization as a computer cannot work on par with a human and abstractive
summarization has problems with semantic representations and Natural Language
generation which are very difficult concepts for a computer to understand (Witbrock and
Mittal, 1999). Abstractive summarization which has been used uses extractive
summarization for the preprocessing work.The earlier research on extractive
summarization was based on straightforward heuristic characteristics of the sentences,
such as their placement in the text, the overall frequency of the words they include, or
certain key phrases indicating the significance of the sentences. For instance, their
position in the document could be used to determine the sentences that are most essential
(Luhn, 1958).
There were different techniques introduced and experimented on text processing
out of which the most common was by using the frequency of the words in a document.
Here, we may make use of the concepts of Term Frequency and Inverse Document
Frequency, where Word Frequency refers to the number of times a certain term appears in
the text and where Term Frequency is determined for each term in the document
(SALTON and YANG, 1973). Inverse Document Frequency involves giving importance
to the terms with low Document Frequency (SPARCK JONES, 1972).
1.2 Objective
Natural Language Processing plays an important role in our day to day activities. is also
used in many industries such as healthcare, finance, and e-commerce, to analyse large
amounts of unstructured data and extract insights that can be used to improve
decision-making and boost efficiency. LSTMs were considered to be the best
architectures when dealing with text data, especially long sequences. The introduction of
Transformers made significant strides in creation of state of the art solutions for text
related problems. Transformers are quite effective in tasks related to text classification,
entity recognition and summarization. Out of all these mentioned tasks text
summarization is the most challenging where a deep learning model has to understand a
full paragraph and write a summary retaining the important information.
The Objective of this project is to show comparison of LSTMs and Transformers in
handling text summarization effectively.
1.3 Structure of report
The structure of the project is as follows:
The total project is structured into seven chapters, The first chapter gives the background
of the project and the objective. The second chapter is Literature review which explains
the technologies involved in text processing and advancements in it.
The Third chapter is Methodology which explains the steps involved in this approach,
including dataset taken, Models that are proposed and the metrics used to evaluate them.
Fourth chapter contains the data exploration, preparation and model building part with
evaluation.
Fifth chapter explains the results achieved in the evaluation of model and graphical
analysis of metrics.
Sixth chapter explains the whole project in short and analyses the results obtained in a
short summary
The final chapter gives conclusion to the project with a future scope where the project
can perform further developments to produce solutions addressed in the implementation.
<!-- 2 Literature Review -->
2.1 Introduction:
NLP is a subfield of Artificial Intelligence that enables robots to read, comprehend, and
interpret human languages. In development studies, the capacity of natural language
processing (NLP) to automatically analyse and interpret human communication is gaining
popularity. (Kang et al., 2020). It's a field that focuses on the interface of data science
with natural language, and it's growing in popularity across a variety of industries
(Zeroual and Lakhouaja, 2018).Today, NLP is prospering as a result of huge
breakthroughs in data availability and computing capability, enabling practitioners to
achieve meaningful outcomes in industries including healthcare, media, finance, and
human resources, among others. (Kalyanathaya, D and P, 2019). In simple words, NLP
refers to the automatic processing of natural human languages such as speech or text, and
while the idea is appealing, the actual value of this technology is found in the
implementation (Khan et al., 2016). NLP allows for the organisation and structuring of
knowledge so that it may be used for a variety of activities, including automatic
summarization, translation, named entity recognition, relationship extraction, semantic
analysis, sentiment analysis, audio recognition, and topic segmentation (Tarwani, 2017).
Use cases:
NLP has a lot of applications, and they tend to increase daily, some of them are
mentioned below
· By discovering and extracting information from sources such as social media,
companies can learn what customers are saying about a service or product. This
sentiment analysis can reveal a lot about users ’ preferences and the factors that affect
their actions and also the mental state of a person (Liu, Shin and Burns, 2019).
· Alexa from Amazon and Siri from Apple are examples of intelligent voice-driven
interfaces that use natural language processing to respond to vocal commands and
perform tasks such as locating a specific store, providing the weather forecast,
recommending the best route to the office, and turning on the lights at home (Hoy, 2018)
(Zhou, Wang and Li, 2017).
· Further, NLP is being applied both in the search and selection stages of talent hiring,
identifying the skills of prospective recruits and spotting prospects before they become
job seekers (Mujtaba and Mahapatra, 2020).
· NLP is rising in popularity, particularly in the health sector. While healthcare
companies are increasingly adopting electronic health records, this technology is helping
care delivery, disease identification, and lowering costs . The ability to improve patient
records enables patients to be better understood and benefit from improved healthcare.
The aim is to enhance their experience, which some companies are currently doing. The
use of natural language processing facilitates the analysis and detection of illnesses using
electronic health records and the patient's voice. This skill is being investigated in a
variety of health issues, including cardiovascular disease, depression, and possibly
schizophrenia (Koleck et al., 2019).
Let us go through the phases involved in NLP.
· Lexical Analysis
This comprises detecting and studying the structural relationships between words, in
terms of language processing lexical analysis is connecting the words with corresponding
words in the dictionary. The term "lexicon" applies to a language's collection of words
and phrases. Numerous uses may be made of lexical analysis. One of them aids in
predicting the originally unknown purpose of grammatical terms. As a result, Lexicon
Normalisation is essential (Daud, 2010).
· Syntactic Analysis
Syntactic analysis is used to search for grammatical mistakes, word formation, and word
connections. Finding a sentence's grammatical structure—that is, how the words are
arranged into constituents like noun phrases and verb phrases—is the goal of syntactic
analysis. The task of assigning a part of speech to each word as part of a sentence is
syntactic analysis. Syntactic parsing is the way of evaluating the syntax of the words in a
phrase. Dependency Grammar and Part of Speech (POS) tags are essential syntactic
characteristics of the text (Brill and Mooney, 1997).
· Semantic Analysis
Provides clear and semantically valid possible interpretations of a statement. It involves
extracting the meaning of the text by identifying the important entities and representing a
relationship among them (Mowery et al., 2015).
· Pragmatic Analysis
It is the study of the meanings of words in a particular language. Extract information
from the text. The assessment of the context and intent of an utterance is at the forefront
of pragmatic analysis, which frequently places an emphasis on narrative-level analysis. It
comprises verbal recurrence, who said what to whom, and so on. It incorporates how
individuals interact with one another, the context in which they speak, and several other
factors (Li, Thomas and Liu, 2020).
2.2 Neural Networks
The rising complexity of data necessitates the employment of mathematical models
capable of capturing the important essential attributes, so far machine learning has been
successful in doing so. Handling big data to discover the hidden patterns from various
fields like banking, health care, social media, health care, etc, have been a tedious job as
the complexity of data varies (Athmaja, Hanumanthappa and Kavitha, 2017).
In the fields of pattern recognition and data analysis, neural networks also known as feed
forward neural networks provide a wide variety of strong new solutions to the issues that
need to be solved. They are artificial adaptive systems that are inspired by the way the
human brain works. They are self-changing systems that can change how they work in
response to a goal (Bishop, 1994). They are especially good at dealing with complicated
problems because they can figure out the best way to solve them based on fuzzy rules..
Synapses on the dendrites or membrane of natural neurons receive messages. When the
received impulses are sufficiently powerful (beyond a particular threshold), the neuron is
engaged and produces a signal via the axon. This signal may be sent to another synapse,
activating further neurons. When making models of artificial neurons, the complexity of
real neurons is simplified a lot. These are made up of inputs (like synapses) multiplied by
weights (the strength of each signal) or synaptic weights, and then a mathematical
function is used to figure out the activity of the neuron.(Ramadhan, 2021).
Above figure illustrates the working of the feedforward neural network
ANNs process data by combining artificial neurons. Neural networks are simply the
clustering of rudimentary artificial neurons. Clustering happens because of the formation
of layers, which are subsequently linked (Jure Zupan, 1994).
A simple neural network looks like this:
Input -> [x 0 , x 1 , x 2 , x 3 … x n] -> [a 0 , a 1 , a 2 , ...a n ] -> k(x)
Layer1 layer 2 layer3 (output layer)
A conventional feed-forward neural network will have three layers: an input layer, a
hidden layer, and an output layer. A network is said to be completely linked when each
neuron in one layer is connected to every neuron in the next layer (Murat H., 2006).
2.3 Backpropagation:
Typically, neural networks acquire knowledge through a process known as backward
propagation (Sometimes called "backpropagation"). This includes comparing the output
of a network to the output it was designed to produce and using the difference to adjust
the weights of the connections between the network's units (Yann Le Cun, G Hinton and
T Sejnowski, 1988). To find the optimised values the network uses gradient descent. Use
of gradient descent has been proved to give best results in back propagation in
classification problems in (Singh, Verma and Thoke, 2015). The propagation taking place
in the networks occurs multiple times and over time, backpropagation trains the network,
decreasing the distance between real and desired output to the point where they exactly
coincide, ensuring that the network always finds things out correctly. The training begins
with some random values as weights and the idea is to adjust them in such a way that the
error is minimum.
Error B = Actual Output – Desired Output
Above figure shows the back propagation of the neural network
Natural language processing is rooted in linguistics, computer science, and mathematics
and has become a prominent area of artificial intelligence studies. Despite this, it has
several problems related to the fact that language is difficult to grasp and use to our
needs. Neural Networks struggle to process more data. NNs use numbers, not texts. For
every NLP problem in an ANN, we must transform each word to a vector, which needs
many neurons if we employ encoding. So much computational power will be used. In
translation, word order is vital; otherwise, the sentence's meaning is lost. ANNs can't
retain sequences. As a result neural networks would be less effective in Language
process.
2.4 Recurrent Neural Networks
Recurrent neural networks (RNN), the cutting-edge method for processing sequential
data that is used by both Apple's Siri and Google's voice search, are now the industry
standard. A network that is capable of expressing dynamic activity is produced when
recurrent neural networks are constructed with individual units that create a directed
cycle with one another. This cycle connects the individual units. RNNs have their own
memory, which enables them to process sequential data in a more efficient manner than
other types of neural networks.
Despite the fact that they were first developed in 1980, they have recently seen a surge in
popularity as a result of advancements in computational power and the fact that they can
be used to handle sequential data as they possess memory units that can store the
information which is captured by the neuron. Both of these factors have contributed to
the rise in popularity that they have seen in recent years (M. Tarwani and Edem, 2017).
There are several applications that have made use of RNNs, such as speech recognition,
the identification of visual objects, language models, and others; however, the application
that makes the most common use of RNNs is language-based model building. Because of
the exceptional computational power that they possess, they have been put to use in the
field of language translation, where they have achieved a great deal of success. These
networks are classified as Deep Neural Networks, and the feedback memory that is
utilised within them gives exceptional accuracies. Both of these factors contribute to their
success in the field (Hermanto, Adji and Setiawan, 2015).
The RNN architecture with feedback networks is shown above
Despite the fact that RNNs are often regarded as one of the best algorithms for sequential
data, they are plagued by the issue of vanishing gradient. As soon as backpropagation is
started, repetitive convolution of weights takes place, which ultimately leads to the initial
gradient values being ineffectively low due to their low magnitude (Kolbusz, Rozycki
and Wilamowski, 2017). Because of this, there is a difficulty with long-term
dependencies, since the model is unable to remember the knowledge that it gained via
forward and backward propagations (Bengio, Simard and Frasconi, 1994).
In order to circumvent the issue of vanishing gradient, many new varieties of RNNs were
developed, the most promising of which being Long Short term Memory Recurrent
Neural Networks (Xiao and Zhou, 2020).
2.5 Long Short Term Memory
To overcome the problem of vanishing gradient Gated Units such as Long Short-Term
Memory and Gated Recurrent Units were introduced out of which the LSTM were the
best choice for speech, video and Natural Language Processing (Rehmer and Kroll,
2020). LSTM are designed to overcome the error feedback involved in the back
propagation of neural networks. Generally LSTM networks consist of a cell, input gate,
output gate and a forget gate.The cell can remember values for an undetermined amount
of time, and its three gates control how information flows into and out of the cell.
Naturally, it is the responsibility of the cell to monitor and record the dependencies that
exist between the components of the input sequence.The input gate controls how much a
new value flows into the cell. The forget gate controls how long a value stays in the cell.
The output gate controls how much the value in the cell is used to calculate the output
initiation of the LSTM unit. Every one of these gates is controlled by the other gates
(Chakraborty et al., 2020).
The working of LSTM network is shown above
The standard RNN has many problems, such as gradient vanishing, exploding, long-term
dependencies, parallelization, and computational limitations. LSTM has solved the
problem of gradient vanishing, but parallelization and handling long-term dependencies
are still problems (Etemad, Abidi and Chhabra, 2021). Different models were added to
RNNs, including the Encoder-Decoder-based LSTM, to solve the problem. This network
has an LSTM encoder and an LSTM decoder. The encoder is used to learn nonlinear
patterns, and the decoder is used to predict future series. Adding attention to the network
helped the model adaptively align the encoded and decoded process(Yuan et al., 2020).
The above Figure represents the working of Bidirectional LSTM network
The bidirectional LSTM is responsible for establishing a unidirectional extra LSTM on
top of the standard unidirectional LSTM layer. This ensures that the connection between
the hidden layers happens in the opposite manner. The LSTM is trained using data from
both the input and the output sides, and as a result, the network is able to extract
information from the past as well as the future using these output vectors. The attention
mechanism aids in the BiLSTM network's ability to remember crucial information on a
word-by-word basis within the text input that is sent to it (Graves and Schmidhuber,
2005). In the end vector values from both input and output are compared.
2.5 Attention
While processing text, one of the challenges that must be overcome is that a neural
network must be able to condense all of the relevant information of a source phrase into a
vector of a specific length. This difficulty is especially difficult when using the
encoder-decoder technique. Because of this, it may be challenging for the neural network
to deal with lengthy phrases, particularly ones that are more extensive than the sentences
contained in the training corpus (Bahdanau, Cho and Bengio, 2015). Attention was the
key to solve such issues. Attention is especially helpful when the input is lengthy or
contains a great deal of information. One example of this is the process of text
summarization, in which the output is a reduced version of a text sequence that might be
rather extensive. Growing the relevance of input pieces through the use of machine
learning is yet another strategy for increasing popularity. In this manner, neural networks
will be able to automatically determine the weight of the significance of each given
section of the input and take this significance into consideration while carrying out the
primary job. The process known as attention is the most prevalent approach that people
take to solve this issue (Galassi, Lippi and Torroni, 2020). Attention with RNNs have
been applied into various other fields such as Image caption generation using visual
attention (Xu et al., 2015), generating complex sequences which are useful in
handwriting synthesis (Graves, 2013), visual attention for Visual Object Classification
(Mnih et al., 2014) and attention based Speech Recognition (Chorowski and Bahdanau,
2015).
Attention mechanism involved in transformer architecture is illustrated above
The sequence-to-sequence model, which has an encoder-decoder architecture, was the
first attention model that was put into action (Cho et al., 2014). The encoder is a recurrent
neural network that accepts as input a series of tokens of the form x1, x2,..., xT and
encodes that sequence into vectors of fixed length h1,h2,...,hT. The value T indicates the
length of the input sequence. The decoder is similarly an RNN, and it generates an output
sequence token by token using the following format: y1, y2,..., yT, where T is the length
of the output sequence. The input for the decoder is a single vector of fixed length, hT,
which serves as the input for the RNN. At each point, the hidden states of the encoder
and decoder are indicated by the symbols t, ht, and st, accordingly. Here the problem
comes with compressing long input sequences which results in loss of information so
attention intends to address these issues by giving the decoder access to the complete
encoded input sequence h1,h2,...,hT. This should make the decoding process more
efficient. The primary objective here is to generate attention weights ALP over the input
sequence in order to prioritise the set of places in which relevant information is available
for the purpose of generating the next output token. (Chaudhari et al., 2021).
Here the attention weight is represented by the following equation where α is the 𝑖𝑗
alignment model.
𝐶 𝑗 =
𝑖 = 1
𝑇
Σ α 𝑖𝑗 ℎ 𝑖
The addition of a feed-forward neural network to the design allows for attention to be
learned. This additional feed-forward network will learn a particular weight as a function
of the encoder state as well as the decoder state. This function is referred to as the
alignment function because it assigns a score to the degree to which the state of the
encoder is relevant to the state of the decoder. The scores that are generated by this
alignment function are then provided to a distribution function, which transforms the
scores into attention weights. The alignment function, as well as the distribution
function,as well as the distribution function are both differentiable, and the entire
attention model becomes one huge function that can be differentiated and trained on the
primary architecture through the use of basic backward propagation(Chaudhari et al.,
2021).
2.6 Transformers
Attention mechanisms have evolved to the point where they are now an indispensable
component of convincing sequence modelling and transduction models in a variety of
tasks. This has made it possible to model dependencies without taking into account how
far apart they are in the input or output sequences. However, in almost every
circumstance, such attention mechanisms are utilised in combination with a recurrent
network (Kim et al., 2017). There are just a few exceptions to this rule.
Transformers architecture was introduced as an alternative to recursion as a model
architecture. The Transformer makes use of an attention mechanism in order to infer
global links between input and output. The Transformer makes it possible to attain
previously unimaginable degrees of parallelism and offers the tools necessary to reach a
revolutionary new benchmark of translation quality. A concept of self-attention is used in
the transformer network. A Transformer model does not need sequence aligned RNNs or
convolution in order to calculate representations of its input and output. Instead, it relies
only on self-attention to accomplish this task (Vaswani et al., 2017). When a model is
pre-trained, it may be taught on generic corpora; after that, it can be swiftly tailored to a
specific job while still preserving its good performance. This is made possible by the
process of pre-training. Pre-training has as its primary purpose the acquisition of
meaningful data representations that may later be applied to the learning of future NLP
tasks (Lv et al., 2020).
Encoder and decoder structures, together with multi-head attention, are what make up the
Transformer network, which is an extension of the attention mechanism. This architecture
offers state-of-the-art performance for a wide variety of tasks involving natural language
processing. (Engel, Belagiannis and Dietmayer, 2021).
The first part of the network has an encoder. Two of the sub-layers that make up each
encoder are referred to as the self-attention layer and the feed-forward layer. After
encoding the input text into the vector by utilising the embedding methods in the first
place, the transformer encoder subsequently performs positional encoding in order to
keep the sequence of tokens intact. After that, the input is processed by the self-attention
layer. Within this layer, the attention of each individual word is determined by calculating
the dot product of the key, query, and value vectors. These vectors are first seeded with
random information. Once the attention of each word has been determined, the attention
vectors of all the words are collected and concatenated in order to determine the overall
attention vector. The feed-forward layer receives the output of the attention layer before it
is processed further. This process repeats itself across the six layers that make up the
encoder, and when it reaches the final encoder, the output is sent to the decoder
component.
Transformer architecture ( encoder decoder architecture) is presented above
When it comes to the decoder network, each decoder has three sub-layers: an
encoder-decoder attention layer, a feed-forward layer, and a multi-head attention layer.
The output is generated by the decoder after it receives the input from the previous
encoder, applies self-attention (also known as encoder-decoder attention), and then does
so. In addition to that, the decoder makes advantage of multi-head attention in order to
concentrate on the proper components of the phrases. This procedure is repeated until the
unique token that indicates the end of the sentence is encountered. After that point, the
decoder begins feeding each step into the section at the very bottom of the decoder in
order to produce the final output. The original construction of the transformer consists of
six layers of encoder and six layers of decoder (Etemad, Abidi and Chhabra, 2021).
The Self attention mechanism used in Transformers at encoder decoder level is illustrated
Transformers were initially used as sequence to sequence models in NLP but later they
have been adopted in fields such as Computer Vision, audio processing and also in Life
sciences. Different transformer models were introduced in NLP such as BERT, GPT-2,
XLNet, RoBERTa, GPT-3, etc. Their efficiency and speed over the recurrence and
convolution models for handling text data made them a new state of art (Lin et al., 2022).
3 Methodology:
This section outlines the prerequisites for text summarization. This section's subsequent
phases provide information regarding the dataset, proposed models, and experimental
methodologies.
3.1 Overview
This Project explains the text summarization process using the LSTM networks and
Transformers. These models are trained on a dataset and their performance with respect
to text summarization is evaluated using different metrics. Encoder decoder based LSTM
models and Transformer based models are used to perform this experiment.
Performance metrics and graphical representations are used to describe the model
efficiency and they are compared with each other to find the best fit for text
summarization task
3.2 Dataset
The chosen dataset to accomplish this task is the News Summary dataset which is scraped
from the news articles of The Hindu, Indian Times and The Guardian. Dataset is
downloaded from Kaggle which has two CSV files. The data is collected over a time
period of February 2017 to August 2017. It is present in two files which are concatenated
to create a single dataset with two columns namely ‘text’ and ‘headlines’ while other
columns are excluded.
The Idea is to generate short summaries which are headlines from the text which are
news articles.
Dataset link - https://www.kaggle.com/datasets/sunnysai12345/news-summary
3.3 Proposed Models
The project contains the implementation of encoder decoder based LSTM networks and
Attention based transformer. Encoder decoder based Long Short-Term Memory
Recurrent Neural Networks are known to be the best neural architecture for the text
processing and can be top pick for text summarization where the sequence of the text is
long and complicated. Two types of LSTM experiments are conducted where one is
simple Multi layer LSTM and the other is advanced Bidirectional LSTM. Attention has
gained its popularity making the neural networks advance in the concept of long term
dependencies and the availability of less data. A transformer network is a combination of
attention and neural network which upgraded the AI exponentially. Transformers have
particularly revolutionised the stream of Natural Language processing with their ability
to learn on very small data, pre-training and working effortlessly on the Long sequences.
This project aims to work on the following model and compare them.
● LSTM
● Bidirectional LSTM
● Transformers
In transformers there are several architectures introduced out of which BERT is termed as
state of the art model for text process.
3.4 Metrics
The evaluation metrics for the text summarization task are mentioned below:
● ROUGE:
ROUGE is an acronym that stands for "Recall-Oriented Understudy for Gisting
Evaluation." It is a metric that is frequently used to evaluate the performance of
text summarization models, in particular extractive summarization approaches.
Comparing the amount of n-gram overlap that each summary has with one
another is how ROUGE determines how similar the generated summary and the
reference summary are to one another. There are many different kinds of ROUGE,
and each one uses a different number of n-grams. ROUGE-N looks at how many
N-grams (sequences of N words) are in both the summary that was made and the
summary that was used as a reference. ROUGE-L looks at the two summaries and
finds the longest common subsequences (LCS). ROUGE-W looks at
skip-bigrams, which are two words that come right after each other but don't have
to be next to each other.
Calculating the ROUGE metric involves finding the harmonic mean of the recall
and accuracy values that differ between the reference summary and the summary
that was created. Recall is determined by comparing the number of common
words between the reference summary and the produced summary and dividing
by the total number of common words in the reference summary. In this context,
accuracy is defined as the fraction of n-grams in the generated summary that also
appear in the reference summary, divided by the total number of n-grams in the
generated summary.
ROUGE metrics returns the following metrics:
● Recall:
The fraction that represents the total number of n-grams that are shared
between the generated summary and the reference summary as a
percentage of the total number of n-grams in the reference summary.
Recall (R) = (Number of overlapping n-grams in produced summary and
reference summary) / (Number of total n-grams in the reference summary)
● Precision:
The percentage of n-grams in both the generated summary and the
reference summary that are shared across the two summaries, expressed as
a ratio to the total number of n-grams in the created summary.
Precision (P) = (Number of overlapping n-grams in produced summary
and reference summary) / (Number of total n-grams in the generated
summary)
● The F1-score, also known as the F-measure or the F-score, is a measure of
the accuracy of a model that combines precision and recall. Other names
for this score include the F-measure and the F-score. It has found
widespread application in the fields of machine learning and natural
language processing (NLP), where it is used to assess how well a model is
doing.
F-Measure (F) = (2 * Precision * Recall) / (Precision + Recall)
4 Implementation and Model Designs
4.1 Data Exploration and Preparation
This section involves data exploration and data preprocessing:
4.1.1 Data Exploration:
This section covers the exploration part of the data.
Word Cloud:
Above figure represents the word cloud of text data. Word clouds are frequently used to
analyse and comprehend massive volumes of textual information. Many NLP (natural
language processing) jobs might benefit from their use. Word clouds are helpful because
they make it easy to spot the most essential or common terms in a given text. This might
be helpful for picking out recurrent ideas in a text or finding trends in a huge dataset.
Word clouds are useful for text summarization because they highlight the most frequently
used terms in a given text, thereby revealing its central idea. You may use word clouds to
see which terms appear most often in a summary, which can reveal what topics are most
heavily addressed.
The above word cloud represents the summaries of the data.
Sentence Length and distribution:
The above graph represents the number of words distributed in the sentence and it clearly
explains the length of texts that are mostly close to 40 word length. Summary data has
thirteen words in the maximum number of summaries. This is helpful to decide the
maximum text length and minimum text length.
The summary's maximum length is intended to keep the summary from being too
extensive and difficult to read and understand. This is especially relevant for sentence- or
phrase-based extractive summarization techniques. A lengthy summary might be
confusing and difficult to read.
To make sure the summary isn't too brief and yet captures the essential details of the
source material, its length is constrained. In the case of abstractive summarising
techniques, where the summary is created by the model rather than being taken from the
original text, this is of paramount importance. A poorly written summary may fail to
convey the major ideas of the original material because it omits crucial details.
4.1.2 Data preprocessing:
The process of data preparation is especially significant in the area of text summarization
since it is an essential component in the process of preparing the input data for the model.
The technique of condensing a lengthy text into a more concise version while preserving
the most vital aspects of the original content is referred to as text summarising. Typically,
a text summarising model's input data consists of lengthy articles, papers, or other kinds
of unstructured text. This procedure may be simplified by importing the relevant libraries
connected with the text processing, such as the NLTK library. The example that follows
demonstrates how text cleaning may be accomplished with the help of a variety of
additional libraries..
● The Python Strip function will truncate the original string by removing the
characters that you specify from the beginning and the end of the string. Because
the data collection comprises a great number of symbols, which hold important
information and include things like currencies and percentages, among other
things, those symbols have been replaced with user-defined phrases using the
function replace.
● In order to construct a substring, the sub() function of the Regex library is used.
This method then returns a string that has the values substituted. On a list, it is
utilised on a spot where it may take the place of numerous other items.
● Processing text requires dealing with punctuation and other symbols that don't
contribute all significantly to the meaning of the text. Punctuations may or may
not be beneficial depending on the issue statement; nevertheless, for the purpose
of the current project, they are being eliminated since they make the process of
model training more difficult.
● Word tokenization refers to the act of separating a block of text into its component
words or phrases. It is an essential stage in the text processing process, and it is
used in a broad variety of natural language processing (NLP) applications,
including text categorization, text summarization, and machine translation.
Tokenization is significant since many NLP tasks act on single words or phrases
rather than the complete text. In order to construct a summary of the text, the
model must be able to recognise the most essential words or phrases. Another
reason for tokenization is that, in most cases, the model cannot understand the text
as it is written. Tokenization allows the model to understand the text as a
sequence of tokens, which is a more structured format that the model can
understand, which is more efficient and accurate for the model to process.
The process of word tokenization is required in natural language processing
(NLP) because it enables the model to comprehend the text as a collection of
individual words or phrases, which is essential for the successful execution of
many NLP tasks.
● The process of reducing words to their fundamental or root form, which is
referred to as a lemma, is known as lemmatization. Lemmatization is a process
that is comparable to stemming; however, while stemming attempts to remove
suffixes and return the root word in its most basic form, lemmatization performs a
vocabulary and morphological analysis of words, typically with the assistance of a
lookup table, in order to return the base or root word, also known as the lemma,
which is the form of a word that is most fundamental. Stemming aims to return
the word to its most fundamental form. This helps to minimise the dimensionality
of the data and enhance the performance of natural language processing (NLP)
models. The purpose of lemmatization is to group together the many variants of a
word so that they may be studied as a single item. Tasks like text categorization
and text summarization, in which several variants of a word might have similar
meanings and should be put together, are examples of situations in which
lemmatization can be beneficial.
Although lemmatization might be more computationally demanding, it provides
more accurate results than stemming does. It is essential to keep in mind that
lemmatization calls for a more extensive data set and also an algorithm that is
more complicated than stemming. This is due to the fact that lemmatization must
comprehend the context of the word in order to yield the appropriate base form
which perfectly suits the summarization problem.
● A word or phrase is said to be contracted when specific letters or sounds are
eliminated and replaced with an apostrophe. This process results in the word or
phrase being abbreviated. Contractions, which are used to express a combination
of two words, are widespread in informal written and spoken language and are
used to shorten the terms. The handling of contractions is an essential part of
NLP jobs because, if not handled properly, contractions have the potential to
change the meaning of a phrase and lead to inaccurate results. Contractions have
the potential to make written language less formal, more colloquial, and more
similar to the way people naturally speak; however, they also have the potential to
make written language more difficult to understand, particularly for natural
language processing models and automated systems, which may not be able to
identify and correctly interpret contractions.
There are numerous ways to handle contractions, including extending the
contractions, leaving them unchanged, or deleting the contractions. Method
selection is dependent on the nature of the undertaking and the used model. In the
present problem we have expanded the contractions as they can affect the model
building greatly.
● Stop words handling is the most important step in the text process, Stop words are
a group of frequently used terms in a language that are often eliminated from text
data as a step in the preprocessing phase of activities involving natural language
processing (NLP). These words are referred to as "stop words" as a result of the
fact that they are regarded as "non-content" or "function" words that do not
convey a significant amount of meaning. The elimination of stop words is an
essential stage in the process of text preprocessing. This is because the
elimination of stop words helps to lower the dimensionality of the data, which in
turn boosts the efficiency of NLP models. Getting rid of stop words is another
strategy that may enhance the model's interpretability and make it simpler to
grasp the underlying patterns and connections that are there in the data.
In general, articles, pronouns, prepositions, conjunctions, and other function
words that do not contribute significantly to the overall meaning of a text are
included in the list of stop words; nevertheless, this list may vary depending on
the language.
It is essential to keep in mind that not all NLP activities need the removal of stop
words. For example, language modelling, text creation, and text-to-speech all
require the context of the stop words in order to function properly. In addition, the
elimination of stop words may not be acceptable for certain languages or for other
genres of texts, such as poetry or literature, in which the usage of common words
may be more intentional and significant. In this particular instance, stop words
were taken out since the text summarising process involves lengthy sequences,
and taking out stop words may help minimise the dimensionality of the problem
while also increasing productivity in a shorter amount of time.
4.1.3 Adding tokens at beginning and end of summary:
After performing all of the steps outlined above in order to clean the data, the data are
now prepared for the subsequent stages of model development. One of these stages
involves adding the words "START" and "END" to the summary in order to assist with
determining whether or not the associated summary cell is empty. This is particularly
useful in situations when there is a large amount of data that cannot be examined
manually.
4.1.4 Calculating Maximum Text and Summary Length
Determining the appropriate maximum length of text and summary is required as it is fed
to the learning algorithm which sets a threshold to the length of the sequence in the text
and the summary. These values are very important in model building and also in
restricting the predicted output summary to determined length. The maximum number of
words or tokens that will be sent into the encoder represents the maximum length of the
text that may be passed through it. The performance of the model will be affected by the
length of the text that is fed into the model, since the model will not be able to parse
extremely lengthy passages of text. If the input text is too lengthy, it might lead to the
model being overfit or underfit, or it could need a significant amount of CPU resources to
analyse. Both of these outcomes are undesirable.
The maximum summary length, on the other hand, refers to the greatest possible amount
of words or tokens that are put forth by the decoder as the summary's output. The
performance of the model will be negatively impacted by the length of the summary since
the model will be unable to construct a summary that is either too lengthy or too short. If
the summary is excessively extensive, it is possible that it will not be able to adequately
convey the most essential aspects of the original text. If the summary is too brief, it is
possible that it will not be able to transmit sufficient information from the original text.
4.1.5 Rare Word analysis
Rare words, which are often referred to as out-of-vocabulary (OOV) words, are terms
that do not appear frequently in the data used to train a text summarization model. The
model may have trouble processing certain words, which may have a negative impact on
the model's overall performance.
In the process of text summarization, uncommon words may appear whenever the model
is presented with a term that it has not before seen in the training data. This may occur if
the model is shown new or unfamiliar terms, or words that are exclusive to a certain
subject or area. If the model comes across an uncommon term, there is a possibility that it
will be unable to comprehend either the meaning of the word or the context in which it is
used. This may cause the resulting summary to include errors. The calculation of rare
word frequency is dependent on the word threshold that is established. For the purpose of
this experiment, the rare word frequency was set to 4. Rare words are defined as having a
frequency of less than four occurrences per sentence.
4.2 Model Design
4.2.1 Data Tokenization:
Data tokenization is the basic preprocess involved in NLP. When vectorizing a text
corpus, the Tokenizer class of the Keras library is put to use. In order to accomplish this,
each piece of text that is inputted is either transformed into an integer sequence or
transformed into a vector that has a coefficient for each token in the form of binary
values. There are certain steps involved in tokenization which are explained below.
● fit_on_texts
The fit on texts method is used to tailor the tokenizer to a particular collection of
texts. Output of fit_on_texts is fed as input to text_to_sequences.
● text_to_sequences
The texts are converted into a list of integers by the texts to sequences method,
where each integer stands for a distinct word in the text.
● pad_sequences
It is possible to pad or truncate the sequences using the pad sequences method in
order to bring them all to the same length. Because many deep learning models
are designed to accept inputs of a given length, this is helpful when working with
sequences of varying lengths.
● This Tokenizer class is saved as an object for future use. It is used in building
decoding functions.
4.2.2 LSTM
An encoder decoder based LSTM is implemented in the experiment. This is commonly
used for sequence to sequence tasks where the network consists of two main components,
encoder and decoder. Keras library is used in LSTM implementation.
The encoder is a network that uses LSTM, which stands for long short-term memory. It
takes the input sequence, processes it, and then creates a vector of fixed length that is
referred to as the context vector. This vector provides a summary of the meaning of the
sequence that was entered. The decoder, which is likewise an LSTM network, is given
the context vector at this point in the process. The decoder works its way through the
context vector and, token by token, produces the output sequence. The decoder accepts
the previous output token and the context vector as input at each time step, and then
creates the next output token based on those two pieces of information. Given the input
sequence as well as the context vector, the decoder is educated to produce the output
sequence that has the highest probability of occurring.
● Embedding Layer
The first layer is the embedding layer which converts every word into a numeric
vector of fixed size.
● LSTM
The embedding layer is inserted on top of the first LSTM layer, which is the layer
that already possesses the predetermined shapes for this encoder network. The
sequence uses three LSTM layers as an encoding component, which contributes to
the model's improved ability to comprehend the statement in its full context.
Encoder output, cell state, and hidden state are the three states that may be found
at the output of this LSTM layer. This layer has an input parameter Latent
dimension as well
● Decoding layer is set with an embedding layer with defined dimension which is
on par with the encoder output. One more LSTM layer is added in the decoder
structure.
● Dense layer gets output vocabulary size with the softmax activation function
which is run over through TimeDistributed function, this is used in handling
sequential data and dense layer as a simple neural network. Softmax activation
function is added which acts as a multicategorical activation function as text
summarization is having numerous output values.
● The encoder decoder LSTM architecture is set and ready to be trained with the
data. The model summary gives the overview of the complete sequence to
sequence network including the parameter. The model so built is compiled and
optimiser is used which modifies the attributes like weights and learning rate to
increase the accuracy of the model.
● Early stopping is utilised in order to perform continuous monitoring of the
training of the model. The parameter that is to be monitored is val loss, and if the
loss does not change after two consecutive epochs, the training of the model is
terminated, and it is considered to be trained to its full potential.
Bidirectional LSTM:
Bidirectional LSTM is the extended and advanced version of conventional LSTM
networks.
Bidirectional LSTM has a Bidirectional layer wrapped over the LSTM layer
which helps it in training in forward and reverse direction. This helps in retaining
the information of every word with its context. These LSTM networks have two
hidden states and two cell states forward and backward, they are concatenated to
form the initial state which is fed to the decoder network. The Network is a three
layered architecture where every layer is a bidirectional layer. Each of its outputs
is connected as an input to the next layer. This helps in fine tuning and also helps
in building an accurate model that can retain very long sentences.
Inference Model:
Inference model is the next step after model training, this involves designing an
encoder decoder model that can readily convert the preprocessed text fed into the
model backwards which would help in predicting summary using a trained model.
Generally in machine learning this can be achieved through pipelines which store
the information but does not change the data which is fed. The inference model
has an encoder model which receives input from the training architecture, hidden
state and cell states are designed using the dimensions used in the model
architecture. The decoding layer from the training model is fed with the decoder
states designed in the inference to create decoder outputs and decoder states.
Decoder dense layer wrapped with Timedistributed function helps in reversing the
sequence at word level with maximum efficiency.
A dictionary is built to convert the index values to word values for the target and
source vocabulary. A function to decode the sequence from the trained model is
designed and the predicted value is given as an input to this function to get the
predicted summary in the test case.
Inference also involves changing the text and summary data that has been
changed into vectors to its original state with two different functions.
4.2.2 Transformer
Two transformer architectures are used in the project which are known to be the state of
the art models in handling text data. Both the models are pertained over large corpora and
are fine tuned to maximum extents. They have an attention mechanism which helps in
memorising and giving importance to the critical details of the text.
4.2.2.1 BERT
Bert is considered to be the best performing architecture for handling the text data. BERT
(Bidirectional Encoder Representations from Transformers) is a neural network model
built by Google for natural language processing tasks. It is based on transformers and has
already been trained. BERT has been trained on a huge amount of text data, so it can
understand what the text is about and how the words in a sentence fit together. One of the
most important things about BERT is that it can handle "bidirectional" context. This
means that it looks at the words to the left and right of a word in a sentence to figure out
what it means. This lets BERT figure out what a word means in the context of the whole
sentence, not just the one it's in.
BERT has been shown to be very good at a number of NLP tasks, and on several
benchmark datasets, it has set new state-of-the-art performance levels. It has been used to
improve the performance of models for summarising texts. BERT uses a multi-layer
transformer encoder. Self-attention and feed-forward layers comprise the transformer
encoder. The feed-forward and self-attention layers allow the model to learn a nonlinear
function of the input and focus on various sections of the input for different tasks. BERT
uses an embedding layer to turn words or subwords into high-dimensional vectors. The
transformer encoder self-attentions the input vectors. BERT's "masked language model"
pre-training method predicts the likelihood of a masked word based on the sentence's
context. This teaches BERT sentence connections and text meaning. Finally, a linear
layer creates a probability distribution for the transformer encoder output. Text
categorization and named-entity recognition employ BERT model output. The input
values that a pre-trained BERT takes is the minimum length of value and the text itself.
These values are enough for BERT to summarise the text and produce an output.
4.2.2.2 GPT-2
GPT-2 is also a language model like BERT. OpenAI made GPT-2, which stands for
"Generative Pre-trained Transformer 2." It is a large-scale language model. It is a neural
network model that is trained on a huge amount of text data and can make text that
sounds like it was written by a person. One of the most important things about GPT-2 is
that it can make text that flows well and sounds like it was written by a person. It does
this by being trained on a wide range of text from the internet, which helps it learn a wide
range of language structures and patterns. GPT-2 can be improved on certain tasks like
summarising, completing, and classifying text. It can also be used to make text in a
certain way or about a certain subject.
Each transformer layer in GPT-2 is built using a feed-forward neural network and a set of
multi-head self-attention techniques. While the feed-forward neural network enables the
model to learn a nonlinear function of the input, the transformer layers teach the model to
focus on certain subsets of the input depending on the job at hand. Tokens (words or
subwords) serve as GPT-2's input and are embedded into a high-dimensional vector
representation through an embedding layer. Next, the input vectors are processed by the
transformer layers, which engage in self-attention.
An additional linear layer is applied to the transformer layers' output to provide a
probability distribution across the outcomes. GPT-2's output may be integrated into a
wide range of NLP applications.