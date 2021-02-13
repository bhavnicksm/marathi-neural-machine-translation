# Importing essential libraries
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from argparse import Namespace
import tensorflow as tf
from tensorflow.keras.preprocessing import text
from tensorflow.keras.preprocessing import sequence
import re
import string
from tensorflow.data import Dataset
from tensorflow import nn
from tensorflow.keras import layers, Model
from tensorflow.keras import optimizers as optim
from tensorflow.keras import losses
import os, uuid
from azure.storage import blob
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__
from matplotlib import ticker



app = Flask(__name__)


hype = Namespace(
          LR = 0.00001,
          BATCH_SIZE = 64,
          NUM_EPOCHS = 100,
          CLIP = 1,
       )
model_hype = Namespace(
            EMBEDDING_SIZE = 128,
            GRU_UNITS = 512,
            ATTN_SIZE = 2,
       )
#Reading the dataset
data = pd.read_csv('data.csv', header=None)
data.columns = ['english', 'marathi']




def sep_punk(text):
  for punk in string.punctuation:
    text = text.replace(punk," "+punk+" ")
  return text
def add_init_token(sent_list):
  new_sent_list = []
  for sent in sent_list:
    sent = sep_punk(sent)
    sent = '<sos> ' + sent + ' <eos>'
    new_sent_list.append(sent)
  return new_sent_list

mar_list = list(data['marathi'])
eng_list = list(data['english'])

mar_list = add_init_token(mar_list)
eng_list = add_init_token(eng_list)



def tokenize(sent_list):
  tokenizer = text.Tokenizer(filters='', oov_token='<unk>')
  tokenizer.fit_on_texts(sent_list)
  tensor_list = tokenizer.texts_to_sequences(sent_list)
  tensor_list = sequence.pad_sequences(tensor_list, padding='post')
  
  return {'Tensors': tensor_list, 'Tokenizer': tokenizer}


marathi = tokenize(sent_list=mar_list)
english = tokenize(sent_list=eng_list)

mar_tokenizer = marathi['Tokenizer']
eng_tokenizer = english['Tokenizer']


mar_tensors = marathi['Tensors']
eng_tensors = english['Tensors']




BUFFER_SIZE = len(mar_tensors)
BATCH_SIZE = hype.BATCH_SIZE

dataset = Dataset.from_tensor_slices((mar_tensors, eng_tensors)).shuffle(BUFFER_SIZE, reshuffle_each_iteration=True)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)


ex_mar_batch, ex_eng_batch = next(iter(dataset))





class Encoder(Model):
  def __init__(self, vocab_size, embedding_size, enc_units, batch_size):
    
    super(Encoder, self).__init__()
    self.batch_size = batch_size
    self.enc_units = enc_units
    self.embedding = layers.Embedding(vocab_size, embedding_size)
    self.gru = layers.GRU(enc_units, return_sequences= True, return_state= True, recurrent_initializer='glorot_uniform')
  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state=hidden)
    return output,state
  def initialize_hidden_state(self):
    return tf.zeros((self.batch_size, self.enc_units))
mar_vocab_size = len(mar_tokenizer.word_index) + 1
EMBEDDING_SIZE = model_hype.EMBEDDING_SIZE
GRU_UNITS = model_hype.GRU_UNITS
encoder = Encoder(mar_vocab_size, EMBEDDING_SIZE, GRU_UNITS, BATCH_SIZE)

#Example input

ex_hidden = encoder.initialize_hidden_state()
sample_out, sample_hidden = encoder(ex_mar_batch, ex_hidden)


class Attention(layers.Layer):
  def __init__(self, units):

    super(Attention, self).__init__()
    self.W1 = layers.Dense(units)
    self.W2 = layers.Dense(units)
    self.V = layers.Dense(1)

  def call(self, q, val):
    #make q i.e. the hidden value into the same shape
    q_with_time_axis = tf.expand_dims(q, 1)

    score = self.V( nn.tanh( self.W1(q_with_time_axis) + self.W2(val) ) )

    attention_w = nn.softmax(score, axis=1)

    context_vec = attention_w * val
    context_vec = tf.reduce_sum(context_vec, axis=1)

    return context_vec, attention_w



ATTN_SIZE = model_hype.ATTN_SIZE
attention = Attention(ATTN_SIZE)
#example code
attn_res ,  attn_w = attention(sample_hidden, sample_out)


class Decoder(Model):

  def __init__(self, vocab_size, embedding_size, dec_units, batch_size):
    super(Decoder,self).__init__()

    self.batch_size = batch_size
    self.dec_units = dec_units
    
    self.embedding = layers.Embedding(vocab_size, embedding_size)
    self.gru = layers.GRU(dec_units, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')

    self.fc = layers.Dense(vocab_size)

    self.attention = Attention(dec_units)

  def call(self, x, hidden, enc_out):
    context_vec, attention_w = self.attention(hidden, enc_out)

    x = self.embedding(x)

    x = tf.concat([tf.expand_dims(context_vec, 1), x], axis=-1)

    output, state = self.gru(x)

    output = tf.reshape(output, (-1, output.shape[2]))

    x = self.fc(output)

    return x, state, attention_w


eng_vocab_size = len(eng_tokenizer.word_index) + 1

decoder = Decoder(eng_vocab_size, EMBEDDING_SIZE, GRU_UNITS, BATCH_SIZE)

#example code

sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),sample_hidden, sample_out)




optimizer = optim.Adam(learning_rate=hype.LR)
criteria = losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss = criteria(real, pred)

  mask = tf.cast(mask, dtype=loss.dtype)
  loss = loss*mask

  return tf.reduce_mean(loss)




#os.mkdir("tf_checkpoint")
connect_str = "DefaultEndpointsProtocol=https;AccountName=tfmodel;AccountKey=PinzJZWJy/mFOWDgkBcCTPA9Fnfr7/qvaZSbjxQVH4YGrBt4MseqbKYjUGNKYX9PpBh+zgAk6uDrVpmvejBCiw==;EndpointSuffix=core.windows.net"
blob_service_client =  BlobServiceClient.from_connection_string(connect_str)
container_client = blob_service_client.get_container_client("tf-ckpt")
blob_list = [blob.name for blob in container_client.list_blobs()]
for file in blob_list:
  blob_client = blob_service_client.get_blob_client('tf-ckpt', file)
  with open('./tf_checkpoint/'+file, "wb") as f:
    f.write(blob_client.download_blob().readall())   
checkpoint_path = './tf_checkpoint'
checkpoint = tf.train.Checkpoint(epoch=tf.Variable(1),
                                 optimizer = optimizer,
                                 encoder=encoder,
                                 decoder=decoder,
                                 )
manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=1)
manager.checkpoints
manager.restore_or_initialize()

def evaluate(sentence):
  attention_plot = np.zeros((eng_tensors.shape[1],mar_tensors.shape[1]))
  sentence = sep_punk(sentence)
  inputs = []
  for i in sentence.lower().split(' '):
    if i != '' :
      if (i in mar_tokenizer.word_docs.keys()):
        inputs.append(mar_tokenizer.word_index[i])
      else: inputs.append(mar_tokenizer.word_index['<unk>'])
 
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                             maxlen=mar_tensors.shape[1],
                                                             padding='post')
  inputs = tf.convert_to_tensor(inputs)
  result = []
  hidden = [tf.zeros((1, GRU_UNITS))]
  enc_out, enc_hidden = encoder(inputs, hidden)
  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([eng_tokenizer.word_index['<sos>']], 0)
  for t in range(eng_tensors.shape[1]):
    predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)
    # storing the attention weights to plot later on
    attention_weights = tf.reshape(attention_weights, (-1, ))
    attention_plot[t] = attention_weights.numpy()
    predicted_id = tf.argmax(predictions[0]).numpy()
    result.append(eng_tokenizer.index_word[predicted_id])
    if eng_tokenizer.index_word[predicted_id] == '<eos>':
      return result, sentence, attention_plot
    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)
  return result, sentence, attention_plot
 
 



@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
		    return render_template('index.html')
    if request.method == 'POST':
     sen = request.form.get("mar_input") 
     res,sen_translation,unk=evaluate(sen)
     res.remove("<eos>")
     res_sen= ' '.join(res)
     return render_template('result.html', prediction=sen_translation,result=res_sen)
   


if __name__ == '__main__':
	app.run()