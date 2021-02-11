# Importing essential libraries
from flask import Flask, render_template, request




app = Flask(__name__)


from argparse import Namespace
hype = Namespace(
          LR = 0.00001,
          BATCH_SIZE = 64,
          NUM_EPOCHS = 100,
          CLIP = 1,
       )
model_hype = Namespace(
            EMBEDDING_SIZE = 8,
            GRU_UNITS = 256,
            ATTN_SIZE = 2,
       )
hype.BATCH_SIZE
'''

import tensorflow as tf
from tensorflow.keras.preprocessing import text
from tensorflow.keras.preprocessing import sequence

import re
import string


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


def tokenize(sent_list):
  tokenizer = text.Tokenizer(filters='')
  tokenizer.fit_on_texts(sent_list)

  tensor_list = tokenizer.texts_to_sequences(sent_list)
  tensor_list = sequence.pad_sequences(tensor_list, padding='post')
  
  return {'Tensors': tensor_list, 'Tokenizer': tokenizer}


  '''




'''






from tensorflow import nn
from tensorflow.keras import layers, Model
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


import os, uuid
from azure.storage import blob
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__
os.mkdir("tf_checkpoint")
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



from matplotlib import ticker
def evaluate(sentence):
  attention_plot = np.zeros((eng_tensors.shape[1],mar_tensors.shape[1]))

  sentence = sep_punk(sentence)

  inputs = [mar_tokenizer.word_index[i] for i in sentence.lower().split(' ') if i!='']
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

  '''
  



@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'GET':
		    return render_template('index.html', x='dev')
    if request.method == 'POST':
	    sen = request.form.get("mar_input") 
	    return render_template('result.html', prediction=sen)
'''
@app.route('/')
def hype():
  from argparse import Namespace
  hype = Namespace(
            LR = 0.00001,
            BATCH_SIZE = 64,
            NUM_EPOCHS = 100,
            CLIP = 1,
         )
  model_hype = Namespace(
              EMBEDDING_SIZE = 8,
              GRU_UNITS = 256,
              ATTN_SIZE = 2,
         )
@app.route('/')
def model():
    from tensorflow import nn
    from tensorflow.keras import layers, Model
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
     

@app.route('/')
def blob():
    import os, uuid
    from azure.storage import blob
    from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__
    os.mkdir("tf_checkpoint")
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



@app.route('/', methods=['POST'])
def inference():
    from matplotlib import ticker
    def evaluate(sentence):
      attention_plot = np.zeros((eng_tensors.shape[1],mar_tensors.shape[1]))

      sentence = sep_punk(sentence)

      inputs = [mar_tokenizer.word_index[i] for i in sentence.lower().split(' ') if i!='']
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

    if request.method == 'POST':
       return render_template('result.html',x="IT WORKS")

'''    
    


if __name__ == '__main__':
	app.run(debug=True)