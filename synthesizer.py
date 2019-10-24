import io
import numpy as np
import tensorflow as tf
from hparams import hparams
from librosa import effects
from models import create_model
from util import audio


class Synthesizer:
  def load(self, checkpoint_path, model_name='tacotron'):
    print('Constructing model: %s' % model_name)
    speaker_id = tf.placeholder(tf.int32, [None], 'speaker_id')
    mask = tf.placeholder(tf.float32, [None, None, 32], 'mask')
    inputs = tf.placeholder(tf.float32, [None, None, 249], 'inputs')
    input_lengths = tf.placeholder(tf.int32, [1], 'input_lengths')
    with tf.variable_scope('model') as scope:
      self.model = create_model(model_name, hparams)
      self.model.initialize(speaker_id, mask, inputs, input_lengths)
      self.mel_outputs = self.model.outputs[0]

    print('Loading checkpoint: %s' % checkpoint_path)
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(self.session, checkpoint_path)


  def synthesize(self, speaker_id, ppgs):
    feed_dict = {self.model.speaker_id: np.asarray([speaker_id], dtype=np.int32),
                 self.model.inputs: [np.asarray(ppgs, dtype=np.float32)],
                 self.model.input_lengths: np.asarray([len(ppgs)], dtype=np.int32)}
    lpc = self.session.run(self.mel_outputs, feed_dic=feed_dic)
    return lpc
