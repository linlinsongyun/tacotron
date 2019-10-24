import numpy as np
import os
import random
import tensorflow as tf
import threading
import time
import traceback
from text import cmudict, text_to_sequence
from util.infolog import log


_batches_per_group = 32
_p_cmudict = 0.5
_pad = 0


class DataFeeder(threading.Thread):
  '''Feeds batches of data into a queue on a background thread.'''

  def __init__(self, coordinator, metadata_filename, hparams):
    super(DataFeeder, self).__init__()
    self._coord = coordinator
    self._hparams = hparams
    self._cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
    self._offset = 0

    # Load metadata:
    self._datadir = os.path.dirname(metadata_filename)
    with open(metadata_filename, encoding='utf-8') as f:
      self._metadata = [line.strip().split('|') for line in f]
      
    # Create placeholders for inputs and targets. Don't specify batch size because we want to
    # be able to feed different sized batches at eval time.
    self._placeholders = [
      tf.placeholder(tf.int32, [None], 'speaker_id'), 
      tf.placeholder(tf.float32, [None, None, hparams.num_mels], 'mask'),
      tf.placeholder(tf.float, [None, None, hparams.num_ppgs], 'inputs'),
      tf.placeholder(tf.int32, [None], 'input_lengths'),
      tf.placeholder(tf.float32, [None, None, hparams.num_mels], 'mel_targets')
    ]

    # Create queue for buffering data:
    queue = tf.FIFOQueue(8, [tf.int32, tf.float32, tf.float32, tf.int32, tf.float32], name='input_queue')
    self._enqueue_op = queue.enqueue(self._placeholders)
    self.speaker_id, self.mask, self.inputs, self.input_lengths, self.mel_targets = queue.dequeue()
    self.speaker_id.set_shape(self._placeholders[0].shape)
    self.mask.set_shape(self._placeholders[1].shape)
    self.inputs.set_shape(self._placeholders[2].shape)
    self.input_lengths.set_shape(self._placeholders[3].shape)
    self.mel_targets.set_shape(self._placeholders[4].shape)
 


  def start_in_session(self, session):
    self._session = session
    self.start()


  def run(self):
    try:
      while not self._coord.should_stop():
        self._enqueue_next_group()
    except Exception as e:
      traceback.print_exc()
      self._coord.request_stop(e)


  def _enqueue_next_group(self):
    start = time.time()

    # Read a group of examples:
    n = self._hparams.batch_size
    r = self._hparams.outputs_per_step
    examples = [self._get_next_example() for i in range(n * _batches_per_group)]

    # Bucket examples based on similar output sequence length for efficiency:
    examples.sort(key=lambda x: x[-1])
    batches = [examples[i:i+n] for i in range(0, len(examples), n)]
    random.shuffle(batches)

    log('Generated %d batches of size %d in %.03f sec' % (len(batches), n, time.time() - start))
    for batch in batches:
      feed_dict = dict(zip(self._placeholders, _prepare_batch(batch, r)))
      self._session.run(self._enqueue_op, feed_dict=feed_dict)


  def _get_next_example(self):
    '''Loads a single example (input, mel_target, linear_target, cost) from disk'''
    if self._offset >= len(self._metadata):
      self._offset = 0
      random.shuffle(self._metadata)
    meta = self._metadata[self._offset]
    self._offset += 1

    ppgs = np.load(os.path.join(self._hparams.ppgs_dir, meta[0]))
    mel_target = np.load(os.path.join(self._hparams.lpc32_dir, meta[2]))
    speaker_id = meta[4]
    mask = np.ones((mel_target.shape[0], mel_target.shape[1]))
    return (speaker_id, mask, ppgs, len(ppgs), mel_target)



  def _maybe_get_arpabet(self, word):
    arpabet = self._cmudict.lookup(word)
    return '{%s}' % arpabet[0] if arpabet is not None and random.random() < 0.5 else word


def _prepare_batch(batch, outputs_per_step):
  random.shuffle(batch)
  speaker_id = np.asarray([x[0] for x in batch], dtype=np.int32)
  mask = _prepare_targets([x[1] for x in batch], outputs_per_step)
  inputs = _prepare_inputs([x[2] for x in batch])
  input_lengths = np.asarray([len(x[2]) for x in batch], dtype=np.int32)
  mel_targets = _prepare_targets([x[4] for x in batch], outputs_per_step)
  if (mel_targets.shape != mask.shape):
    print('not equal')
    os.exist(0)
  return (speaker_id, mask, inputs, input_lengths, mel_target)


def _prepare_inputs(inputs):
  max_len = max((len(x) for x in inputs))
  return np.stack([_pad_input(x, max_len) for x in inputs])


def _prepare_targets(targets, alignment):
  max_len = max((len(t) for t in targets)) + 1
  return np.stack([_pad_target(t, _round_up(max_len, alignment)) for t in targets])


def _pad_input(x, length):
  return np.pad(x, [(0, length - x.shape[0]), (0,0)], mode='constant', constant_values=_pad)


def _pad_target(t, length):
  return np.pad(t, [(0, length - t.shape[0]), (0,0)], mode='constant', constant_values=_pad)


def _round_up(x, multiple):
  remainder = x % multiple
  return x if remainder == 0 else x + multiple - remainder
