# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import modeling_tf2
import optimization_tf2
import tokenization
import tensorflow as tf
import pandas as pd
import json

# import pickle
# import nltk
# import re
flags = tf.compat.v1.flags

FLAGS = flags.FLAGS
base=os.path.dirname(__file__)
print(base)
## Required parameters
# if os.name == 'nt':
#     bert_path = 'D:\\Research\\multi_cased_L-12_H-768_A-12'
#     root_path = r'D:/Research/Bert_Ner'
# else:
#     # bert_path = '/home/linhlt/matt/bert_ner/bert-models/multi_cased_L-12_H-768_A-12'
#     # root_path = '/home/linhlt/Levi/chatbot_platform_nlp'
# bert_path = 'gs://test_bucket_share_1/uncased_L-12_H-768_A-12'
# input=os.path.join(base,'/../input/bertpretrained')
input='/kaggle/input/bertpretrained'
# output=os.path.join(base,'/../output')
output='/kaggle/output'
bert_path=os.path.join(input,'multi_cased_L-12_H-768_A-12/multi_cased_L-12_H-768_A-12')
print(bert_path)
project_path=base
root_path = base
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

flags.DEFINE_string(
    "data_dir", os.path.join(root_path, 'data'),
    "The input datadir.",
)
flags.DEFINE_string(
    "bert_config_file", os.path.join(bert_path, 'bert_config.json'),
    "The config json file corresponding to the pre-trained BERT model."
)

flags.DEFINE_string(
    "task_name", 'Uland', "The name of the task to train."
)

flags.DEFINE_string(
    "output_dir",os.path.join(output,'model_zalo'),
    "The output directory where the model checkpoints will be written."
)

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", os.path.join(bert_path, 'bert_model.ckpt'),
    "Initial checkpoint (usually from a pre-trained BERT model)."
)

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text."
)

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization."
)

flags.DEFINE_boolean('clean', True, 'remove the files which created by last training')

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("use_tpu", True, "Whether to use TPU or GPU/CPU.")
flags.DEFINE_string(
    "tpu_name",'grpc://10.0.0.2:8470' ,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

flags.DEFINE_bool("do_eval",True, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_predict",False, "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 8, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 10.0, "Total number of training epochs to perform.")
flags.DEFINE_float('droupout_rate', 0.5, 'Dropout rate')
flags.DEFINE_float('clip', 5, 'Gradient clip')
flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_string("vocab_file", os.path.join(bert_path, 'vocab.txt'),
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")
flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")
flags.DEFINE_string('data_config_path', os.path.join(project_path, 'data.conf'),
                    'data config file, which save train and dev config')


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """
  # def __init__(self):
  #   self.text_a = ''
  #   self.text_b = ''
  #   # self.labels = labels
class InputExample(object):
  """A single training/test example for simple sequence classification."""
  def __init__(self, guid, text_a, text_b, labels=None,entity1=None,entity2=None):
    """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.labels = labels
    # self.entity1=entity1
    # self.entity2=entity2

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_ids,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_ids = label_ids
    self.is_real_example = is_real_example


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_csv(self, data_dir):
      """Reads a tab separated value file."""
      y = []
      X1 = []
      X2 = []
      # EN1=[]
      # EN2=[]
      df = pd.read_csv(data_dir, sep='\t', encoding='utf-8', error_bad_lines=False)
      for i in df.index:
        y.append(str(int(df['is_duplicate'][i])))
        X1.append(str(df['question1'][i]))
        X2.append(str(df['question2'][i]))

      return (X1, X2, y)


class UlandProcessor(DataProcessor):
    """Processor for the XNLI data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        print('PATH', os.path.join(data_dir, 'train_new.json'))
        X1 = []
        X2 = []
        Y=[]
        with open(os.path.join(data_dir,'train_new.json'), 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        for i in range(len(data)):
            if data[i]['label'] == True:
                Y.append('1')
            else:
                Y.append('0')
            X1.append(str(data[i]['question']))
            X2.append(str(data[i]['text']))
        num_yes=sum([1 if y=='1' else 0 for y in Y])
        print("------------------------------------------------")
        print('NUM Yes: ', num_yes)
        print("------------------------------------------------")
        examples = []
        for i in range(len(X1)):
            # if i == 0:
            #     continue
            guid = "test-%d" % (i)
            if not isinstance(X1[i], str):
                continue
            text1 = tokenization.convert_to_unicode(X1[i])
            text2 = tokenization.convert_to_unicode(X2[i])
            label = tokenization.convert_to_unicode(Y[i])
            # en1=tokenization.convert_to_unicode(EN1[i])
            # en2=tokenization.convert_to_unicode(EN2[i])
            examples.append(
                InputExample(guid=guid, text_a=text1, text_b=text2,labels=label))
        return examples

    def get_test_examples(self, data_dir):
        """See base class."""
        # (X1,X2, Y) = self._read_csv(os.path.join(data_dir,'test.tsv'))
        X1 = []
        X2 = []
        Y = []
        with open(os.path.join(data_dir,'test.json'), 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        for i in range(len(data)):
            if data[i]['label'] == True:
                Y.append('1')
            else:
                Y.append('0')
            X1.append(str(data[i]['question']))
            X2.append(str(data[i]['text']))
        num_yes=sum([1 if y=='1' else 0 for y in Y])
        print("------------------------------------------------")
        print('NUM Yes: ', num_yes)
        print("------------------------------------------------")
        examples = []
        for i in range(len(X1)):
            # if i == 0:
            #     continue
            guid = "test-%d" % (i)
            if not isinstance(X1[i], str):
                continue
            text1 = tokenization.convert_to_unicode(X1[i])
            text2 = tokenization.convert_to_unicode(X2[i])
            label = tokenization.convert_to_unicode(Y[i])
            # en1=tokenization.convert_to_unicode(EN1[i])
            # en2=tokenization.convert_to_unicode(EN2[i])
            examples.append(
                    InputExample(guid=guid, text_a=text1,text_b=text2,labels=label))
        return examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        X1 = []
        X2 = []
        Y = []
        with open(os.path.join(data_dir,'valid.json'), 'r', encoding='utf-8') as json_file:
            data = json.load(json_file)
        for i in range(len(data)):
            if data[i]['label'] == True:
                Y.append('1')
            else:
                Y.append('0')
            X1.append(str(data[i]['question']))
            X2.append(str(data[i]['text']))
        num_yes=sum([1 if y=='1' else 0 for y in Y])
        print("------------------------------------------------")
        print('NUM Yes: ', num_yes)
        print("------------------------------------------------")
        examples = []
        for i in range(len(X1)):
            # if i == 0:
            #     continue
            guid = "test-%d" % (i)
            if not isinstance(X1[i], str):
                continue
            text1 = tokenization.convert_to_unicode(X1[i])
            text2 = tokenization.convert_to_unicode(X2[i])
            label = tokenization.convert_to_unicode(Y[i])
            # en1=tokenization.convert_to_unicode(EN1[i])
            # en2=tokenization.convert_to_unicode(EN2[i])
            examples.append(
                InputExample(guid=guid, text_a=text1, text_b=text2,labels=label))
        return examples

    def get_labels(self):
        """See base class."""
        return ['1', '0']


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
      return InputFeatures(
          input_ids=[0] * max_seq_length,
          input_mask=[0] * max_seq_length,
          segment_ids=[0] * max_seq_length,
          label_ids=0,
          is_real_example=False)

  label_map = {}
  for (i, label) in enumerate(label_list):
      label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
      tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
      # Modifies `tokens_a` and `tokens_b` in place so that the total
      # length is less than the specified length.
      # Account for [CLS], [SEP], [SEP] with "- 3"
      _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
      # Account for [CLS] and [SEP] with "- 2"
      if len(tokens_a) > max_seq_length - 2:
          tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
      tokens.append(token)
      segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
      for token in tokens_b:
          tokens.append(token)
          segment_ids.append(1)
      tokens.append("[SEP]")
      segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.labels]
  if ex_index < 5:
      tf.compat.v1.logging.info("*** Example ***")
      tf.compat.v1.logging.info("guid: %s" % (example.guid))
      tf.compat.v1.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in tokens]))
      tf.compat.v1.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
      tf.compat.v1.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
      tf.compat.v1.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
      tf.compat.v1.logging.info("label: %s (id = %d)" % (example.labels, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_ids=label_id,
      is_real_example=True)
  return feature


def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.compat.v1.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature([feature.label_ids])
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.io.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.io.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.io.FixedLenFeature([], tf.int64),
      "is_real_example": tf.io.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.io.parse_single_example(serialized=record, features=name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, dtype=tf.int32)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling_tf2.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()
  print('output shape',output_layer.shape)
  hidden_size = bert_config.hidden_size # output_layer.shape[-1].value

  output_weights = tf.compat.v1.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.compat.v1.get_variable(
      "output_bias", [num_labels], initializer=tf.compat.v1.zeros_initializer())

  with tf.compat.v1.variable_scope("loss"):
    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, rate=1 - (0.9))

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(input_tensor=one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(input_tensor=per_example_loss)

    return (loss, per_example_loss, logits, probabilities)


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.compat.v1.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.compat.v1.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(input=label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.compat.v1.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling_tf2.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.compat.v1.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.compat.v1.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.compat.v1.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization_tf2.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(input=logits, axis=-1, output_type=tf.int32)
        accuracy = tf.compat.v1.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.compat.v1.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])
      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.compat.v1.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.compat.v1.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features


def main(_):
  tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

  processors = {
      "uland": UlandProcessor,
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling_tf2.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.compat.v1.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.compat.v1.estimator.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.compat.v1.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.compat.v1.estimator.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    file_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
    tf.compat.v1.logging.info("***** Running training *****")
    tf.compat.v1.logging.info("  Num examples = %d", len(train_examples))
    tf.compat.v1.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.compat.v1.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_eval:
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    num_actual_eval_examples = len(eval_examples)
    if FLAGS.use_tpu:
      
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on. These do NOT count towards the metric (all tf.metrics
      # support a per-instance weight, and these get a weight of 0.0).
      while len(eval_examples) % FLAGS.eval_batch_size != 0:
        eval_examples.append(PaddingInputExample())

    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

    tf.compat.v1.logging.info("***** Running evaluation *****")
    tf.compat.v1.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
    tf.compat.v1.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    if FLAGS.use_tpu:
      assert len(eval_examples) % FLAGS.eval_batch_size == 0
      eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.compat.v1.gfile.GFile(output_eval_file, "w") as writer:
      tf.compat.v1.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.compat.v1.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))

  if FLAGS.do_predict:
    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    num_actual_predict_examples = len(predict_examples)
    if FLAGS.use_tpu:
      pass
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on.
      # while len(predict_examples) % FLAGS.predict_batch_size != 0:
      #   predict_examples.append(PaddingInputExample())

    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, label_list,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file)

    tf.compat.v1.logging.info("***** Running prediction*****")
    tf.compat.v1.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.compat.v1.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)

    output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
    with tf.compat.v1.gfile.GFile(output_predict_file, "w") as writer:
      num_written_lines = 0
      tf.compat.v1.logging.info("***** Predict results *****")
      for (i, prediction) in enumerate(result):
        probabilities = prediction["probabilities"]
        if i >= num_actual_predict_examples:
          break
        output_line = "\t".join(
            str(class_probability)
            for class_probability in probabilities) + "\n"
        writer.write(output_line)
        num_written_lines += 1
    assert num_written_lines == num_actual_predict_examples


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.compat.v1.app.run()

import re
class Bert_classifi(object):
    def __init__(self):
        tf.compat.v1.reset_default_graph()
        bert_config = modeling_tf2.BertConfig.from_json_file(FLAGS.bert_config_file)
        init_checkpoint = FLAGS.init_checkpoint
        self.use_one_hot_embeddings = False
        processor= UlandProcessor()
        self.label_list = processor.get_labels()
        self.num_labels= len(self.label_list)
        self.predict, self.prob = self._build(bert_config, self.use_one_hot_embeddings)
        lastest_checkpoint=tf.train.latest_checkpoint(FLAGS.output_dir)
        self.initialize_and_restore_session(lastest_checkpoint)
        self.eval_features = []
        self.ori_eval_features = []
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
        label_map = {}
        for (i, label) in enumerate(self.label_list):
            label_map[label] = i
        # with open(os.path.join(FLAGS.output_dir, 'label2id.pkl'), 'rb') as w:
        #     label_map= pickle.load(w)
        self.label_map = dict((v, k) for k, v in label_map.items())
        pass

    def initialize_and_restore_session(self, model_dir):
        """Defines self.sess and initialize the variables"""
        print("Initializing tf session")
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True, device_count={'GPU': 0})
        # config.gpu_options.per_process_gpu_memory_fraction = 0.3
        self.sess = tf.compat.v1.Session(config=config)
        self.sess = tf.compat.v1.Session()
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.saver = tf.compat.v1.train.Saver()
        self.saver.restore(self.sess, model_dir)
        print("Complete restore model from ", model_dir)

    def create_model(self,bert_config,is_training, input_ids, input_mask, segment_ids,
                      num_labels, use_one_hot_embeddings):
        """Creates a classification model."""
        model = modeling_tf2.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

        # In the demo, we are doing a simple classification task on the entire
        # segment.
        #
        # If you want to use the token-level output, use model.get_sequence_output()
        # instead.
        output_layer = model.get_pooled_output()

        hidden_size = output_layer.shape[-1].value

        output_weights = tf.compat.v1.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.compat.v1.get_variable(
            "output_bias", [num_labels], initializer=tf.compat.v1.zeros_initializer())

        with tf.compat.v1.variable_scope("loss"):
            logits = tf.matmul(output_layer, output_weights, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            prob= tf.nn.softmax(logits)
            predictions = tf.argmax(input=logits, axis=-1, output_type=tf.int32)
            return predictions, prob

    def _build(self, bert_config, use_one_hot_embeddings):
        self.input_ids = tf.compat.v1.placeholder(tf.int32, shape=[None, FLAGS.max_seq_length])
        self.input_mask = tf.compat.v1.placeholder(tf.int32, shape=[None, FLAGS.max_seq_length])
        self.segment_ids = tf.compat.v1.placeholder(tf.int32, shape=[None, FLAGS.max_seq_length])
        predict, prob = self.create_model(bert_config, False, self.input_ids, self.input_mask,
                 self.segment_ids, self.num_labels, use_one_hot_embeddings)
        return predict, prob

    def init_checkpoint(self, init_checkpoint):
        tvars = tf.compat.v1.trainable_variables()
        (assignment_map, initialized_variable_names) = modeling_tf2.get_assignment_map_from_checkpoint(tvars,
                                                                                                   init_checkpoint)
        tf.compat.v1.train.init_from_checkpoint(init_checkpoint, assignment_map)
        pass


    def _create_example(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[0])
            examples.append(InputExample(guid=guid, text=text, label=label))
        return examples

    def get_feed_dict(self, features):
        feed={}
        feed[self.input_ids]=features['input_ids']
        feed[self.input_mask]= features['input_mask']
        feed[self.segment_ids]= features['segment_ids']
        return feed


    def predict_raw_sentence(self,sentence1,sentence2):
        fetures=self.convert_example_to_features(sentence1,sentence2,
                                                 FLAGS.max_seq_length, self.tokenizer)
        predict, prob= self.sess.run([self.predict, self.prob],feed_dict=self.get_feed_dict(fetures))
        return int(self.label_map[ list(predict)[0]]), prob[0][predict[0]]

    def convert_single_prediction_example(self, sentence1 , sentence2, max_seq_length, tokenizer):
        textlist1 = re.split(r"\s+",sentence1)
        textlist2 = re.split(r"\s+", sentence2)
        tokens = []
        for i, word in enumerate(textlist1):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
        if len(tokens) >= max_seq_length - 1:
            tokens = tokens[0:(max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        # append("O") or append("[CLS]") not sure!
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
        ntokens.append("[SEP]")
        segment_ids.append(0)

        tokens2=[]
        for i, word in enumerate(textlist2):
            token = tokenizer.tokenize(word)
            tokens2.extend(token)
        if len(tokens2) >= max_seq_length - 1:
            tokens2 = tokens[0:(max_seq_length - 2)]
        # append("O") or append("[SEP]") not sure!=
        for i, token in enumerate(tokens2):
            ntokens.append(token)
            segment_ids.append(1)
        ntokens.append("[SEP]")
        segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_ids=[]# label_mask = label_mask
        )
        return feature

    def convert_example_to_features(self, sentence1,sentence2, max_seq_length, tokenizer):
        feature = self.convert_single_prediction_example(sentence1,sentence2, max_seq_length, tokenizer)
        def create_int_feature(values):
            f = list(values)
            return f
        features = collections.OrderedDict()
        features["input_ids"] = [create_int_feature(feature.input_ids)]
        features["input_mask"] = [create_int_feature(feature.input_mask)]
        features["segment_ids"] = [create_int_feature(feature.segment_ids)]
        return features

    def test(self,text1=None,text2=None):
        if text1==None:
            print(self.predict_raw_sentence("Minh muon mua nha o Hoang Cau"))
        else:
            print(self.predict_raw_sentence(text1,text2))

# classifi=Bert_classifi()
# # classifi.test('vi Thông thường, mỗi ngôn ngữ sẽ có một bộ đọc tương ứng',
# #               'en Typically, each language will have a corresponding reader')
# print(classifi.predict_raw_sentence('How can I be a good geologist?','What should I do to be a great geologist?')[0])
# i=0
# directory = os.fsencode('./drive/My Drive/AI_COLAB/BERT_tensor/division_test_no_entiti2')
# for file in os.listdir(directory):
#   print(i)
#   filename = os.fsdecode(file)
#   filename=os.path.join('./drive/My Drive/AI_COLAB/BERT_tensor/division_test_no_entiti2',filename)
#   data={'test_id':[],'is_duplicate':[]}
#   df=pd.read_csv(filename, sep='\t', encoding='utf-8', error_bad_lines=False)
#   for index in df.index:
#     try:
#         label=classifi.predict_raw_sentence(str(df['question1'][index]),str(df['question2'][index]))
#                                             # +' [CLS] '+str(df['entity1'][index]),str(df['q2'][index])+' [SEP] '+str(df['entity2'][index]))[0]
#         # print(label[0])
#         data['is_duplicate'].append(int(label[0]))
#         data['test_id'].append(df['test_id'][index])
#     except:
#         data['is_duplicate'].append(0)
#         data['test_id'].append(df['test_id'][index])
#         pass
#   data=pd.DataFrame(data)
#   data.to_csv('./drive/My Drive/AI_COLAB/BERT_tensor/predict1112/'+str(i)+'.csv',index=False,sep=',')
#   print(filename)
#   i+=1
