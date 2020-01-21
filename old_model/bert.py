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
import csv
import os
import modeling
import optimization
import tokenization
import tensorflow as tf
import tensorflow_ranking as tfr
import numpy as np

flags = tf.flags

FLAGS = flags.FLAGS
BERT_BASE_DIR = "/home1/fangzheng/project/bert/uncased_L-12_H-768_A-12/"
GLUE_DIR = "/home1/fangzheng/project/bert/glue_data/"
task_name_flag = "AIDA"
do_train_flag = False
do_eval_flag = False
do_predict_flag = True
data_dir_flag = GLUE_DIR + "aida_cut/"
vocab_file_flag = BERT_BASE_DIR + "vocab.txt"
bert_config_file_flag = BERT_BASE_DIR + "bert_config.json" 
# init_checkpoint_flag = BERT_BASE_DIR + "bert_model.ckpt"
init_checkpoint_flag = "/home1/fangzheng/project/bert/glue_data/aida_cut/output/" + "old_model.ckpt-11779"
# init_checkpoint_flag = "/home1/fangzheng/project/bert/glue_data/AIDA_data/output/" + "old_model.ckpt-10501"
max_seq_length_flag = 100
train_batch_size_flag = 4
eval_batch_size_flag = 4
test_batch_size_flag = 4
learning_rate_flag = 2e-5 
num_train_epochs_flag = 3 
output_dir_flag = "/home1/fangzheng/project/bert/glue_data/aida_cut/output"

seq_length_smaller_50 = 0
seq_length_50_100 = 0
seq_length_100_128 = 0
seq_length_128_150 = 0
seq_length_150_200 = 0
seq_length_larger_200 = 0

## Required parameters
flags.DEFINE_string(
    "data_dir", data_dir_flag,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", bert_config_file_flag,
    "The config json file corresponding to the pre-trained BERT old_model. "
    "This specifies the old_model architecture.")

flags.DEFINE_string("task_name", task_name_flag, "The name of the task to train.")

flags.DEFINE_string("vocab_file", vocab_file_flag,
                    "The vocabulary file that the BERT old_model was trained on.")

flags.DEFINE_string(
    "output_dir", output_dir_flag,
    "The output directory where the old_model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", init_checkpoint_flag,
    "Initial checkpoint (usually from a pre-trained BERT old_model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", max_seq_length_flag,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", do_train_flag, "Whether to run training.")

flags.DEFINE_bool("do_eval", do_eval_flag, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", do_predict_flag,
    "Whether to run the old_model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", train_batch_size_flag, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", eval_batch_size_flag, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", test_batch_size_flag, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", learning_rate_flag, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", num_train_epochs_flag,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the old_model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_float(
    "margin", 0.3,
    "max marigin between positive and negative instances")

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None, group_id=None):
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
    self.label = label
    self.group_id = group_id


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               group_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.group_id = group_id
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
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines


class XnliProcessor(DataProcessor):
  """Processor for the XNLI data set."""

  def __init__(self):
    self.language = "zh"

  def get_train_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(
        os.path.join(data_dir, "multinli",
                     "multinli.train.%s.tsv" % self.language))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "train-%d" % (i)
      text_a = tokenization.convert_to_unicode(line[0])
      text_b = tokenization.convert_to_unicode(line[1])
      label = tokenization.convert_to_unicode(line[2])
      if label == tokenization.convert_to_unicode("contradictory"):
        label = tokenization.convert_to_unicode("contradiction")
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_dev_examples(self, data_dir):
    """See base class."""
    lines = self._read_tsv(os.path.join(data_dir, "xnli.dev.tsv"))
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "dev-%d" % (i)
      language = tokenization.convert_to_unicode(line[0])
      if language != tokenization.convert_to_unicode(self.language):
        continue
      text_a = tokenization.convert_to_unicode(line[6])
      text_b = tokenization.convert_to_unicode(line[7])
      label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]


class MnliProcessor(DataProcessor):
  """Processor for the MultiNLI data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
        "dev_matched")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
      text_a = tokenization.convert_to_unicode(line[8])
      text_b = tokenization.convert_to_unicode(line[9])
      if set_type == "test":
        label = "contradiction"
      else:
        label = tokenization.convert_to_unicode(line[-1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class MrpcProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        # self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
        self._read_tsv(os.path.join(data_dir, "msr_paraphrase_test.txt")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line[3])
      text_b = tokenization.convert_to_unicode(line[4])
      # if set_type == "test":
      #   label = "0"
      # else:
      #   label = tokenization.convert_to_unicode(line[0])
      label = tokenization.convert_to_unicode(line[0])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

class AidaProcessor(DataProcessor):
  """
  Processor for the Aida data (preprocessed)
  an example represents a group.
  """

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "aida_train_rank_format")), "train")
    # return self._create_examples(
    #     self._read_tsv(os.path.join(data_dir, "bert_train")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "aida_testA_rank_format")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "aida_testB_rank_format")), "test")

  def get_labels(self):
    """
    基于任务来设置label list
    """
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    example = []
    last_group_id = -1
    exit_one = 0
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      group_id = tokenization.convert_to_unicode(line[0])
      # if int(group_id)==802:
      #   print(1)
      label = tokenization.convert_to_unicode(line[1])
      # if int(label) == 1:
      #   if exit_one == 1 and group_id == last_group_id:
      #     if i<10000:
      #       print(i+1)
      #   else:
      #     pass
      text_a = tokenization.convert_to_unicode(line[2])
      text_b = tokenization.convert_to_unicode(line[3])
      instance = InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, group_id=group_id)
      if group_id == last_group_id or i == 0:
        example.append(instance)
      else:
        # new group
        examples.append(example)
        example = []
        example.append(instance)
        exit_one = int(label)
      last_group_id = group_id
    # examples: list of list, axis0 represents each group, axis1 represents each instance in the groups
    # for example in examples:
    #   exit_zero = 0
    #   for instance in example:
    #     if int(instance.label)==1:
    #       exit_zero=1
    #       break
    #   if exit_zero==0:
    #     print(instance.group_id)
    return examples

class ColaProcessor(DataProcessor):
  """Processor for the CoLA data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      # Only the test set has a header
      if set_type == "test" and i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      if set_type == "test":
        text_a = tokenization.convert_to_unicode(line[1])
        label = "0"
      else:
        text_a = tokenization.convert_to_unicode(line[3])
        label = tokenization.convert_to_unicode(line[1])
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


def convert_single_group(gp_index, group, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  # if isinstance(group, PaddingInputExample):
  #   # 因为没有padding，所以不存在这种情况
  #   return InputFeatures(
  #       input_ids=[0] * max_seq_length,
  #       input_mask=[0] * max_seq_length,
  #       segment_ids=[0] * max_seq_length,
  #       label_id=0,
  #       is_real_example=False)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i
  group_features = []

  for example in group:
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
    # it easier for the old_model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire old_model is fine-tuned.
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

    label_id = label_map[example.label]
    # if ex_index < 5:
    #   tf.logging.info("*** Example ***")
    #   tf.logging.info("guid: %s" % (example.guid))
    #   tf.logging.info("tokens: %s" % " ".join(
    #       [tokenization.printable_text(x) for x in tokens]))
    #   tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    #   tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    #   tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    #   tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

    # input_ids(list): the corresponding id of tokens in vocabulary
    # input_mask(list): whether the token is masked? in the fine-tune case values are all 1 execpt the padding examples
    # segment ids(list): the first sentence or second sentence
    # label_ids(integer): the label of example
    # is real example: whether the example is real or the fake padding example in the last batch 
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        group_id=int(example.group_id),
        is_real_example=True)
    group_features.append(feature)
  return group_features


def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    # 一个example代表一个group
    if ex_index % 1000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    group_features = convert_single_group(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    input_ids_list = []
    input_mask_list = []
    segment_ids_list = []
    label_ids_list = []
    group_id_list = []
    for feature in group_features:
      input_ids_list.append(create_int_feature(feature.input_ids))
      input_mask_list.append(create_int_feature(feature.input_mask))
      segment_ids_list.append(create_int_feature(feature.segment_ids))
      label_ids_list.append(create_int_feature([feature.label_id]))
      group_id_list.append(create_int_feature([feature.group_id]))

    feature_lists_dict = {}
    feature_lists_dict["input_ids"] = tf.train.FeatureList(feature = input_ids_list)
    feature_lists_dict["input_mask"] = tf.train.FeatureList(feature = input_mask_list)
    feature_lists_dict["segment_ids"] = tf.train.FeatureList(feature = segment_ids_list)
    feature_lists_dict["label_ids"] = tf.train.FeatureList(feature = label_ids_list)
    feature_lists_dict["group_ids"] = tf.train.FeatureList(feature=group_id_list)
    feature_lists = tf.train.FeatureLists(feature_list = feature_lists_dict)

    seq_example = tf.train.SequenceExample(context=None, feature_lists=feature_lists)
    writer.write(seq_example.SerializeToString())



    # features = collections.OrderedDict()
    # features["input_ids"] = create_int_feature(feature.input_ids)
    # features["input_mask"] = create_int_feature(feature.input_mask)
    # features["segment_ids"] = create_int_feature(feature.segment_ids)
    # features["label_ids"] = create_int_feature([feature.label_id])
    # features["is_real_example"] = create_int_feature(
    #     [int(feature.is_real_example)])

    # tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    # writer.write(tf_example.SerializeToString())

  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenSequenceFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenSequenceFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenSequenceFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenSequenceFeature([], tf.int64),
      "group_ids": tf.FixedLenSequenceFeature([], tf.int64)
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    _, seq_example = tf.parse_single_sequence_example(record, context_features=None,sequence_features=name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in seq_example.keys():
      t = seq_example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      seq_example[name] = t

    return seq_example

  def input_fn(params):
    """The actual input function."""
    # params: TPUEstimator内部构造的batch_size参数
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)
    d = d.map(lambda record:  _decode_record(record, name_to_features))

    padded_shapes = {
      "input_ids": [None, seq_length], 
      "input_mask": [None, seq_length],
      "segment_ids": [None, seq_length],
      "label_ids": [None],
      "group_ids": [None]
    }

    padded_values = {
      "input_ids": tf.constant(0, dtype=tf.int32), 
      "input_mask": tf.constant(0, dtype=tf.int32),
      "segment_ids": tf.constant(0, dtype=tf.int32),
      "label_ids": tf.constant(-1, dtype=tf.int32),
      "group_ids": tf.constant(-1, dtype=tf.int32)
    }
    d = d.padded_batch(batch_size=batch_size, padded_shapes=padded_shapes, 
      padding_values=padded_values, drop_remainder=True)

    # 对数据做预处理（转化成int32，适合TPU处理）
    # d = d.apply(
    #     tf.contrib.data.map_and_batch(
    #         lambda record: _decode_record(record, name_to_features),
    #         batch_size=batch_size,
    #         drop_remainder=drop_remainder))
    # print(d)
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
  """Creates a classification old_model.
  return: loss, per_example_loss, logits, probabilities
  """
  with tf.variable_scope("old_model", reuse=tf.AUTO_REUSE):

    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use old_model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()
    # token_level_output = old_model.get_sequence_output()
    # print(output_layer)
    # print(token_level_output)

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    # tf.get_variable_scope().reuse_variables()
    # gl_step = tf.get_variable("global_step", dtype=tf.int64)
    # print(gl_step)
    # print(gl_step.eval(tf.Session()))
    # print(1)

    with tf.variable_scope("loss"):
      if is_training:
        # I.e., 0.1 dropout
        output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

      logits = tf.matmul(output_layer, output_weights, transpose_b=True) # (N, 2)
      logits = tf.nn.bias_add(logits, output_bias)
      
      probabilities = tf.nn.softmax(logits, axis=-1) # (N, 2)
      log_probs = tf.nn.log_softmax(logits, axis=-1) # (N, 2)
      
      # labels (N)
      pos_indices = tf.where(tf.equal(labels, 1)) # (NP, 1)
      neg_indices = tf.where(tf.equal(labels, 0)) # (NN, 1)
      print(pos_indices)

      pos_metric = tf.gather_nd(probabilities, pos_indices) # (NP, 2)
      neg_metric = tf.gather_nd(probabilities, neg_indices) # (NN, 2)

      pos_one_hot = tf.constant([[0, 1]], dtype=tf.float32)
      pos_one_hot_labels = tf.tile(pos_one_hot, [tf.shape(pos_indices)[0], 1]) # (NP, 2)
      neg_one_hot_labels = tf.tile(pos_one_hot, [tf.shape(neg_indices)[0], 1]) # (NN, 2)

      pos_metric = tf.reduce_sum(pos_metric * pos_one_hot_labels, axis=-1) # (NP,)
      neg_metric = tf.reduce_sum(neg_metric * neg_one_hot_labels, axis=-1) # (NN,)

      # do the substraction
      pos_metric = tf.tile(tf.expand_dims(pos_metric, 1), [1, tf.shape(neg_indices)[0]]) # (NP, NN)
      neg_metric = tf.tile(tf.expand_dims(neg_metric, 0), [tf.shape(pos_indices)[0], 1]) # (NP, NN)
      delta = neg_metric - pos_metric # (NP, NN)

      loss = tf.reduce_mean(tf.nn.relu(FLAGS.margin + delta))

      indices_padded = tf.equal(labels, -1) # bool (N, )
      indices_not_padded = tf.where(tf.math.logical_not(indices_padded))
      labels_not_padded = tf.gather_nd(labels, indices_not_padded) # (N,2)
      probabilities_not_padded = tf.gather_nd(probabilities, indices_not_padded)[:, -1] # (N, 2)

      return loss, delta, tf.shape(delta), pos_metric, neg_metric, probabilities, labels


def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
    
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    group_ids = features["group_ids"] # (B, N, H)
    # print(group_ids[-1][0])
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    batch_size = params["batch_size"]
    total_loss = []
    pos_probabilities_not_padded_list = []
    batch_probabilities_list = []
    batch_labels_list = []
    for i in range(batch_size):
      (loss, delta, delta_shape, pos_metric, neg_metric, probabilities, labels) =\
        create_model(bert_config, is_training, input_ids[i, :, :], input_mask[i, :, :], 
        segment_ids[i, :, :], label_ids[i, :], num_labels, use_one_hot_embeddings)
      
      batch_probabilities_list.append(probabilities)
      batch_labels_list.append(labels)
      total_loss.append(loss)

    total_loss = tf.reduce_sum(total_loss)
    batch_probabilities = tf.stack(batch_probabilities_list)
    batch_labels = tf.stack(batch_labels_list)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
      hook_dict = {
        "loss": total_loss, 
        # "pos_metric": pos_metric,
        # "neg_metric": neg_metric, 
        # "delta": delta, 
        # "delta_shape": delta_shape,
        "group_id": group_ids
      }
      logging_hook = tf.train.LoggingTensorHook(hook_dict, every_n_iter=1)
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          training_hooks=[logging_hook],
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(total_loss, labels, probabilities):
        # labels_not_padded_list: (B, N), pos_probabilities_not_padded_list: (B, N)
        
        predictions = tf.argmax(probabilities, axis = -1) # (B, )
        labels = tf.argmax(labels, axis = -1) # (B, )
        # 再加个weight把 padded的例子去掉
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": total_loss,
        }

      eval_metrics = (metric_fn, [total_loss, batch_labels, batch_probabilities])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={
            "probabilities": batch_probabilities, # (B, N, 2)
            "labels": batch_labels,
            "group_ids": group_ids
            },
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
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_group(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "cola": ColaProcessor,
      "mnli": MnliProcessor,
      "mrpc": MrpcProcessor,
      "xnli": XnliProcessor,
      "aida": AidaProcessor,
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
    raise ValueError(
        "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT old_model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      log_step_count_steps=10,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    # format of train_examples: 
    # guid: 'train-1/2/3'
    # label: ground truth
    # text_a: sentence 1
    # text_b: sentence 2
    train_examples = processor.get_train_examples(FLAGS.data_dir)
    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    # 最后一个batch要drop掉 ，在file_based_convert_examples_to_features()函数中
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
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train_group_id.tf_record")
    # if the features have been built once before, this line of code can be commented
    # file_based_convert_examples_to_features(
    #     train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
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
      pass
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on. These do NOT count towards the metric (all tf.metrics
      # support a per-instance weight, and these get a weight of 0.0).

      # while len(eval_examples) % FLAGS.eval_batch_size != 0:
      #   eval_examples.append(PaddingInputExample())

    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

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
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
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

    predict_file = os.path.join(FLAGS.output_dir, "preidct.tf_record")
   
    # can be commented
    # file_based_convert_examples_to_features(predict_examples, label_list,
    #                                         FLAGS.max_seq_length, tokenizer,
    #                                         predict_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)
    print(result)

  # output_predict_file = os.path.join(FLAGS.output_dir, "test_results.txt")
  # with tf.gfile.GFile(output_predict_file, "w") as writer:
    # num_written_lines = 0
    # predict_label_list = []
    tf.logging.info("***** Predict results *****")
    correct_num = 0
    total_num = 0
    for (i, prediction) in enumerate(result):
      probabilities = prediction["probabilities"][:, 1] # (N,1)
      labels = prediction["labels"]
      index = len(labels)
      for i in range(len(labels)):
        if labels[i] == -1:
          index = i
          break
      # 把不是padding的元素取出来
      probabilities = probabilities[:index]
      labels = labels[:index]
      # 判断是否匹配
      prediction = np.argmax(probabilities)
      groud_truth = [np.argmax(labels)]
      if prediction==groud_truth:
        correct_num = correct_num + 1
      total_num = total_num+1
    print("accuracy:", correct_num/total_num)
      # if probabilities[0]>probabilities[1]:
      #   predict_label_list.append(0)
      # else:
      #   predict_label_list.append(1)
      # if i >= num_actual_predict_examples:
      #   break
      # output_line = "\t".join(
      #     str(class_probability)
      #     for class_probability in probabilities) + "\n"
      # writer.write(output_line)
      # num_written_lines += 1
  # assert num_written_lines == num_actual_predict_examples

    # compute the correct rate of prediction
    # correct_count = 0
    # for (index, example) in enumerate(predict_examples):
    #   if int(example.label) == predict_label_list[index]:
    #     correct_count = correct_count + 1
    # rate = correct_count / len(predict_label_list)
    # print("accurcy: %f" % rate)


if __name__ == "__main__":
  os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.app.run()

