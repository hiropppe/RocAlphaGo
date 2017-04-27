#!/usr/bin/env python
# -*- coding: utf-8 -*-

import h5py as h5
import json
import numpy as np
import os
import tensorflow as tf
import threading
import time

from tqdm import tqdm

from AlphaGo import go
from AlphaGo.util import confirm, sgf_iter_states
from AlphaGo.models import tf_nn_util
from AlphaGo.models.rollout_policy import RolloutPolicy

# metdata file
FILE_METADATA = 'metadata_policy_supervised.json'

# shuffle files
FILE_TRAIN = 'shuffle_policy_train.npz'
FILE_TEST = 'shuffle_policy_test.npz'

flags = tf.app.flags
flags.DEFINE_string('train_data', '',
                    'h5 file of training data..')

flags.DEFINE_integer("board_size", 19, "")

flags.DEFINE_string('traindir', '/tmp/train',
                    'Directory where to save training working file.')
flags.DEFINE_string('logdir', '/tmp/train/logs',
                    'Directory where to save latest parameters.')

flags.DEFINE_integer("epoch_size", 10, "")
flags.DEFINE_integer("batch_size", 16, "")
flags.DEFINE_integer('decay_every', 8000000, '')
flags.DEFINE_float('test_size', .1, '')
flags.DEFINE_float('decay', .5, '')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')

flags.DEFINE_string('symmetries', 'all', '')

flags.DEFINE_float('gpu_memory_fraction', 0.15,
                   'config.per_process_gpu_memory_fraction for training session')

flags.DEFINE_boolean('log_device_placement', False, '')

flags.DEFINE_boolean('resume', False, '')
flags.DEFINE_boolean('overwrite', True, '')

flags.DEFINE_boolean('verbose', True, '')
FLAGS = flags.FLAGS

BOARD_MAX = FLAGS.board_size**2

TRANSFORMATION_INDICES = {
    "noop": 0,
    "rot90": 1,
    "rot180": 2,
    "rot270": 3,
    "fliplr": 4,
    "flipud": 5,
    "diag1": 6,
    "diag2": 7
}

BOARD_TRANSFORMATIONS = {
    0: lambda feature: feature,
    1: lambda feature: np.rot90(feature, 1),
    2: lambda feature: np.rot90(feature, 2),
    3: lambda feature: np.rot90(feature, 3),
    4: lambda feature: np.fliplr(feature),
    5: lambda feature: np.flipud(feature),
    6: lambda feature: np.transpose(feature),
    7: lambda feature: np.fliplr(np.rot90(feature, 1))
}


def one_hot_action(action, size=19):
    """Convert an (x,y) action into a size x size array of zeros with a 1 at x,y
    """

    categorical = np.zeros((size, size))
    categorical[action] = 1
    return categorical


class threading_shuffled_hdf5_batch_generator:
    """A generator of batches of training data for use with the fit_generator function
       of Keras. Data is accessed in the order of the given indices for shuffling.

       it is threading safe but not multiprocessing therefore only use it with
       pickle_safe=False when using multiple workers
    """
    def __init__(self, state_dataset, action_dataset, indices, batch_size, metadata=None,
                 validation=False):
        self.action_dataset = action_dataset
        self.state_dataset = state_dataset
        # lock used for multithreaded workers
        self.data_lock = threading.Lock()
        self.indices_max = len(indices)
        self.validation = validation
        self.batch_size = batch_size
        self.indices = indices

        if metadata is not None:
            self.metadata = metadata
        else:
            # create metadata object
            self.metadata = {
                "generator_seed": None,
                "generator_sample": 0
            }

        # shuffle indices
        # when restarting generator_seed and generator_batch will
        # reset generator to the same point as before
        self.shuffle_indices(self.metadata['generator_seed'], self.metadata['generator_sample'])

    def __iter__(self):
        return self

    def shuffle_indices(self, seed=None, idx=0):
        # set generator_sample to idx
        self.metadata['generator_sample'] = idx

        # check if seed is provided or generate random
        if seed is None:
            # create random seed
            self.metadata['generator_seed'] = np.random.random_integers(4294967295)

        # feed numpy.random with seed in order to continue with certain batch
        np.random.seed(self.metadata['generator_seed'])
        # shuffle indices according to seed
        if not self.validation:
            np.random.shuffle(self.indices)

    def next_indice(self):
        # use lock to prevent double hdf5 acces and incorrect generator_sample increment
        with self.data_lock:

            # get next training sample
            training_sample = self.indices[self.metadata['generator_sample'], :]
            # get state
            state = self.state_dataset[training_sample[0]]
            # get action
            # must be cast to a tuple so that it is interpreted as (x,y) not [(x,:), (y,:)]
            action = tuple(self.action_dataset[training_sample[0]])

            # increment generator_sample
            self.metadata['generator_sample'] += 1
            # shuffle indices when all have been used
            if self.metadata['generator_sample'] >= self.indices_max:
                self.shuffle_indices()

            # return state, action and transformation
            return state, action, training_sample[1]

    def next(self):
        state_batch_shape = (self.batch_size,) + self.state_dataset.shape[1:]
        game_size = state_batch_shape[-1]
        Xbatch = np.zeros(state_batch_shape)
        Ybatch = np.zeros((self.batch_size, game_size * game_size))

        for batch_idx in xrange(self.batch_size):
            state, action, transformation = self.next_indice()

            # get rotation symmetry belonging to state
            transform = BOARD_TRANSFORMATIONS[transformation]

            # get state from dataset and transform it.
            # loop comprehension is used so that the transformation acts on the
            # 3rd and 4th dimensions
            state_transform = np.array([transform(plane) for plane in state])
            action_transform = transform(one_hot_action(action, game_size))

            Xbatch[batch_idx] = state_transform
            Ybatch[batch_idx] = action_transform.flatten()

        Xbatch = Xbatch.transpose(0, 2, 3, 1)

        return (Xbatch, Ybatch)


def load_indices_from_file(shuffle_file):
    # load indices from shuffle_file
    with open(shuffle_file, "r") as f:
        indices = np.load(f)

    return indices


def save_indices_to_file(shuffle_file, indices):
    # save indices to shuffle_file
    with open(shuffle_file, "w") as f:
        np.save(f, indices)


def remove_unused_symmetries(indices, symmetries):
    # remove all rows with a symmetry not in symmetries
    remove = []

    # find all rows with incorrect symmetries
    for row in range(len(indices)):
        if not indices[row][1] in symmetries:
            remove.append(row)

    # remove rows and return new array
    return np.delete(indices, remove, 0)


def create_and_save_shuffle_indices(n_total_data_size,
                                    test_size,
                                    shuffle_file_train,
                                    shuffle_file_test):
    """ create an array with all unique state and symmetry pairs,
        calculate test/validation/training set sizes,
        seperate those sets and save them to seperate files.
    """

    symmetries = TRANSFORMATION_INDICES.values()

    # Create an array with a unique row for each combination of a training example
    # and a symmetry.
    # shuffle_indices[i][0] is an index into training examples,
    # shuffle_indices[i][1] is the index (from 0 to 7) of the symmetry transformation to apply
    shuffle_indices = np.empty(shape=[n_total_data_size * len(symmetries), 2], dtype=int)
    for dataset_idx in range(n_total_data_size):
        for symmetry_idx in range(len(symmetries)):
            shuffle_indices[dataset_idx * len(symmetries) + symmetry_idx][0] = dataset_idx
            shuffle_indices[dataset_idx * len(symmetries) + symmetry_idx][1] = symmetries[symmetry_idx]

    # shuffle rows without affecting x,y pairs
    np.random.shuffle(shuffle_indices)

    # test set size
    n_test_data = int(test_size * len(shuffle_indices))

    # train set size
    n_train_data = len(shuffle_indices) - n_test_data

    # create training set and save to file shuffle_file_train
    train_indices = shuffle_indices[0:n_train_data]
    save_indices_to_file(shuffle_file_train, train_indices)

    # create test set and save to file shuffle_file_test
    val_indices = shuffle_indices[n_train_data:n_train_data + n_test_data]
    save_indices_to_file(shuffle_file_test, val_indices)


def load_train_val_test_indices(arg_symmetries, dataset_length, batch_size):
    """Load indices from .npz files
       Remove unwanted symmerties
       Make Train set dividable by batch_size
       Return train/val/test set
    """
    # shuffle file locations for train/validation/test set
    shuffle_file_train = os.path.join(FLAGS.traindir, FILE_TRAIN)
    shuffle_file_test = os.path.join(FLAGS.traindir, FILE_TEST)

    # load from .npz files
    train_indices = load_indices_from_file(shuffle_file_train)
    test_indices = load_indices_from_file(shuffle_file_test)

    # used symmetries
    if arg_symmetries == "all":
        # add all symmetries
        symmetries = TRANSFORMATION_INDICES.values()
    elif arg_symmetries == "none":
        # only add standart orientation
        symmetries = [TRANSFORMATION_INDICES["noop"]]
    else:
        # add specified symmetries
        symmetries = [TRANSFORMATION_INDICES[name] for name in arg_symmetries.strip().split(",")]

    if FLAGS.verbose:
        print("Used symmetries: " + arg_symmetries)

    # remove symmetries not used during current run
    if len(symmetries) != len(TRANSFORMATION_INDICES):
        train_indices = remove_unused_symmetries(train_indices, symmetries)
        test_indices = remove_unused_symmetries(test_indices, symmetries)

    # Need to make sure training data is dividable by minibatch size or get
    # warning mentioning accuracy from keras
    if len(train_indices) % batch_size != 0:
        # remove first len(train_indices) % args.minibatch rows
        train_indices = np.delete(train_indices, [row for row in range(len(train_indices)
                                                  % batch_size)], 0)

    if FLAGS.verbose:
        print("dataset loaded")
        print("\t%d total positions" % dataset_length)
        print("\t%d total samples" % (dataset_length * len(symmetries)))
        print("\t%d total samples check" % (len(train_indices) + len(test_indices)))
        print("\t%d training samples" % len(train_indices))
        print("\t%d test samples" % len(test_indices))

    return train_indices, test_indices


def set_training_settings(metadata, dataset_length):
    """ save all args to metadata,
        check if critical settings have been changed and prompt user about it.
        create new shuffle files if needed.
    """

    # shuffle file locations for train/validation/test set
    shuffle_file_train = os.path.join(FLAGS.traindir, FILE_TRAIN)
    shuffle_file_test = os.path.join(FLAGS.traindir, FILE_TEST)

    # determine if new shuffle files have to be created
    save_new_shuffle_indices = not FLAGS.resume

    # save current symmetries to metadata
    metadata["symmetries"] = FLAGS.symmetries

    if FLAGS.resume:
        if FLAGS.learning_rate is not None and metadata["learning_rate"] != FLAGS.learning_rate:
            print("Setting a new --learning-rate might result in a different learning rate, restarting training might be advisable.")  # noqa: E501
            if FLAGS.override or confirm("Are you sure you want to use new learning rate setting?", False):  # noqa: E501
                metadata["learning_rate"] = FLAGS.learning_rate

        if FLAGS.decay is not None and metadata["decay"] != FLAGS.decay:
            print("Setting a new --decay might result in a different learning rate, restarting training might be advisable.")  # noqa: E501
            if FLAGS.override or confirm("Are you sure you want to use new decay setting?", False):  # noqa: E501
                metadata["decay"] = FLAGS.decay

        if FLAGS.decay_every is not None and metadata["decay_every"] != FLAGS.decay_every:
            print("Setting a new --decay-every might result in a different learning rate, restarting training might be advisable.")  # noqa: E501
            if FLAGS.override or confirm("Are you sure you want to use new decay every setting?", False):  # noqa: E501
                metadata["decay_every"] = FLAGS.decay_every

        if FLAGS.batch_size is not None and metadata["batch_size"] != FLAGS.batch_size:
            print("current_batch will be recalculated, restarting training might be advisable.")
            if FLAGS.override or confirm("Are you sure you want to use new batch_size setting?", False):  # noqa: E501
                metadata["current_batch"] = int((metadata["current_batch"] *
                                                metadata["batch_size"]) / FLAGS.batch_size)
                metadata["batch_size"] = FLAGS.batch_size

        if FLAGS.epoch_size is not None and metadata["epoch_size"] != FLAGS.epoch_size:
            metadata["epoch_size"] = FLAGS.epoch_size

        if FLAGS.test_size is not None and metadata["test_size"] != FLAGS.test_size:
            metadata["test_size"] = FLAGS.test_size

        # check if shuffle files exist and data file is the same
        if not os.path.exists(shuffle_file_train) or \
           not os.path.exists(shuffle_file_test) or \
           metadata["training_data"] != FLAGS.train_data or \
           metadata["available_states"] != dataset_length:
            # shuffle files do not exist or training data file has changed
            print("WARNING! shuffle files have to be recreated.")
            save_new_shuffle_indices = True
    else:
        # save all argument or default settings to metadata

        metadata["epoch_size"] = FLAGS.epoch_size
        metadata["batch_size"] = FLAGS.batch_size
        metadata["learning_rate"] = FLAGS.learning_rate
        metadata["decay"] = FLAGS.decay
        metadata["decay_every"] = FLAGS.decay_every
        metadata["test_size"] = FLAGS.test_size

    # create and save new shuffle indices if needed
    if save_new_shuffle_indices:
        # create and save new shuffle indices to file
        create_and_save_shuffle_indices(
                dataset_length, metadata["test_size"],
                shuffle_file_train, shuffle_file_test)

        # save total amount of states to metadata
        metadata["available_states"] = dataset_length

        # save training data file name to metadata
        metadata["training_data"] = FLAGS.train_data

        if FLAGS.verbose:
            print("created new data shuffling indices")


def placeholder_inputs(n_features):
    statesholder = tf.placeholder(tf.float32,
                                  shape=(None, FLAGS.board_size, FLAGS.board_size, n_features))
    actionsholder = tf.placeholder(tf.float32,
                                   shape=(None, BOARD_MAX))
    return statesholder, actionsholder


def train(metadata, meta_file, n_features):

    # features of training data
    dataset = h5.File(metadata["training_data"])

    # get train/validation/test indices
    train_indices, test_indices \
        = load_train_val_test_indices(metadata['symmetries'], len(dataset["states"]),
                                      metadata["batch_size"])

    # create dataset generators
    train_data_generator = threading_shuffled_hdf5_batch_generator(
        dataset["states"],
        dataset["actions"],
        train_indices,
        metadata["batch_size"],
        metadata)
    test_data_generator = threading_shuffled_hdf5_batch_generator(
        dataset["states"],
        dataset["actions"],
        test_indices,
        metadata["batch_size"],
        validation=True)

    if FLAGS.verbose:
        print("STARTING TRAINING")

    # check that remaining epoch > 0
    if metadata["epoch_size"] <= len(metadata["epoch_logs"]):
        raise ValueError("No more epochs to train!")

    with tf.Graph().as_default():
        policy = RolloutPolicy(n_features, FLAGS.board_size, FLAGS.logdir)

        states_pl, actions_pl = placeholder_inputs(n_features)

        logits = policy.inference(states_pl, n_features)

        loss = policy.loss(logits, actions_pl)

        probs = policy.softmax(logits)

        accuracy = policy.accuracy(probs, actions_pl)

        train_op = policy.train(loss, metadata['learning_rate'])

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement,
                                device_count={'GPU': 1})
        sess = tf.Session(config=config)

        sess.run(init)

        step = 0
        loss_buffer, acc_buffer = [], []
        n_epoch = metadata["epoch_size"] - len(metadata["epoch_logs"])
        for epoch in range(n_epoch):
            print('Starting Epoch: {:d}'.format(epoch))
            for (states, actions) in train_data_generator:

                feed_dict = {
                    states_pl: states,
                    actions_pl: actions,
                }

                start = time.time()

                _, loss_value, acc_value = sess.run([train_op, loss, accuracy],
                                                    feed_dict=feed_dict)

                duration = time.time() - start

                loss_buffer.append(loss_value)
                acc_buffer.append(acc_value)

                if (step+1) % 100 == 0:
                    start = time.time()
                    for s in states:
                        sess.run(logits, feed_dict={states_pl: np.expand_dims(s, axis=0)})
                    avg_predict_speed = (time.time() - start) * 1000 / states.shape[0]
                    print('Step {:d}: loss = {:.4f} acc = {:.4f} ({:.3f} sec)'.format(
                            step+1, np.mean(loss_buffer), np.mean(acc_buffer), duration))
                    print('Prediction Speed {:.3f} ms'.format(avg_predict_speed))
                    del loss_buffer[:]
                    del acc_buffer[:]
                step += 1


def start_training():
    if not os.path.exists(FLAGS.traindir):
        os.makedirs(FLAGS.traindir)
    if not os.path.exists(FLAGS.logdir):
        os.makedirs(FLAGS.logdir)

    # metadata json file location
    meta_file = os.path.join(FLAGS.logdir, FILE_METADATA)

    # create or load metadata from json file
    if FLAGS.resume and os.path.exists(meta_file):
        # load metadata
        with open(meta_file, "r") as f:
            metadata = json.load(f)

        if FLAGS.verbose:
            print("previous metadata loaded: %d epochs. new epochs will be appended." %
                  len(metadata["epoch_logs"]))
    else:
        # create new metadata
        metadata = {
            "epoch_logs": [],
            "current_batch": 0,
            "current_epoch": 0,
            "best_epoch": 0,
            "generator_seed": None,
            "generator_sample": 0
        }

        if FLAGS.verbose:
            print("starting with empty metadata")

    # features of training data
    dataset = h5.File(FLAGS.train_data)

    # set all settings: default, from args or from metadata
    # generate new shuffle files if needed
    set_training_settings(metadata, len(dataset['states']))

    # start training
    train(metadata, meta_file, dataset['states'][0].shape[0])


def main(argv=None):
    start_training()


if __name__ == '__main__':
    tf.app.run(main=main)
