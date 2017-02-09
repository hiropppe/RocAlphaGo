#!/usr/bin/env python

import json
import os
import time
import h5py as h5
import numpy as np
import tensorflow as tf

import AlphaGo.go as go

from AlphaGo.training import rl_policy

from shutil import copyfile

from AlphaGo.ai import GreedyPolicyPlayer
from AlphaGo.models.policy import CNNPolicy
from AlphaGo.util import flatten_idx


flags = tf.app.flags
flags.DEFINE_integer('num_games', 1, 'Game batch size.')
flags.DEFINE_integer('max_steps', 10, 'Max step size.')
flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')
flags.DEFINE_float('policy_temperature', 0.67, 'Policy temperature.')

flags.DEFINE_integer('save_every', 2, 'Save policy as a new opponent every n batch.')
flags.DEFINE_integer('summarize_every', 1, 'Write summary every n batch.')

flags.DEFINE_string('model_json', './model/policy.json', 'SL Policy model json')
flags.DEFINE_string('initial_weights', './model/policy.hdf5', 'SL Policy weights hdf5')

flags.DEFINE_string('train_directory', './logs', 'Directory where to write train logs')

flags.DEFINE_boolean('resume', False, '')
flags.DEFINE_boolean('verbose', True, '')
FLAGS = flags.FLAGS


def game_generator(player, opponent, metadata):
    for step in xrange(FLAGS.max_steps):
        opponent_weights = np.random.choice(metadata["opponents"])
        opponent_weights = os.path.join(FLAGS.train_directory, opponent_weights)
        opponent.policy.model.load_weights(opponent_weights)
        if FLAGS.verbose:
            print("Batch {}\tsampled opponent is {}".format(step, opponent_weights))

        win_ratio, states, actions, rewards = playout_n(player, opponent, FLAGS.num_games)

        yield (win_ratio, states, actions, rewards)


def _make_training_pair(st, mv, preprocessor):
    st_tensor = preprocessor.state_to_tensor(st)
    # Transpose input(state) dimention ordering.
    # TF uses the last dimension as channel dimension,
    # K input shape: (samples, input_depth, row, cols)
    # TF input shape: (samples, rows, cols, input_depth)
    st_tensor = st_tensor.transpose((0, 2, 3, 1))
    mv_tensor = np.zeros((1, st.size * st.size))
    mv_tensor[(0, flatten_idx(mv, st.size))] = 1
    return (st_tensor, mv_tensor)


def zero_baseline(state):
    return 0


def playout_n(learner, opponent, num_games, value=zero_baseline):
    board_size = learner.policy.model.input_shape[-1]
    states = [go.GameState(size=board_size) for _ in range(num_games)]

    state_batch, action_batch, reward_batch = [], [], []

    game_states = [[] for _ in range(num_games)]
    game_actions = [[] for _ in range(num_games)]

    learner_won = [None] * num_games

    learner_color = [go.BLACK if i % 2 == 0 else go.WHITE for i in range(num_games)]
    odd_states = states[1::2]
    moves = opponent.get_moves(odd_states)
    for st, mv in zip(odd_states, moves):
        st.do_move(mv)

    current = learner
    other = opponent
    idxs_to_unfinished_states = {i: states[i] for i in range(num_games)}

    while len(idxs_to_unfinished_states) > 0:
        moves = current.get_moves(idxs_to_unfinished_states.values())
        just_finished = []

        for (idx, state), mv in zip(idxs_to_unfinished_states.iteritems(), moves):
            is_learnable = current is learner and mv is not go.PASS_MOVE
            if is_learnable:
                (st_tensor, mv_tensor) = _make_training_pair(state, mv, learner.policy.preprocessor)
                game_states[idx].append(st_tensor)
                game_actions[idx].append(mv_tensor)

            state.do_move(mv)

            if state.is_end_of_game:
                learner_won[idx] = state.get_winner() == learner_color[idx]
                just_finished.append(idx)

                state_batch.append(np.concatenate(game_states[idx], axis=0))
                action_batch.append(np.concatenate(game_actions[idx], axis=0))
                z = 1 if learner_won[idx] else -1
                reward_batch.append(np.array([z - value(state) for state in game_states[idx]]))

        for idx in just_finished:
            del idxs_to_unfinished_states[idx]

        current, other = other, current

    wins = sum(state.get_winner() == pc for (state, pc) in zip(states, learner_color))

    state_batch = np.concatenate(state_batch, axis=0)
    action_batch = np.concatenate(action_batch, axis=0)
    reward_batch = np.concatenate(reward_batch)

    return float(wins) / num_games, state_batch, action_batch, reward_batch


def main(argv=None):
    if not FLAGS.resume:
        if not os.path.exists(FLAGS.train_directory):
            if FLAGS.verbose:
                print("creating output directory {}".format(FLAGS.train_directory))
            os.makedirs(FLAGS.train_directory)

        ZEROTH_FILE = "weights.00000.hdf5"

        initial_weights = os.path.join(FLAGS.train_directory, ZEROTH_FILE)
        copyfile(FLAGS.initial_weights, initial_weights)

        if FLAGS.verbose:
            print("copied {} to {}".format(FLAGS.initial_weights,
                                           os.path.join(FLAGS.train_directory, ZEROTH_FILE)))

        metadata = {
            "model_file": FLAGS.model_json,
            "init_weights": os.path.basename(initial_weights),
            "learning_rate": FLAGS.learning_rate,
            "temperature": FLAGS.policy_temperature,
            "game_batch": FLAGS.num_games,
            "opponents": [ZEROTH_FILE],
            "win_ratio": {}  # map from player to tuple of (opponent, win ratio) Useful for
                             # validating in lieu of 'accuracy/loss'
        }
    else:
        if not os.path.exists(os.path.join(FLAGS.train_directory, "metadata.json")):
            raise ValueError("Cannot resume without existing output directory")

        with open(os.path.join(FLAGS.train_directory, "metadata.json"), "r") as f:
            metadata = json.load(f)

        initial_weights = os.path.join(FLAGS.train_directory, FLAGS.initial_weights)

        if not os.path.exists(initial_weights):
            raise ValueError("Cannot resume; weights {} do not exist".format(initial_weights))
        elif FLAGS.verbose:
            print("Resuming with weights {}".format(initial_weights))

    # RL policy training on TF
    weights = h5.File(initial_weights)
    model_weights = weights['model_weights']

    with tf.Graph().as_default():
        # Define placeholder
        states = tf.placeholder(tf.float32,
                                shape=(None,
                                       rl_policy.BOARD_SIZE,
                                       rl_policy.BOARD_SIZE,
                                       rl_policy.NUM_CHANNELS))
        actions = tf.placeholder(tf.float32,
                                 shape=(None, rl_policy.BOARD_SIZE**2))
        rewards = tf.placeholder(tf.float32, shape=(None,))

        # Forward
        probs = rl_policy.inference(states, model_weights)

        # Compute loss
        loss = rl_policy.loss(probs, actions, rewards)

        # Check accuracy for actual actions
        acc = rl_policy.accuracy(probs, actions)

        # Train
        train = rl_policy.train(loss, FLAGS.learning_rate, rewards)

        # Build the summary Tensor based on the TF collection of Summaries.
        summary = tf.summary.merge_all()

        init = tf.global_variables_initializer()

        saver = tf.train.Saver()

        sess = tf.Session()

        summary_writer = tf.summary.FileWriter(FLAGS.train_directory, sess.graph)

        sess.run(init)

        # Prepare playout data generator
        player_weights = opponent_weights = initial_weights

        player_policy = CNNPolicy.load_model(FLAGS.model_json)
        player = GreedyPolicyPlayer(player_policy, move_limit=500)
        player_policy.model.load_weights(player_weights)

        opponent_policy = CNNPolicy.load_model(FLAGS.model_json)
        opponent = GreedyPolicyPlayer(opponent_policy, move_limit=500)

        train_game_generator = game_generator(player, opponent, metadata)

        step = 0
        for (win_ratio, state_batch, action_batch, reward_batch) in train_game_generator:
            start_time = time.time()

            feed_dict = {
                states: state_batch,
                actions: action_batch,
                rewards: reward_batch
            }

            """
            # Debug computation results
            clip_probs = tf.clip_by_value(probs, 1e-07, 1.0)
            good_probs = tf.reduce_sum(tf.mul(clip_probs, actions), reduction_indices=[1])
            good_probs_value = sess.run(good_probs, feed_dict)

            log_good_probs = tf.log(good_probs)
            log_good_probs_value = sess.run(log_good_probs, feed_dict)

            log_good_probs_mean = tf.reduce_mean(log_good_probs)
            log_good_probs_mean_value = sess.run(log_good_probs_mean, feed_dict)
            """
            import pdb; pdb.set_trace()
            _, loss_value, acc_value = sess.run([train, loss, acc], feed_dict)

            duration = time.time() - start_time

            # Update win_ratio
            metadata["win_ratio"][player_weights] = (opponent_weights, win_ratio)
            with open(os.path.join(FLAGS.train_directory, "metadata.json"), "w") as f:
                json.dump(metadata, f, sort_keys=True, indent=2)

            # Save Keras model weights then reload player weights
            player_weights = rl_policy.save_keras_weights(sess, step + 1)
            player_weights = os.path.join(FLAGS.train_directory, player_weights)
            player.policy.model.load_weights(player_weights)

            # Write the summaries and print an overview fairly often.
            if step % FLAGS.summarize_every == 0:
                # Print status to stdout.
                print('Step %d: loss = %.2f acc = %.2f (%.3f sec)' % (step, loss_value, acc_value, duration))
                # Update the events file.
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, step)
                summary_writer.flush()

            # Write checkpoint file
            if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
                checkpoint_file = os.path.join(FLAGS.train_directory, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=step)    

            if (step + 1) == FLAGS.max_steps:
                break

            step += 1

    # close h5
    weights.close()


if __name__ == '__main__':
    tf.app.run(main=main)
