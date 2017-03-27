#!/usr/bin/env python

import glob
import json
import numpy as np
import os
import re
import time
import tensorflow as tf

import AlphaGo.go as go
import AlphaGo.util as util

from AlphaGo.ai import ProbabilisticPolicyPlayer
from AlphaGo.models.tf_policy import CNNPolicy

# input flags
flags = tf.app.flags
flags.DEFINE_integer('num_games', 20, 'Number of games in batch.')
flags.DEFINE_integer("move_limit", 500, "Maximum number of moves per game.")

flags.DEFINE_float('policy_temperature', 0.67, 'Policy temperature.')
flags.DEFINE_float('gpu_memory_fraction', 0.15,
                   'config.per_process_gpu_memory_fraction for training session')

flags.DEFINE_string('learner', '/s3/logs',
                    'Directory where to be saved RL policy model.')
flags.DEFINE_string('opponent', '/s3/opponents/0',
                    'Directory where to be saved opponent model.')
flags.DEFINE_string('game_logdir', '/s3/test/games',
                    'Directory where to write game results.')

flags.DEFINE_boolean('random_choice_for_slpolicy', True,
                     'Randomly choice parameter from supervised policy privious iteration.')
flags.DEFINE_boolean('save_sgf', True,
                     'Save game state in sgf format.')

flags.DEFINE_boolean('verbose', True, '')

FLAGS = flags.FLAGS


def load_player(logdir):
    # See as migrated Keras weights if model.json exists.
    if os.path.exists(os.path.join(logdir, 'model.json')):
        # Get filter size from json
        data = json.load(open(os.path.join(logdir, 'model.json')))
        model = json.loads(data['keras_model'])
        filters = model['config'][0]['config']['nb_filter']        
        if FLAGS.random_choice_for_slpolicy:
            # We play games between the current policy network and
            # a randomly selected previous iteration of the policy network.
            checkpoint_dir = np.random.choice(glob.glob(os.path.join(logdir, '[0-9]*')))
            if FLAGS.verbose:
                print("Randomly selected previous iteration of the SL policy network. {}".format(checkpoint_dir))
        else:
            best_epoch = json.load(open(os.path.join(logdir, 'metadata.json')))['best_epoch']
            best_epoch = str(best_epoch).rjust(5, '0')
            checkpoint_dir = os.path.join(logdir, best_epoch)

        policy_net = CNNPolicy(checkpoint_dir=checkpoint_dir,
                               filters=filters)
    else:
        policy_net = CNNPolicy(checkpoint_dir=logdir)

    policy_net.init_graph(train=False)

    config = tf.ConfigProto(device_count={'GPU': 1})
    config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction

    policy_net.start_session(config)
    policy_net.load_model()

    player = ProbabilisticPolicyPlayer(policy_net,
                                       FLAGS.policy_temperature,
                                       move_limit=FLAGS.move_limit)
    return player


def playout(learner, opponent, num_games, step, test_count):
    board_size = learner.policy.bsize
    states = [go.GameState(size=board_size) for _ in range(num_games)]

    learner_won = [None] * num_games

    learner_color = [go.BLACK if i % 2 == 0 else go.WHITE for i in range(num_games)]
    white_states = states[1::2]
    moves = opponent.get_moves(white_states)
    for st, mv in zip(white_states, moves):
        st.do_move(mv)

    current = learner
    other = opponent

    idxs_to_unfinished_states = {i: states[i] for i in range(num_games)}

    while len(idxs_to_unfinished_states) > 0:
        moves = current.get_moves(idxs_to_unfinished_states.values())
        just_finished = []

        for (idx, state), mv in zip(idxs_to_unfinished_states.iteritems(), moves):
            state.do_move(mv)

            if state.is_end_of_game:
                learner_won[idx] = state.get_winner() == learner_color[idx]
                just_finished.append(idx)

        for idx in just_finished:
            del idxs_to_unfinished_states[idx]

        current, other = other, current

    wins = sum(state.get_winner() == color for (state, color) in zip(states, learner_color))

    if FLAGS.save_sgf:
        sgf_path = os.path.join(FLAGS.game_logdir, 'sgf')
        if not os.path.exists(sgf_path):
            os.mkdir(sgf_path)

        for game_idx, (state, color) in enumerate(zip(states, learner_color)):
            color_name = 'B' if color == go.BLACK else 'W'
            if state.get_winner() == color:
                result = 'win'
            elif state.get_winner() == -color:
                result = 'lose'
            else:
                result = 'tie'
            sgf_name = '_'.join([str(step).rjust(5, '0'),
                                 os.path.basename(FLAGS.opponent).rjust(5, '0'),
                                 str(test_count), str(game_idx),
                                 color_name, result]) + '.sgf'
            util.save_gamestate_to_sgf(state, sgf_path, sgf_name)

    return wins


def main(argv=None):
    log_name = os.path.join(FLAGS.game_logdir,
                            'win_ratio_for_opponent_' + os.path.basename(FLAGS.opponent))
    with open(log_name, 'w') as log:
        test_count = 0
        while True:
            if FLAGS.verbose:
                print("Loading learner. {}".format(FLAGS.learner))
            meta_files = [os.path.basename(meta)
                          for meta in glob.glob(os.path.join(FLAGS.learner, 'model.ckpt-*.meta'))]
            step = max(int(re.findall(r'model\.ckpt\-(\d+)\.meta', f)[0]) for f in meta_files)
            learner = load_player(FLAGS.learner)
            if FLAGS.verbose:
                print("Loading opponent. {}".format(FLAGS.opponent))
            opponent = load_player(FLAGS.opponent)

            start_time = time.time()
            wins = playout(learner, opponent, FLAGS.num_games, step, test_count)
            elapsed_sec = time.time() - start_time
            test_count += 1

            win_ratio = float(wins)*100/FLAGS.num_games
            print("Step {:d}.".format(step) +
                  " {:.2f}% ({:d}/{:d}) wins.".format(win_ratio, wins, FLAGS.num_games) +
                  " Elapsed: {:.2f} sec".format(float(elapsed_sec)))

            log.write('{:d}\t{:.2f}\t{:d}\t{:d}\t{:.2f}\n'.format(step,
                                                                  win_ratio,
                                                                  wins,
                                                                  FLAGS.num_games,
                                                                  elapsed_sec))
            log.flush()

            learner.policy.close_session()
            opponent.policy.close_session()


if __name__ == '__main__':
    tf.app.run(main=main)
