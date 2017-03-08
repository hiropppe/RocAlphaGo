import copy
import h5py
import numpy as np
import os
import sys
import tensorflow as tf
import traceback
import warnings

from AlphaGo.ai import ProbabilisticPolicyPlayer
from AlphaGo.go import GameState
from AlphaGo.models.tf_policy import CNNPolicy
from AlphaGo.preprocessing.preprocessing import Preprocess

from tqdm import tqdm


# input flags
flags = tf.app.flags
flags.DEFINE_string('sl_policy_logdir', '/tmp/logs/sl_policy/',
                    'Directory where to load SL policy checkpoint files')
flags.DEFINE_string('rl_policy_logdir', '/tmp/logs/rl_policy/',
                    'Directory where to load SL policy checkpoint files')
flags.DEFINE_string('outfile', '/tmp/state_winner.hdf5',
                    'Path to where the training pairs will be saved.')

flags.DEFINE_float('policy_temperature', 0.67,
                   'Policy temperature.')
flags.DEFINE_float('gpu_memory_fraction', 0.15,
                   'config.per_process_gpu_memory_fraction for each policy session')

flags.DEFINE_integer("num_batches", 2,
                     "Number of batches to generate. ")
flags.DEFINE_integer("batch_size", 2,
                     "Number of games to run in parallel.")
flags.DEFINE_integer("move-limit", 500,
                     "Maximum number of moves per game.")

flags.DEFINE_boolean('log_device_placement', False, '')
flags.DEFINE_boolean('verbose', True, '')

FLAGS = flags.FLAGS


def load_player(logdir):
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction

    policy = CNNPolicy(checkpoint_dir=logdir)
    policy.init_graph(train=False)
    policy.start_session(config)
    policy.load_model()

    player = ProbabilisticPolicyPlayer(policy,
                                       FLAGS.policy_temperature,
                                       move_limit=FLAGS.move_limit)
    return player


def play_batch(player_RL, player_SL, batch_size, preprocessor):
    """Play a batch of games in parallel and return one training pair from each game.

    As described in Silver et al, the method for generating value net training data is as follows:

    * pick a number between 1 and 450
    * use the supervised-learning policy to play a game against itself up to that number of moves.
    * now go off-policy and pick a totally random move
    * play out the rest of the game with the reinforcement-learning policy
    * save the state that occurred *right after* the random move, and the end result of the game, as the training pair
    """

    def do_move(states, moves):
        for st, mv in zip(states, moves):
            if not st.is_end_of_game:
                # Only do more moves if not end of game already
                st.do_move(mv)
        return states

    def do_rand_move(states, player):
        """Do a uniform-random move over legal moves and record info for
        training. Only gets called once per game.
        """
        colors = [st.current_player for st in states]  # Record player color
        legal_moves = [st.get_legal_moves() for st in states]
        rand_moves = [lm[np.random.choice(len(lm))] for lm in legal_moves]
        states = do_move(states, rand_moves)
        eval_states = [st.copy() for st in states]  # For later 1hot preprocessing
        return eval_states, colors, states

    def convert(eval_states, preprocessor):
        """Convert states to 1-hot and concatenate. states's are game state objects.
        """
        states = np.concatenate(
            [preprocessor.state_to_tensor(state) for state in eval_states], axis=0)
        return states

    # Lists of game training pairs (1-hot)
    player = player_SL
    states = [GameState() for i in xrange(batch_size)]
    # Randomly choose turn to play uniform random. Move prior will be from SL
    # policy. Moves after will be from RL policy.
    i_rand_move = np.random.choice(range(450))
    eval_states = None
    winners = None
    turn = 0
    while True:
        # Do moves (black)
        if turn == i_rand_move:
            # Make random move, then switch from SL to RL policy
            eval_states, colors, states = do_rand_move(states, player)
            player = player_RL
        else:
            # Get moves (batch)
            moves_black = player.get_moves(states)
            # Do moves (black)
            states = do_move(states, moves_black)
        turn += 1
        # Do moves (white)
        if turn == i_rand_move:
            # Make random move, then switch from SL to RL policy
            eval_states, colors, states = do_rand_move(states, player)
            player = player_RL
        else:
            moves_white = player.get_moves(states)
            states = do_move(states, moves_white)
        turn += 1
        # If all games have ended, we're done. Get winners.
        done = [st.is_end_of_game or len(st.history) > 500 for st in states]
        if all(done):
            break

    if eval_states:
        # Concatenate training examples
        eval_states = convert(eval_states, preprocessor)
        winners = np.array([st.get_winner() for st in states]).reshape(batch_size, 1)
        return eval_states, winners
    else:
        # All game have ended before sampling
        return None, None


def generate_training_data(player_SL, player_RL):
    assert player_SL.policy.preprocessor.feature_list == player_RL.policy.preprocessor.feature_list, \
        'Features does not match.'
    assert player_SL.policy.preprocessor.output_dim == player_RL.policy.preprocessor.output_dim, \
        'Feature depth does not match.'

    bd_size = player_SL.policy.bsize
    # Define preprocessor for value network.
    value_features = copy.copy(player_SL.policy.preprocessor.feature_list)
    value_features.append('color')
    value_preprocessor = Preprocess(value_features)
    n_features = value_preprocessor.output_dim

    tmp_file = os.path.join(os.path.dirname(FLAGS.outfile),
                            ".tmp." + os.path.basename(FLAGS.outfile))
    h5f = h5py.File(tmp_file, 'w')
    try:
        h5_states = h5f.require_dataset(
            'states',
            dtype=np.uint8,
            shape=(1, n_features, bd_size, bd_size),
            maxshape=(None, n_features, bd_size, bd_size),
            exact=False,
            chunks=(1024, n_features, bd_size, bd_size),
            compression="lzf")
        h5_winners = h5f.require_dataset(
            'winners',
            dtype=np.int8,
            shape=(1, 1),
            maxshape=(None, 1),
            exact=False,
            chunks=(1024, 1),
            compression="lzf")

        next_idx = 0
        for n in tqdm(range(FLAGS.num_batches)):
            states, winners = play_batch(player_SL,
                                         player_RL,
                                         FLAGS.batch_size,
                                         value_preprocessor)
            if states is not None:
                try:
                    h5_states.resize((next_idx + FLAGS.batch_size, n_features, bd_size, bd_size))
                    h5_winners.resize((next_idx + FLAGS.batch_size, 1))
                    h5_states[next_idx:] = states
                    h5_winners[next_idx:] = winners
                    next_idx += FLAGS.batch_size
                except Exception as e:
                    warnings.warn("Unknown error occured during batch save to HDF5 file: {}".format(FLAGS.outfile))
                    raise e
            else:
                warnings.warn("All game have ended before sampling")

    except Exception as e:
        sys.stderr.write(traceback.format_exc())
        os.remove(tmp_file)
        raise e

    h5f.close()
    os.rename(tmp_file, FLAGS.outfile)


def main(argv=None):
    player_SL = load_player(FLAGS.sl_policy_logdir)
    player_RL = load_player(FLAGS.rl_policy_logdir)

    generate_training_data(player_SL, player_RL)


if __name__ == '__main__':
    tf.app.run(main=main)
