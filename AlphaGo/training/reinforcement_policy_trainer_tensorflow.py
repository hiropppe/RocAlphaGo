#!/usr/bin/env python

import glob
import os
import time
import numpy as np
import tensorflow as tf

import AlphaGo.go as go

from shutil import copyfile

from AlphaGo.ai import GreedyPolicyPlayer, ProbabilisticPolicyPlayer
from AlphaGo.util import flatten_idx
from AlphaGo.models.tf_policy import CNNPolicy


# input flags
flags = tf.app.flags
flags.DEFINE_integer('num_games', 3, 'Number of games in batch.')
flags.DEFINE_integer('max_steps', 10000, 'Number of batches to run.')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
flags.DEFINE_float('policy_temperature', 0.67, 'Policy temperature.')
flags.DEFINE_float('gpu_memory_fraction', 0.25, 'config.per_process_gpu_memory_fraction.')

flags.DEFINE_integer('save_every', 500, 'Save policy as a new opponent every n batch.')
flags.DEFINE_integer('summarize_every', 5, 'Write summary every n batch.')

flags.DEFINE_string('sl_logdir', './SL_logs', 'Directory where to write SL policy train logs')
flags.DEFINE_string('rl_logdir', './RL_logs', 'Directory where to write RL policy train logs')
flags.DEFINE_string('opponent_pool', './RL_opponents')

flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
flags.DEFINE_integer("task_index", 0, "Index of task within the job")
flags.DEFINE_integer('checkpoint', 100, 'Interval steps to save checkpoint.')

flags.DEFINE_boolean('resume', False, '')
flags.DEFINE_boolean('verbose', True, '')
FLAGS = flags.FLAGS


# cluster specification
parameter_servers = ["ps0:2222"]
workers = ["worker0:2222",
           "worker1:2222"]


def get_game_batch(step, learner_policy):
    learner = ProbabilisticPolicyPlayer(learner_policy,
                                        tenperature=FLAGS.policy_temperature,
                                        move_limit=500)

    # sampling opponent from pool
    opponent_policy_logdir = np.random.choice(glob.glob(os.path.join(FLAGS.opponent_pool, '*')))
    opponent_policy = CNNPolicy(checkpoint_dir=opponent_policy_logdir)
    opponent = ProbabilisticPolicyPlayer(opponent_policy,
                                         tenperature=FLAGS.policy_temperature,
                                         move_limit=500)

    if FLAGS.verbose:
        print("[Batch {:d}] opponent is loaded from {}".format(
            step, opponent_policy_logdir))

    win_ratio, states, actions, rewards = playout_n(learner, opponent, FLAGS.num_games)

    if FLAGS.verbose:
        print("{:d} games. {:d} states. {:d} moves. {:d} rewards. {:.2f}% win."
              .format(FLAGS.num_games, states.shape[0], actions.shape[0], rewards.shape[0], 100 * win_ratio))

    return (win_ratio, states, actions, rewards)


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

    # dim ordering is different to keras input
    state_batch = np.concatenate(state_batch, axis=0).transpose((0, 2, 3, 1))
    action_batch = np.concatenate(action_batch, axis=0)
    reward_batch = np.concatenate(reward_batch)

    return float(wins) / num_games, state_batch, action_batch, reward_batch


def load_learner_policy():
    if not FLAGS.resume:
        if not os.path.exists(FLAGS.rl_logdir):
            if FLAGS.verbose:
                print("creating RL train log directory {}".format(FLAGS.rl_logdir))
            os.makedirs(FLAGS.rl_logdir)

        SL_POLICY_CHECKPOINT_FILES = ['model.ckpt.data-00000-of-00001',
                                      'model.ckpt.index',
                                      'model.ckpt.mdata',
                                      'checkpoint']

        for each_file in SL_POLICY_CHECKPOINT_FILES:
            sl_ckpt = os.path.join(FLAGS.sl_logdir, each_file)
            rl_ckpt = os.path.join(FLAGS.rl_logdir, each_file)
            copyfile(sl_ckpt, rl_ckpt)
            if FLAGS.verbose:
                print("copied {} to {}".format(sl_ckpt, rl_ckpt))

    policy = CNNPolicy(checkpoint_dir=FLAGS.checkpoint_dir)
    policy.load_model()


def run_training(policy, cluster, server):
    # Between-graph replication
    with tf.device(tf.train.replica_device_setter(
                   worker_device="/job:worker/task:%d" % FLAGS.task_index,
                   cluster=cluster)):

        # count the number of updates
        global_step = tf.get_variable('global_step', [], tf.int32,
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)

        probs_op = policy.inference()
        loss_op = policy.loss(probs_op)
        acc_op = policy.accuracy(loss_op)

        # specify replicas optimizer
        with tf.name_scope('dist_train'):
            grad_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
            rep_op = tf.train.SyncReplicasOptimizer(grad_op,
                                                    replicas_to_aggregate=len(workers),
                                                    replica_id=FLAGS.task_index,
                                                    total_num_replicas=len(workers),
                                                    use_locking=True)

            train_op = rep_op.minimize(loss_op, global_step=global_step)

        init_token_op = rep_op.get_init_tokens_op()
        chief_queue_runner = rep_op.get_chief_queue_runner()

        # create a summary for our cost and accuracy
        tf.summary.scalar("loss", loss_op)
        tf.summary.scalar("accuracy", acc_op)

        # merge all summaries into a single "operation" which we can execute in a session
        summary_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()
        print("Variables initialized ...")

    is_chief = FLAGS.task_index == 0
    sv = tf.train.Supervisor(is_chief=is_chief,
                             logdir=FLAGS.rl_logdir,
                             global_step=global_step,
                             summary_op=None,
                             saver=tf.train.Saver(),
                             init_op=init_op)

    config = tf.ConfigProto(allow_soft_placement=True)
    with sv.prepare_or_wait_for_session(server.target, config=config) as sess:
        if is_chief:
            sv.start_queue_runners(sess, [chief_queue_runner])
            sess.run(init_token_op)

        # perform training cycles
        reports = 0
        start_time = time.time()
        for local_step in xrange(FLAGS.max_steps):
            start_time = time.time()

            (win_ratio, states, actions, rewards) = get_game_batch(local_step, policy)

            # perform the operations we defined earlier on batch
            _, loss, acc, summary, step = sess.run(
                [train_op, loss_op, acc_op, summary_op, global_step],
                feed_dict={
                    policy.statesholder: states,
                    policy.actionsholder: actions,
                    policy.rewardsholder: rewards
                })

            elapsed_time = time.time() - start_time
            print("Step: :d,".format(step),
                  " Loss: :.4f,".format(loss),
                  " Accuracy: :.4f,".format(acc),
                  " Elapsed: :3.2fms".format(float(elapsed_time*1000)))

            if is_chief:
                if step > FLAGS.checkpoint * reports:
                    reports += 1
                    # save summary
                    sv.summary_computed(sess, summary, global_step=step)
                    sv.summary_writer.flush()
                    # save checkpoint
                    sv.saver.save(sess, sv.save_path, global_step=step)

    if is_chief:
        sv.request_stop()
    else:
        sv.stop()


def main(argv=None):
    cluster = tf.train.ClusterSpec({"ps": parameter_servers, "worker": workers})

    # start a server for a specific task
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction
    server = tf.train.Server(cluster,
                             config=config,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    if FLAGS.job_name == 'job_name':
        server.join()
    elif FLAGS.jon_name == 'task_index':
        learner_policy = load_learner_policy()

        run_training(learner_policy, cluster, server)


if __name__ == '__main__':
    tf.app.run(main=main)
