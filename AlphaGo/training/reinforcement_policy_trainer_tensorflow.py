#!/usr/bin/env python

import concurrent.futures
import glob
import json
import numpy as np
import os
import re
import sys
import shutil
import time
import traceback
import tensorflow as tf

import AlphaGo.go as go

from AlphaGo.ai import ProbabilisticPolicyPlayer
from AlphaGo.util import flatten_idx
from AlphaGo.models import policy, tf_policy

# input flags
flags = tf.app.flags
flags.DEFINE_string("cluster_spec", "", "Cluster specification")
flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
flags.DEFINE_integer("task_index", 0, "Index of task within the job")
flags.DEFINE_boolean('sync', False, 'Aggregate worker gradients synchronously.')

flags.DEFINE_integer('max_steps', 10000, 'Max number of steps to run.')
flags.DEFINE_integer('num_games', 1, 'Number of games in batch.')
flags.DEFINE_integer('num_playout_cpu', 1, 'Number of cpu for playout.')
flags.DEFINE_integer("move_limit", 500, "Maximum number of moves per game.")

flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate.')
flags.DEFINE_float('policy_temperature', 0.67, 'Policy temperature.')
flags.DEFINE_float('gpu_memory_fraction', 0.15,
                   'config.per_process_gpu_memory_fraction for training session')
flags.DEFINE_boolean('log_device_placement', False, '')

flags.DEFINE_integer('checkpoint', 5, 'Interval steps to execute checkpoint.')
flags.DEFINE_integer('opponent_checkpoint', 10, 'Interval steps to save policy as a new opponent.')

flags.DEFINE_string('logdir', '/mnt/s3/logs',
                    'Shared directory where to write train logs.')
flags.DEFINE_string('summary_logdir', '/tmp/logs',
                    'Directory where to write summary logs.')
flags.DEFINE_string('opponent_pool', '/mnt/opponents',
                    'Shared directory where to save trained policy weights for opponent')

flags.DEFINE_string('keras_weights', None, 'Keras policy model file to migrate')

flags.DEFINE_boolean('resume', False, '')
flags.DEFINE_boolean('verbose', True, '')

FLAGS = flags.FLAGS


def zero_baseline(state):
    return 0


def load_player(logdir):
    # See as migrated Keras weights if model.json exists.
    if os.path.exists(os.path.join(logdir, 'model.json')):
        # Get filter size from json
        data = json.load(open(os.path.join(logdir, 'model.json')))
        model = json.loads(data['keras_model'])
        filters = model['config'][0]['config']['nb_filter']        
        # We play games between the current policy network and
        # a randomly selected previous iteration of the policy network.
        checkpoint_dir = np.random.choice(glob.glob(os.path.join(logdir, '*')))
        if FLAGS.verbose:
            print("Randomly selected previous iteration of the SL policy network. {}".format(checkpoint_dir))
        policy_net = tf_policy.CNNPolicy(checkpoint_dir=checkpoint_dir,
                                         filters=filters)
    else:
        policy_net = tf_policy.CNNPolicy(checkpoint_dir=logdir)

    policy_net.init_graph(train=False)
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement,
                            device_count={'GPU': 1})
    policy_net.start_session(config)
    policy_net.load_model()

    player = ProbabilisticPolicyPlayer(policy_net,
                                       FLAGS.policy_temperature,
                                       move_limit=FLAGS.move_limit)
    return player


def playout(step, num_games, value=zero_baseline):
    learner = load_player(FLAGS.logdir)

    opponent_policy_logdir = np.random.choice(glob.glob(os.path.join(FLAGS.opponent_pool, '*')))
    if FLAGS.verbose:
        print("Randomly selected opponent from pool. {}".format(opponent_policy_logdir))
    opponent = load_player(opponent_policy_logdir)

    board_size = learner.policy.bsize
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
    state_batch = np.concatenate(state_batch, axis=0)
    action_batch = np.concatenate(action_batch, axis=0)
    reward_batch = np.concatenate(reward_batch)

    learner.policy.close_session()
    opponent.policy.close_session()

    return float(wins) / num_games, state_batch, action_batch, reward_batch


def get_game_batch(step):
    # executor = concurrent.futures.ThreadPoolExecutor(max_workers=FLAGS.num_playout_cpu)
    # num_game_per_cpu = FLAGS.num_games/FLAGS.num_playout_cpu

    # win_ratio_list, states_list, actions_list, rewards_list = [], [], [], []

    start_time = time.time()
    """
    futures = [executor.submit(playout, step, num_game_per_cpu) for i in range(FLAGS.num_playout_cpu)]

    try:
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            (win_ratio, states, actions, rewards) = future.result()
            if FLAGS.verbose:
                print("[Playout {:d}] {:d} games. {:d} states. {:d} moves. {:d} rewards. {:.2f}% win."
                      .format(i,
                              num_game_per_cpu,
                              states.shape[0],
                              actions.shape[0],
                              rewards.shape[0],
                              100 * win_ratio))
            win_ratio_list.append(win_ratio)
            states_list.append(states)
            actions_list.append(actions)
            rewards_list.append(rewards)
    except concurrent.futures.TimeoutError:
        sys.stderr.write('Playout timed out.\n')
    except:
        err, msg, _ = sys.exc_info()
        sys.stderr.write("{} {}\n".format(err, msg))
        sys.stderr.write(traceback.format_exc())
    """

    (win_ratio, states, actions, rewards) = playout(step, FLAGS.num_games)

    """
    win_ratio = np.mean(win_ratio_list)
    states = np.concatenate(states_list, axis=0)
    actions = np.concatenate(actions_list, axis=0)
    rewards = np.concatenate(rewards_list)
    """

    elapsed_sec = time.time() - start_time

    if FLAGS.verbose:
        print("[Worker Batch] {:d} games.".format(FLAGS.num_games) +
              " {:d} states.".format(states.shape[0]) +
              " {:d} moves.".format(actions.shape[0]) +
              " {:d} rewards.".format(rewards.shape[0]) +
              " {:.2f}% win.".format(100*win_ratio) +
              " Elapsed: {:3.2f}s".format(float(elapsed_sec)))

    """
    try:
        if FLAGS.verbose:
            print('Shutting down executor.')
        executor.shutdown()
    except:
        err, msg, _ = sys.exc_info()
        sys.stderr.write("{} {}\n".format(err, msg))
        sys.stderr.write(traceback.format_exc())
        if FLAGS.verbose:
            print('Shutting down executor with nowait.')
        executor.shutdown(wait=False)
    """
    return win_ratio, states, actions, rewards


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


def init_keras_weight_setter():
    import h5py as h5
    keras_model_weights = h5.File(FLAGS.keras_weights)['model_weights']

    def keras_weight(layer_i, wb, scope_name):
        layer = keras_model_weights[scope_name]
        if wb.upper() == 'W':
            value = layer[scope_name + '_W:0'].value
            value = value.transpose((2, 3, 1, 0))
            # Transpose kernel dimention ordering.
            # TF uses the last dimension as channel dimension,
            # K kernel shape: (output_depth, input_depth, rows, cols)
            # TF kernel shape: (rows, cols, input_depth, output_depth)
            return tf.Variable(value, name=scope_name + '_W')
        elif wb.lower() == 'b':
            if layer_i == 14:
                return tf.Variable(layer['Variable:0'].value, name='Variable')
            else:
                return tf.Variable(layer[scope_name + '_b:0'].value, name=scope_name + '_b')
    return keras_weight


def init_checkpoint_with_keras_weights(policy, cluster, server, num_workers):
    weight_setter = init_keras_weight_setter()
    # Between-graph replication
    with tf.device(tf.train.replica_device_setter(
                   worker_device="/job:worker/task:{:d}".format(FLAGS.task_index),
                   cluster=cluster)):

        # count the number of updates
        global_step = tf.get_variable('global_step', [], tf.int32,
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)

        statesholder = policy._statesholder()
        actionsholder = policy._actionsholder()
        rewardsholder = policy._rewardsholder()

        probs_op = policy.inference(statesholder, weight_setter=weight_setter)
        loss_op = policy.loss(probs_op, actionsholder, rewardsholder)
        policy.accuracy(probs_op, actionsholder)

        # specify replicas optimizer
        with tf.name_scope('dist_train'):
            grad_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
            rep_op = tf.train.SyncReplicasOptimizer(grad_op,
                                                    replicas_to_aggregate=num_workers,
                                                    replica_id=FLAGS.task_index,
                                                    total_num_replicas=num_workers,
                                                    use_locking=True)

            rep_op.minimize(loss_op, global_step=global_step)

        init_token_op = rep_op.get_init_tokens_op()
        chief_queue_runner = rep_op.get_chief_queue_runner()

        # create a summary for our cost and accuracy
        init_op = tf.global_variables_initializer()
        print("Variables initialized ...")

    is_chief = FLAGS.task_index == 0
    sv = tf.train.Supervisor(is_chief=is_chief,
                             logdir=FLAGS.logdir,
                             global_step=global_step,
                             summary_op=None,
                             saver=tf.train.Saver(),
                             init_op=init_op)

    config = tf.ConfigProto(allow_soft_placement=True)
    with sv.managed_session(server.target, config=config) as sess:
        if is_chief:
            sv.start_queue_runners(sess, [chief_queue_runner])
            sess.run(init_token_op)
            sv.saver.save(sess, sv.save_path, global_step=0)

    if is_chief:
        sv.request_stop()
    else:
        sv.stop()


def run_training(policy, cluster, server, num_workers):
    # Between-graph replication
    with tf.device(tf.train.replica_device_setter(
                   worker_device="/job:worker/task:{:d}".format(FLAGS.task_index),
                   cluster=cluster)):

        # count the number of updates
        global_step = tf.get_variable('global_step', [], tf.int32,
                                      initializer=tf.constant_initializer(0),
                                      trainable=False)

        statesholder = policy._statesholder()
        actionsholder = policy._actionsholder()
        rewardsholder = policy._rewardsholder()

        probs_op = policy.inference(statesholder)
        loss_op = policy.loss(probs_op, actionsholder, rewardsholder)
        acc_op = policy.accuracy(probs_op, actionsholder)

        # specify replicas optimizer
        with tf.name_scope('dist_train'):
            grad_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)

            if FLAGS.sync:
                rep_op = tf.train.SyncReplicasOptimizer(grad_op,
                                                        replicas_to_aggregate=num_workers,
                                                        replica_id=FLAGS.task_index,
                                                        total_num_replicas=num_workers,
                                                        use_locking=True)
                """
                grads = rep_op.compute_gradients(loss_op)
                mean_reward = tf.reduce_mean(rewardsholder)
                for i, (grad, var) in enumerate(grads):
                    if grad is not None:
                        grads[i] = (tf.mul(grad, mean_reward), var)
                train_op = rep_op.apply_gradients(grads, global_step=global_step)
                """
                train_op = rep_op.minimize(loss_op, global_step=global_step)
            else:
                """
                grads = grad_op.compute_gradients(loss_op)
                mean_reward = tf.reduce_mean(rewardsholder)
                for i, (grad, var) in enumerate(grads):
                    if grad is not None:
                        grads[i] = (tf.mul(grad, mean_reward), var)
                train_op = grad_op.apply_gradients(grads, global_step=global_step)
                """
                train_op = grad_op.minimize(loss_op, global_step=global_step)

        if FLAGS.sync:
            init_token_op = rep_op.get_init_tokens_op()
            chief_queue_runner = rep_op.get_chief_queue_runner()

        # create a summary for our cost and accuracy
        tf.summary.scalar("loss", loss_op)
        tf.summary.scalar("accuracy", acc_op)
        # tf.summary.scalar("mean_reward", mean_reward)

        # merge all summaries into a single "operation" which we can execute in a session
        summary_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()
        print("Variables initialized ...")

    is_chief = FLAGS.task_index == 0
    sv = tf.train.Supervisor(is_chief=is_chief,
                             logdir=FLAGS.logdir,
                             global_step=global_step,
                             summary_op=None,
                             saver=tf.train.Saver(max_to_keep=0),
                             init_op=init_op)

    config = tf.ConfigProto(allow_soft_placement=True)
    with sv.managed_session(server.target, config=config) as sess:
        policy.sess = sess
        policy.probs = probs_op
        policy.statesholder = statesholder

        if is_chief:
            summary_writer = tf.summary.FileWriter(FLAGS.summary_logdir, sess.graph)
            if FLAGS.sync:
                sv.start_queue_runners(sess, [chief_queue_runner])
                sess.run(init_token_op)

        # perform training cycles
        step = 0
        reports = 0
        opponents = len(glob.glob(os.path.join(FLAGS.opponent_pool, '*'))) - 1
        while not sv.should_stop() and step <= FLAGS.max_steps:
            start_time = time.time()

            (win_ratio, states, actions, rewards) = get_game_batch(step)

            # perform the operations we defined earlier on batch
            _, loss, acc, summary, step = sess.run(
                [train_op, loss_op, acc_op, summary_op, global_step],
                feed_dict={
                    statesholder: states,
                    actionsholder: actions,
                    rewardsholder: rewards
                })

            elapsed_time = time.time() - start_time
            print("[Step {:d}]".format(step) +
                  " Loss: {:.4f},".format(loss) +
                  " Accuracy: {:.4f},".format(acc) +
                  " Elapsed: {:3.2f}s".format(float(elapsed_time)))

            if is_chief:
                try:
                    # execute checkpoint manually
                    if step >= FLAGS.checkpoint * (reports+1):
                        print("Execute checkpoint at step {:d}.").format(step)
                        reports += 1
                        sv.saver.save(sess, sv.save_path, global_step=step)
                        summary_writer.add_summary(summary, global_step=step)
                        summary_writer.flush()

                    # update opponent pool
                    if step >= FLAGS.opponent_checkpoint * (opponents+1):
                        print("Update opponent pool at step {:d}.").format(step)
                        opponents += 1
                        meta_files = [os.path.basename(f) for f in glob.glob(os.path.join(FLAGS.logdir, 'model.ckpt-*.meta'))]
                        opponent_step = max(int(re.findall(r'model\.ckpt\-(\d+)\.meta', f)[0]) for f in meta_files)
                        new_opponent_path = os.path.join(FLAGS.opponent_pool, str(opponent_step))
                        os.mkdir(new_opponent_path)
                        for src in glob.glob(os.path.join(FLAGS.logdir, 'model.ckpt-{:d}.*'.format(opponent_step))):
                            shutil.copy2(src, new_opponent_path)
                        shutil.copy2(os.path.join(FLAGS.logdir, 'checkpoint'), new_opponent_path)
                except:
                    err, msg, _ = sys.exc_info()
                    sys.stderr.write("{} {}\n".format(err, msg))
                    sys.stderr.write(traceback.format_exc())

    if is_chief:
        sv.request_stop()
    else:
        sv.stop()


def main(argv=None):
    from ast import literal_eval
    cluster_spec = literal_eval(open(FLAGS.cluster_spec).read())
    num_workers = len(cluster_spec['worker'])
    cluster = tf.train.ClusterSpec(cluster_spec)

    # start a server for a specific task
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement,
                            device_count={'GPU': 1})
    config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_memory_fraction

    server = tf.train.Server(cluster,
                             config=config,
                             job_name=FLAGS.job_name,
                             task_index=FLAGS.task_index)

    learner_policy = tf_policy.CNNPolicy(checkpoint_dir=FLAGS.logdir)
    if FLAGS.job_name == 'ps':
        server.join()
    elif FLAGS.job_name == 'worker':
        if FLAGS.keras_weights:
            init_checkpoint_with_keras_weights(learner_policy, cluster, server, num_workers)
        else:
            run_training(learner_policy, cluster, server, num_workers)


if __name__ == '__main__':
    tf.app.run(main=main)
