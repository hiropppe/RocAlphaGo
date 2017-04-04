#!/usr/bin/env python

import datetime
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
import AlphaGo.util as util

from AlphaGo.ai import GreedyPolicyPlayer, ProbabilisticPolicyPlayer
from AlphaGo.util import flatten_idx
from AlphaGo.models import tf_policy

# input flags
flags = tf.app.flags
flags.DEFINE_string("cluster_spec", "", "Cluster specification")
flags.DEFINE_string("job_name", "", "Either 'ps' or 'worker'")
flags.DEFINE_integer("task_index", 0, "Index of task within the job")
flags.DEFINE_boolean('sync', False, 'Aggregate worker gradients synchronously.')

flags.DEFINE_integer('max_steps', 10000, 'Max number of steps to run.')
flags.DEFINE_integer('num_games', 1, 'Number of games in batch.')
flags.DEFINE_integer("move_limit", 500, "Maximum number of moves per game.")
flags.DEFINE_boolean('greedy_leaner', False, '')
flags.DEFINE_integer('greedy_start', 100, '')
flags.DEFINE_boolean('save_sgf', True,
                     'Save game state in sgf format.')
flags.DEFINE_integer("num_train_timestep", 100, "Maximum number of train time-step per game.")

flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate.')
flags.DEFINE_float('policy_temperature', 0.67, 'Policy temperature.')
flags.DEFINE_float('gpu_memory_fraction', 0.15,
                   'config.per_process_gpu_memory_fraction for training session')
flags.DEFINE_boolean('log_device_placement', False, '')

flags.DEFINE_integer('checkpoint', 5, 'Interval steps to execute checkpoint.')
flags.DEFINE_integer('opponent_checkpoint', 500, 'Interval steps to save polic in.')
flags.DEFINE_integer('worker_checkin_timeout', 60*3, 'Timeout for worker checkin.')

flags.DEFINE_string('sharedir', '/s3',
                    'Shared directory where to write train logs.')
flags.DEFINE_string('local_logdir', '/tmp/logs',
                    'Directory where to save latest parameters for playout.')
flags.DEFINE_string('summary_logdir', '/tmp/logs',
                    'Directory where to write summary logs.')
flags.DEFINE_string('sgf_logdir', '/tmp/sgf',
                    'Directory where to save playout sgf files.')
flags.DEFINE_string('opponent_pool', '/tmp/opponents',
                    'Opponent weights pool for playout.')

flags.DEFINE_string('keras_weights', None, 'Keras policy model file to migrate')

flags.DEFINE_boolean('verbose', True, '')

FLAGS = flags.FLAGS

# Define shared working sub directory
logdir = os.path.join(FLAGS.sharedir, 'logs')  # Save checkpoint files.
checkin_dir = os.path.join(FLAGS.sharedir, 'checkin')


def zero_baseline(state):
    return 0


def init_player(logdir, restore_checkpoint=False, greedy=False):
    # See as migrated Keras weights if model.json exists.
    if os.path.exists(os.path.join(logdir, 'model.json')):
        # Get filter size from json
        data = json.load(open(os.path.join(logdir, 'model.json')))
        model = json.loads(data['keras_model'])
        filters = model['config'][0]['config']['nb_filter']        
        # We play games between the current policy network and
        # a randomly selected previous iteration of the policy network.
        checkpoint_dir = np.random.choice(glob.glob(os.path.join(logdir, '[0-9]*')))
        if FLAGS.verbose:
            print("Randomly selected previous iteration of the SL policy network. {}".format(checkpoint_dir))
        policy_net = tf_policy.CNNPolicy(checkpoint_dir=checkpoint_dir,
                                         filters=filters)
    else:
        policy_net = tf_policy.CNNPolicy(checkpoint_dir=logdir)

    policy_net.init_graph()
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement,
                            device_count={'GPU': 1})
    policy_net.start_session(config)

    if restore_checkpoint:
        policy_net.load_model()

    if greedy:
        player = GreedyPolicyPlayer(policy_net,
                                    move_limit=FLAGS.move_limit)
    else:
        player = ProbabilisticPolicyPlayer(policy_net,
                                           FLAGS.policy_temperature,
                                           move_limit=FLAGS.move_limit,
                                           greedy_start=FLAGS.greedy_start)
    return player


def playout(step, learner, num_games, value=zero_baseline):
    opponent_policy_logdir = np.random.choice(glob.glob(os.path.join(FLAGS.opponent_pool, '*')))
    if FLAGS.verbose:
        print("Randomly selected opponent from pool. {}".format(opponent_policy_logdir))

    opponent = init_player(opponent_policy_logdir, restore_checkpoint=True)

    board_size = learner.policy.bsize
    states = [go.GameState(size=board_size) for _ in range(num_games)]

    state_batch, action_batch, reward_batch = [], [], []

    game_states = [[] for _ in range(num_games)]
    game_actions = [[] for _ in range(num_games)]

    learner_won = [None] * num_games

    # Wanna change color by step for mini-batch of 1 game (no batch)
    base = step % 2
    learner_color = [go.BLACK if (base+i) % 2 == 0 else go.WHITE for i in range(num_games)]
    white_states = states[0 if base else 1::2]
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
            is_learnable = current is learner and mv is not go.PASS_MOVE
            if is_learnable:
                (st_tensor, mv_tensor) = _make_training_pair(state, mv, learner.policy.preprocessor)
                game_states[idx].append(st_tensor)
                game_actions[idx].append(mv_tensor)

            state.do_move(mv)

            if state.is_end_of_game:
                learner_won[idx] = state.get_winner() == learner_color[idx]
                just_finished.append(idx)

                state_batch.append(np.concatenate(game_states[idx][:FLAGS.num_train_timestep], axis=0))
                action_batch.append(np.concatenate(game_actions[idx][:FLAGS.num_train_timestep], axis=0))
                z = 1 if learner_won[idx] else -1
                reward_batch.append(np.array([z - value(state) for state in game_states[idx][:FLAGS.num_train_timestep]]))

        for idx in just_finished:
            del idxs_to_unfinished_states[idx]

        current, other = other, current

    wins = sum(state.get_winner() == pc for (state, pc) in zip(states, learner_color))

    state_batch = np.concatenate(state_batch, axis=0)
    action_batch = np.concatenate(action_batch, axis=0)
    reward_batch = np.concatenate(reward_batch)

    opponent.policy.close_session()

    if FLAGS.save_sgf:
        sgf_path = os.path.join(FLAGS.sgf_logdir, str(FLAGS.task_index))
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
                                 os.path.basename(opponent_policy_logdir).rjust(5, '0'),
                                 str(game_idx), color_name, result]) + '.sgf'
            util.save_gamestate_to_sgf(state, sgf_path, sgf_name)

    return float(wins) / num_games, state_batch, action_batch, reward_batch


def get_game_batch(step, learner):
    if FLAGS.verbose:
        print("Playing self match at step {:d}".format(step))

    start_time = time.time()
    try:
        (win_ratio, states, actions, rewards) = playout(step, learner, FLAGS.num_games)
        elapsed_sec = time.time() - start_time

        if FLAGS.verbose:
            print("[Batch {:d}] {:d} games.".format(step, FLAGS.num_games) +
                  " {:d} states.".format(states.shape[0]) +
                  " {:d} moves.".format(actions.shape[0]) +
                  " {:d} rewards.".format(rewards.shape[0]) +
                  " {:.2f}% win.".format(100*win_ratio) +
                  " Elapsed: {:3.2f}s".format(float(elapsed_sec)))
    except:
        err, msg, _ = sys.exc_info()
        sys.stderr.write("{} {}\n".format(err, msg))
        sys.stderr.write(traceback.format_exc())
        return False, None, None, None, None

    return True, win_ratio, states, actions, rewards


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
            if scope_name + '_W:0' in layer:
                value = layer[scope_name + '_W:0'].value
            else:
                value = layer[scope_name + '_W'].value

            value = value.transpose((2, 3, 1, 0))
            # Transpose kernel dimention ordering.
            # TF uses the last dimension as channel dimension,
            # K kernel shape: (output_depth, input_depth, rows, cols)
            # TF kernel shape: (rows, cols, input_depth, output_depth)
            return tf.Variable(value, name=scope_name + '_W')
        elif wb.lower() == 'b':
            if layer_i == 14:
                if 'Variable:0' in layer:
                    return tf.Variable(layer['Variable:0'].value, name='Variable')
                else:
                    return tf.Variable(layer['param_0'].value, name='Variable')
            else:
                if scope_name + '_b:0' in layer:
                    return tf.Variable(layer[scope_name + '_b:0'].value, name=scope_name + '_b')
                else:
                    return tf.Variable(layer[scope_name + '_b'].value, name=scope_name + '_b')
    return keras_weight


def wait_for_all_worker_checking_in(num_workers):
    # Workaround. some worker's session is not initialized.
    # write self check in file then wait for checking in all workers
    # worker that is not checked in requires reboot or else ... 
    worker_checkin_dir = os.path.join(checkin_dir, FLAGS.job_name)
    if not os.path.exists(worker_checkin_dir):
        os.mkdir(worker_checkin_dir)
    worker_checkin_file = os.path.join(worker_checkin_dir, 'worker' + str(FLAGS.task_index))
    if os.path.exists(worker_checkin_file):
        os.remove(worker_checkin_file)
    with open(worker_checkin_file, 'w') as checkin:
        checkin.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%s'))

    while True:
        exists_all = all(os.path.exists(os.path.join(worker_checkin_dir, 'worker' + str(task_index)))
                         for task_index in range(num_workers))
        if exists_all:
            ps_checkin_file = os.path.join(checkin_dir, 'ps', 'ps0')
            ps_checkin_mtime = os.stat(ps_checkin_file).st_mtime
            running_all = all(ps_checkin_mtime <= os.stat(os.path.join(worker_checkin_dir, 'worker' + str(task_index))).st_mtime
                              for task_index in range(num_workers))
            if running_all:
                print("All worker checked in.")
                break
            else:
                print("Some worker timestamp is not updated.")
        else:
            print("Missing some worker")
        print("Wait for all worker checking in.")
        time.sleep(10)


def init_checkpoint_with_keras_weights(cluster, server, num_workers):
    policy = tf_policy.CNNPolicy()

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

        logits_op = policy.inference(statesholder, weight_setter=init_keras_weight_setter())
        loss_op = policy.loss(logits_op, actionsholder, rewardsholder)

        # specify replicas optimizer
        with tf.name_scope('dist_train'):
            grad_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
            rep_op = tf.train.SyncReplicasOptimizer(grad_op,
                                                    replicas_to_aggregate=num_workers,
                                                    replica_id=FLAGS.task_index,
                                                    total_num_replicas=num_workers,
                                                    use_locking=True)

            rep_op.minimize(loss_op, global_step=global_step)

        init_op = tf.global_variables_initializer()
        print("Variables initialized ...")

    is_chief = FLAGS.task_index == 0
    sv = tf.train.Supervisor(is_chief=is_chief,
                             logdir=logdir,
                             global_step=global_step,
                             summary_op=None,
                             saver=tf.train.Saver(),
                             init_op=init_op)

    config = tf.ConfigProto(allow_soft_placement=True)
    print("Wait for session ...")
    with sv.managed_session(server.target, config=config):
        print("Session initialized.")

        wait_for_all_worker_checking_in(num_workers)

    sv.stop()


def run_training(cluster, server, num_workers):
    learner = init_player(FLAGS.local_logdir, greedy=True)

    # Between-graph replication
    with tf.device(tf.train.replica_device_setter(
                   worker_device="/job:worker/task:{:d}".format(FLAGS.task_index),
                   cluster=cluster)):

        # count the number of updates
        global_step = tf.get_variable('global_step', [], tf.int32,
                                      initializer=tf.constant_initializer(1),
                                      trainable=False)

        statesholder = learner.policy._statesholder()
        actionsholder = learner.policy._actionsholder()
        rewardsholder = learner.policy._rewardsholder()

        logits_op = learner.policy.inference(statesholder)
        probs_op = learner.policy.probs(logits_op)
        loss_op = learner.policy.loss(logits_op, actionsholder, rewardsholder)
        acc_op = learner.policy.accuracy(probs_op, actionsholder)

        mean_reward_op = tf.reduce_mean(rewardsholder)

        # specify replicas optimizer
        with tf.name_scope('dist_train'):
            grad_op = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
            if FLAGS.sync:
                rep_op = tf.train.SyncReplicasOptimizer(grad_op,
                                                        replicas_to_aggregate=num_workers,
                                                        replica_id=FLAGS.task_index,
                                                        total_num_replicas=num_workers,
                                                        use_locking=True)

                #grads_op = rep_op.compute_gradients(loss_op)
                #for i, (grad, var) in enumerate(grads_op):
                #    if grad is not None:
                #        grads_op[i] = (tf.mul(grad, mean_reward_op), var)
                #train_op = rep_op.apply_gradients(grads_op, global_step=global_step)
                train_op = rep_op.minimize(loss_op, global_step=global_step)
            else:
                #grads_op = grad_op.compute_gradients(loss_op)
                #for i, (grad, var) in enumerate(grads_op):
                #    if grad is not None:
                #        grads_op[i] = (tf.mul(grad, mean_reward_op), var)
                #train_op = grad_op.apply_gradients(grads_op, global_step=global_step)
                train_op = grad_op.minimize(loss_op, global_step=global_step)

        #for grad, var in grads_op:
        #    tf.summary.histogram(var.name, var)
        #    if grad is not None:
        #        tf.summary.histogram(var.name + '/gradients', grad)

        if FLAGS.sync:
            init_token_op = rep_op.get_init_tokens_op()
            chief_queue_runner = rep_op.get_chief_queue_runner()

        # workaround. remote session is too slow to playout.
        # save latest parameter to local disk then load parameter in local session.
        local_saver = tf.train.Saver()

        # create a summary for our cost and accuracy
        tf.summary.scalar("loss", loss_op)
        tf.summary.scalar("accuracy", acc_op)
        tf.summary.scalar("mean_reward_op", mean_reward_op)

        # merge all summaries into a single "operation" which we can execute in a session
        summary_op = tf.summary.merge_all()
        init_op = tf.global_variables_initializer()
        print("Variables initialized ...")

    is_chief = FLAGS.task_index == 0
    sv = tf.train.Supervisor(is_chief=is_chief,
                             logdir=logdir,
                             global_step=global_step,
                             summary_op=None,
                             saver=tf.train.Saver(max_to_keep=0),
                             init_op=init_op)

    config = tf.ConfigProto(allow_soft_placement=True)
    print("Wait for session ...")
    with sv.managed_session(server.target, config=config) as sess:
        print("Session initialized.")

        wait_for_all_worker_checking_in(num_workers)

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
            # workaround. remote session is too slow to playout.
            # save latest parameter to local disk then load parameter in local session.
            print("Save latest parameter" +
                  " to {:s} to execute playout.".format(FLAGS.local_logdir))
            local_saver.save(sess, os.path.join(FLAGS.local_logdir, 'model.ckpt'), global_step=0)
            # restore latest checkpoint
            learner.policy.load_model()

            (succeed, win_ratio, states, actions, rewards) = get_game_batch(step, learner)

            # Retry if playout failed unexpectedly.
            if not succeed:
                continue

            start_time = time.time()

            # perform the operations we defined earlier on batch
            _, loss, acc, mean_reward, summary, step = sess.run(
                [train_op, loss_op, acc_op, mean_reward_op, summary_op, global_step],
                feed_dict={
                    statesholder: states,
                    actionsholder: actions,
                    rewardsholder: rewards
                })

            elapsed_time = time.time() - start_time
            print("[Step {:d}]".format(step) +
                  " Loss: {:.4f},".format(loss) +
                  " Mean reward: {:.4f},".format(mean_reward) +
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
                        meta_files = [os.path.basename(f) for f in glob.glob(os.path.join(logdir, 'model.ckpt-*.meta'))]
                        opponent_step = max(int(re.findall(r'model\.ckpt\-(\d+)\.meta', f)[0]) for f in meta_files)
                        new_opponent_path = os.path.join(FLAGS.opponent_pool, str(opponent_step))
                        os.mkdir(new_opponent_path)
                        for src in glob.glob(os.path.join(logdir, 'model.ckpt-{:d}.*'.format(opponent_step))):
                            shutil.copy2(src, new_opponent_path)
                        shutil.copy2(os.path.join(logdir, 'checkpoint'), new_opponent_path)
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

    if FLAGS.job_name == 'ps':
        # Workaround. check in for worker management
        ps_checkin_dir = os.path.join(checkin_dir, FLAGS.job_name)
        if not os.path.exists(ps_checkin_dir):
            os.mkdir(ps_checkin_dir)
        ps_checkin_file = os.path.join(ps_checkin_dir, 'ps' + str(FLAGS.task_index))
        if os.path.exists(ps_checkin_file):
            os.remove(ps_checkin_file)
        with open(ps_checkin_file, 'w') as checkin:
            checkin.write(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%s'))
        print('Checked in.')

        server.join()
    elif FLAGS.job_name == 'worker':
        if FLAGS.keras_weights:
            init_checkpoint_with_keras_weights(cluster, server, num_workers)
        else:
            run_training(cluster, server, num_workers)


if __name__ == '__main__':
    tf.app.run(main=main)
