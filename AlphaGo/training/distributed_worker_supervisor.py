#!/usr/bin/env python

import os
import subprocess
import time
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('sharedir', '/s3',
                    'Shared directory for distributed training.')
flags.DEFINE_integer('checkin_timeout', 60*3,
                     'Restart container if given timeout exceeded.')
flags.DEFINE_integer('check_interval', 10,
                     'Worker state check interval in sec.')

FLAGS = flags.FLAGS

checkin_dir = os.path.join(FLAGS.sharedir, 'checkin')


if __name__ == '__main__':
    """ Workaround. some worker's session is not initialized.
        reboot worker container which is not initialized.
    """
    docker_ps = subprocess.Popen("docker ps | grep 'worker'", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    worker_lines = docker_ps.stdout.read().split('\n')[:-1]
    workers = [worker.split()[-1] for worker in worker_lines]

    start_time = time.time()
    while True:
        worker_checkin_dir = os.path.join(checkin_dir, 'worker')
        exists_all = all(os.path.exists(os.path.join(worker_checkin_dir, worker)) for worker in workers)
        if exists_all:
            ps_checkin_file = os.path.join(checkin_dir, 'ps', 'ps0')
            ps_checkin_mtime = os.stat(ps_checkin_file).st_mtime
            running_all = all(ps_checkin_mtime <= os.stat(os.path.join(worker_checkin_dir, worker)).st_mtime
                              for worker in workers)
            if running_all:
                print("All worker checked in.")
                break
            else:
                print("Some worker timestamp is not updated.")
        else:
            print("Missing some worker")

        if time.time() - start_time > FLAGS.checkin_timeout:
            print("Check in timed out. Restarting lazy worker containers.")
            lazy_workers = [worker for worker in workers
                            if not os.path.exists(os.path.join(worker_checkin_dir, worker))
                            or ps_checkin_mtime > os.stat(os.path.join(worker_checkin_dir, worker)).st_mtime]
            for lazy_worker in lazy_workers:
                stop = subprocess.Popen("docker stop " + lazy_worker, shell=True)
                print("Stoping " + lazy_worker + " ...")
                stop.wait()
                print("Stopped.")
                stop = subprocess.Popen("docker start " + lazy_worker, shell=True)
                print("Starting " + lazy_worker + " ...")
                stop.wait()
                print("Stopped.")

            start_time = time.time()

        print("Waiting for all worker checking in.")
        time.sleep(FLAGS.check_interval)
