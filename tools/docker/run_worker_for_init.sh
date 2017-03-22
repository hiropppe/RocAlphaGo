#!/bin/bash
/etc/init.d/rsyslog start

mkdir /mnt/s3
/root/go/bin/goofys s3_bucket_name /mnt/s3

cd /root/RocAlphaGo
nohup python -u -m AlphaGo.training.reinforcement_policy_trainer_tensorflow --cluster_spec /cluster --job_name $(hostname | awk -F- '{print $1}') --task_index $(hostname | awk -F- '{print $2}') --logdir /mnt/s3/logs --sync True --keras_weights /mnt/s3/weights/policy.hdf5 --gpu_memory_fraction 0.2 > /root/worker.log 2>&1 &

tail -f /dev/null
