#!/bin/bash
/etc/init.d/rsyslog start

mkdir /s3
/root/go/bin/goofys s3_mount_name /s3

cd /root/RocAlphaGo
nohup python -u -m AlphaGo.training.reinforcement_policy_trainer_tensorflow --cluster_spec /cluster --job_name $(hostname | awk -F- '{print $1}') --task_index $(hostname | awk -F- '{print $2}') --logdir /s3/logs --gpu_memory_fraction 0.1 > /root/ps.log 2>&1 &

tail -f /dev/null
