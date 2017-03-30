#!/bin/bash
/etc/init.d/rsyslog start

mkdir /s3
/root/go/bin/goofys s3_bucket_name /s3
# /root/go/bin/goofys --stat-cache-ttl 0 --type-cache-ttl 0 s3_bucket_name /s3

cd /root/RocAlphaGo
nohup python -u -m AlphaGo.training.reinforcement_policy_trainer_tensorflow --cluster_spec /cluster --job_name $(hostname | awk -F- '{print $1}') --task_index $(hostname | awk -F- '{print $2}') --sharedir /s3 --gpu_memory_fraction 0.1 > ps.log 2>&1 &

tail -f /dev/null
