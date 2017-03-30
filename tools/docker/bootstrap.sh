#!/bin/bash
mkdir /s3
/root/go/bin/goofys -f s3_bucket_name /s3
# /root/go/bin/goofys -f --stat-cache-ttl 0 --type-cache-ttl 0 s3_bucket_name /s3
