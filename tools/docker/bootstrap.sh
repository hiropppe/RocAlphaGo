#!/bin/bash
mkdir /mnt/s3_mount_name
/root/go/bin/goofys -f s3_bucket_name /mnt/s3_mount_name
