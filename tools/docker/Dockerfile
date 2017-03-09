FROM ubuntu:16.04

MAINTAINER sergei <sergei.0910@gmail.com>

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python \
        python-dev \
        rsync \
        software-properties-common \
        unzip \
        iputils-ping \
        net-tools \
        netcat \
        vim \
        git \
        golang \
        fuse \
        rsyslog \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install goofys
RUN export GOPATH=/root/go && \
    go get -v github.com/kahing/goofys && \
    go install -v github.com/kahing/goofys && \
    mkdir /root/.aws

# Copy AWS credentials
COPY credentials /root/.aws/credentials

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py

RUN pip --no-cache-dir install \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        scipy \
        sklearn \
        pandas \
        Pillow \
        awscli \
        && \
    python -m ipykernel.kernelspec

# Install TensorFlow CPU version from central repo
RUN pip --no-cache-dir install \
    http://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.0-cp27-none-linux_x86_64.whl

# We just add a little wrapper script.
COPY bootstrap.sh /

# TensorBoard
EXPOSE 6006

CMD ["/bootstrap.sh"]
