FROM ubuntu:18.04
USER root
WORKDIR /

ENV http_proxy $HTTP_PROXY
ENV https_proxy $HTTP_PROXY

ARG DEPENDENCIES="autoconf \
                  automake \
                  build-essential \
                  cmake \
                  cpio \
                  curl \
                  gnupg2 \
                  libdrm2 \
                  libglib2.0-0 \
                  lsb-release \
                  libgtk-3-0 \
                  libtool \
                  python3-pip \
                  python3-dev \
                  python3-setuptools \
                  udev \
                  git \
                  unzip"

ENV TZ 'Europe/Madrid'
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends ${DEPENDENCIES} && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /tmp
COPY docker/l_openvino_toolkit_p_2020.3.341.tgz /tmp/
RUN tar -xzf ./*.tgz \
    && cd l_openvino_toolkit_p_2020.3.341 \
    && sed -i 's/decline/accept/g' silent.cfg \
    && ./install.sh -s silent.cfg \
    && ./install_openvino_dependencies.sh \
    && rm -rf /tmp/l_openvino_toolkit_* \
    && cd /opt/intel/openvino/inference_engine/samples/cpp/ \
    && ./build_samples.sh \
    && cd /root/inference_engine_cpp_samples_build/intel64/Release \
    && mv benchmark_app /root/ \
    && mv lib/ /root/ \
    && cd /root \
    && rm -rf inference_engine_cpp_samples_build/

WORKDIR /root/

COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt
COPY inference.py /root/
COPY utils/ /root/

ENTRYPOINT ["python3", "/root/inference.py"]
