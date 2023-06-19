FROM ghcr.io/allenai/cuda:11.3-cudnn8-dev-ubuntu20.04-v0.0.15

# Set up the main python environment
SHELL ["/bin/sh", "-c"]

COPY requirements.txt /aries/requirements.txt
WORKDIR /aries
RUN pip install -r requirements.txt

RUN python -m nltk.downloader -d /opt/miniconda3/share/nltk_data stopwords punkt book popular

RUN curl https://sh.rustup.rs -sSf | bash -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN bash -c "cd /tmp/; git clone https://github.com/openai/tiktoken tiktoken; cd tiktoken; git checkout 0.3.3; pip install ."

RUN aws s3 sync --no-sign-request s3://ai2-s2-research-public/aries/ data/aries/
RUN tar -C data/aries -xf data/aries/s2orc.tar.gz

COPY . /aries

RUN pip install -e .
