# set base image (host OS)
FROM ghcr.io/scc-digitalhub/digitalhub-serverless/python-runtime:3.10-latest

LABEL org.opencontainers.image.source = "https://github.com/tn-aixpa/audio-early-exit-transformer"

RUN git clone https://github.com/tn-aixpa/audio-early-exit-transformer.git

# set the working directory in the container
WORKDIR /home/jovyan/audio-early-exit-transformer

# install dependencies
RUN pip install -r requirements.txt

EXPOSE 8051

ENTRYPOINT ["python", "serve_eng_model.py"]



