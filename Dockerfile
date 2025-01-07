# set base image (host OS)
FROM python:3.10

LABEL org.opencontainers.image.source = "https://github.com/tn-aixpa/audio-early-exit-transformer"

RUN git clone https://github.com/tn-aixpa/audio-early-exit-transformer.git

# set the working directory in the container
WORKDIR /audio-early-exit-transformer

# install dependencies
RUN pip install -r requirements.txt

RUN chown -R 65535:65535 /audio-early-exit-transformer

USER 65535:65535

EXPOSE 8051

ENTRYPOINT ["python", "serve_eng_model.py"]



