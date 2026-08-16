# MAMA-SYNTH submission container (build context = the mama_synth/ package dir).
FROM --platform=linux/amd64 docker.io/library/python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONPATH=/opt/app

# non-root user (Grand Challenge requirement)
RUN groupadd -r user && useradd -m --no-log-init -r -g user user
WORKDIR /opt/app

# SimpleITK needs these on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

COPY --chown=user:user submission/requirements.txt /opt/app/requirements.txt
RUN python -m pip install --no-cache-dir -r /opt/app/requirements.txt

# the mama_synth package (incl. submission/inference.py + submission/model/{model.pt,config.json})
COPY --chown=user:user . /opt/app/mama_synth

RUN mkdir -p /output && chown user:user /output
USER user

ENTRYPOINT ["python", "/opt/app/mama_synth/submission/inference.py"]
