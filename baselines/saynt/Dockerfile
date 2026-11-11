# Standalone; build from the repository root:
#   docker build -t predictive-fsc-saynt baselines/saynt
# Native compilation uses one job by default; add --build-arg SAYNT_BUILD_JOBS=N.
ARG PYTHON_IMAGE=python:3.10-slim-bookworm@sha256:5be5aaecb962a41567bc8041f1e9e3432cb085b037fe057f9a562630b2e4b370
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.13.7@sha256:96f40d3904aad8d0bd010539001d36c8e2188b981e1334471a90c4a1d93bb490

FROM ${UV_IMAGE} AS uv

FROM ${PYTHON_IMAGE}
# SAYNT does not use the predictive-fsc CLI's thread limits, so set them here.
ENV OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git automake autoconf libboost-all-dev libgmp-dev libglpk-dev \
    libhwloc-dev libz3-dev libxerces-c-dev libeigen3-dev uuid-dev \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /opt/saynt/
RUN uv pip install --system --require-hashes -r /opt/saynt/requirements.txt
COPY install.sh /opt/saynt/
ARG SAYNT_BUILD_JOBS=1
RUN sh /opt/saynt/install.sh
RUN useradd --uid 1000 --create-home runner && mkdir /work && chown runner:runner /work
WORKDIR /work
USER runner
ENTRYPOINT ["python", "/opt/saynt/source/paynt/paynt.py"]
CMD ["--help"]
