ARG PYTHON_IMAGE=python:3.12-slim-bookworm@sha256:935c3c5b12357a7e10caa9c1e6ca5692344d97cf229832ef584be2cec9fc03df
ARG UV_IMAGE=ghcr.io/astral-sh/uv:0.13.7@sha256:96f40d3904aad8d0bd010539001d36c8e2188b981e1334471a90c4a1d93bb490

FROM ${UV_IMAGE} AS uv

# Build the locked environment and compile the C++ extension inside the image.
FROM ${PYTHON_IMAGE} AS core-build
ENV UV_PROJECT_ENVIRONMENT=/opt/venv UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=copy \
    NUMBA_CACHE_DIR=/tmp/numba MPLCONFIGDIR=/tmp/matplotlib
COPY --from=uv /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends g++ make \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml setup.py README.md uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src/ src/
COPY cpp/ cpp/
RUN uv sync --frozen --no-dev --no-editable

# Runtime: only the installed environment, models and experiment recipes.
FROM ${PYTHON_IMAGE} AS core
ARG SOURCE_REVISION=unknown
ENV PATH=/opt/venv/bin:$PATH PREDICTIVE_FSC_REVISION=${SOURCE_REVISION} \
    NUMBA_CACHE_DIR=/tmp/numba MPLCONFIGDIR=/tmp/matplotlib
COPY --from=core-build /opt/venv /opt/venv
WORKDIR /app
COPY models/ models/
COPY experiments/ experiments/
RUN useradd --uid 1000 --create-home runner
USER runner
ENTRYPOINT ["predictive-fsc"]
CMD ["--help"]
