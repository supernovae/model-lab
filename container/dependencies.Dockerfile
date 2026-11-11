FROM ghcr.io/astral-sh/uv:0.12.17@sha256:10787c682e4184e4f290de1171fd4703dc63de99221f10fe1c99002ce7fa9acc AS uv
FROM python:3.14.4-bookworm@sha256:786d12398a4b61918e68bc30d93d0bd679a95136bbbd1dbe4d96f116636516bc
COPY --from=uv /uv /usr/local/bin/uv
ARG BACKEND
LABEL org.opencontainers.image.source="https://github.com/supernovae/model-lab" org.opencontainers.image.licenses="MIT"
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=hardlink PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin
COPY container/backends/${BACKEND}/requirements.lock /opt/requirements.lock
COPY container/backends/${BACKEND}/config.json /opt/backend.json
COPY container/build-requirements.txt /opt/build-requirements.txt
COPY container/image_plan.py /opt/image_plan.py
RUN test "$(uname -m)" = x86_64 \
 && uv venv --python /usr/local/bin/python3.14 /opt/venv \
 && uv pip install --python /opt/venv/bin/python --no-deps --only-binary=:all: --require-hashes -r /opt/requirements.lock \
 && uv pip install --python /opt/venv/bin/python --no-deps --only-binary=:all: --require-hashes -r /opt/build-requirements.txt \
 && python /opt/image_plan.py verify /opt/backend.json \
 && uv cache clean
