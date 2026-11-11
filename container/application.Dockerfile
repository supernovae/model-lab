ARG DEPENDENCY_IMAGE
FROM ${DEPENDENCY_IMAGE}
WORKDIR /build
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/
RUN uv pip install --python /opt/venv/bin/python --no-deps --no-build-isolation --offline . \
 && uv pip check --python /opt/venv/bin/python \
 && uv cache clean \
 && rm -rf /build
ARG SOURCE_REVISION=local
LABEL org.opencontainers.image.source="https://github.com/supernovae/model-lab" \
      org.opencontainers.image.revision=$SOURCE_REVISION \
      org.opencontainers.image.licenses="MIT"
COPY container/start.sh /usr/local/bin/lab-start
RUN /usr/sbin/groupadd -g 10001 lab && /usr/sbin/useradd -u 10001 -g 10001 -M lab \
 && mkdir /state /observe && chown 10001:10001 /state
ENV PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HOME=/state/home TMPDIR=/state/tmp \
    KLAB_WORK_DIR=/state/work KLAB_WORKER_LEASE_DIR=/state/work/worker-leases \
    HF_HOME=/state/cache/hf HF_HUB_CACHE=/state/cache/hf/hub HF_DATASETS_CACHE=/state/cache/hf/datasets \
    XDG_CACHE_HOME=/state/cache/xdg HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
USER 10001:10001
WORKDIR /state
EXPOSE 8501
# CLI jobs have no HTTP server. Viewer invocations opt into a web healthcheck.
HEALTHCHECK NONE
ENTRYPOINT ["/bin/sh", "/usr/local/bin/lab-start"]
CMD ["klab", "--help"]
