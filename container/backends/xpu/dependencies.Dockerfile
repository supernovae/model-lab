# Immutable XPU userspace; no host drivers or configuration are modified.
FROM ghcr.io/astral-sh/uv:0.12.17@sha256:10787c682e4184e4f290de1171fd4703dc63de99221f10fe1c99002ce7fa9acc AS uv
FROM python:3.14.4-bookworm@sha256:786d12398a4b61918e68bc30d93d0bd679a95136bbbd1dbe4d96f116636516bc AS certificates
FROM ubuntu:26.04@sha256:88a381d5b5eeb2b35d3ad70925a362c37ce569daf43ede89ff818ec20e4d3794
LABEL org.opencontainers.image.source="https://github.com/supernovae/tiny-sparse-lab" org.opencontainers.image.licenses="MIT"
COPY --from=uv /uv /usr/local/bin/uv
COPY --from=certificates /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
ADD --checksum=sha256:d8faa1b588506b9e3e5a3f57807bb056d534a1641a6b10d15b6ea3f0b413b960 https://github.com/astral-sh/python-build-standalone/releases/download/20261013/cpython-3.14.8%2B20261013-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz /tmp/python.tar.gz
ADD --checksum=sha256:6031a63d6e8a12ce61c14efc15f2c8e727061286e3820b8594e6d00615e04d54 https://github.com/intel/compute-runtime/releases/download/26.35.39758.10/libigdgmm12_22.10.0_amd64.deb /tmp/debs/libigdgmm12_22.10.0_amd64.deb
ADD --checksum=sha256:c19a641b953d55aebbf1d51bec364a84bf629f985e02fbbe6dc70224c0e88470 https://github.com/intel/compute-runtime/releases/download/26.35.39758.10/libze-intel-gpu1_26.35.39758.10-0_amd64.deb /tmp/debs/libze-intel-gpu1_26.35.39758.10-0_amd64.deb
ADD --checksum=sha256:0a6e64a663ae65a0fa02d6912ae3b6b37cf85b90c21cc423fd9fef70aaf4f628 https://github.com/intel/intel-graphics-compiler/releases/download/v2.41.5/intel-igc-core-2_2.41.5%2B22716_amd64.deb /tmp/debs/intel-igc-core-2_2.41.5+22716_amd64.deb
ADD --checksum=sha256:45210e4549cd965ad7b9f160eefe53cbcdba01af341bea7ac7ba0aedeeb613ba https://github.com/oneapi-src/level-zero/releases/download/v1.34.0/libze1_1.34.0%2Bu24.04_amd64.deb /tmp/debs/libze1_1.34.0+u24.04_amd64.deb
ADD --checksum=sha256:281b3188840a7abdca85a02eb2609b0efad852e8994b3804a0e4227ec4ec3f90 https://snapshot.ubuntu.com/ubuntu/20261015T160000Z/pool/main/g/gcc-16/gcc-16-base_16-20260322-1ubuntu1_amd64.deb /tmp/debs/gcc-16-base_16-20260322-1ubuntu1_amd64.deb
ADD --checksum=sha256:2fb4d81c14fdf34251639ae82f5181f9f98480ea16125d535571ac1be9db3065 https://snapshot.ubuntu.com/ubuntu/20261015T160000Z/pool/main/g/gcc-16/libgcc-s1_16-20260322-1ubuntu1_amd64.deb /tmp/debs/libgcc-s1_16-20260322-1ubuntu1_amd64.deb
ADD --checksum=sha256:a32b9ad585e39bdc7bd15d1eae6293461e6d466859a1dd2f7862d1e40f89b9d8 https://snapshot.ubuntu.com/ubuntu/20261015T160000Z/pool/main/g/gcc-16/libstdc++6_16-20260322-1ubuntu1_amd64.deb /tmp/debs/libstdc++6_16-20260322-1ubuntu1_amd64.deb
ADD --checksum=sha256:34365d611ccf1f717f90a92fbe52ed96ae1829ab9be363c68a83563322797f8f https://snapshot.ubuntu.com/ubuntu/20261015T160000Z/pool/main/libz/libzstd/libzstd1_1.5.7+dfsg-3_amd64.deb /tmp/debs/libzstd1_1.5.7+dfsg-3_amd64.deb
ADD --checksum=sha256:450d9322ac4e9baea0f9a74548a20fde73b078235c1f5d1fb7084c2638e33cdd https://snapshot.ubuntu.com/ubuntu/20261015T160000Z/pool/main/z/zlib/zlib1g_1.3.dfsg+really1.3.1-1ubuntu3.1_amd64.deb /tmp/debs/zlib1g_1.3.dfsg+really1.3.1-1ubuntu3.1_amd64.deb
RUN test "$(uname -m)" = x86_64 \
 && dpkg -i /tmp/debs/*.deb \
 && mkdir /opt/python \
 && tar -xzf /tmp/python.tar.gz -C /opt/python --strip-components=1 \
 && rm -rf /tmp/debs /tmp/python.tar.gz \
 && ldconfig
ENV UV_PYTHON_DOWNLOADS=never UV_LINK_MODE=hardlink PATH=/opt/venv/bin:/opt/python/bin:/usr/local/bin:/usr/bin:/bin SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
COPY container/backends/xpu/requirements.lock /opt/requirements.lock
COPY container/backends/xpu/config.json /opt/backend.json
COPY container/image_plan.py /opt/image_plan.py
RUN /opt/python/bin/python3.14 -c "import sys; assert sys.version_info[:3] == (3, 14, 8)" \
 && uv venv --python /opt/python/bin/python3.14 /opt/venv \
 && uv pip install --python /opt/venv/bin/python --no-deps --only-binary=:all: --require-hashes -r /opt/requirements.lock \
 && python -c "import ctypes; [ctypes.CDLL(name) for name in ('libze_loader.so.1', 'libze_intel_gpu.so.1', 'libigc.so.2')]" \
 && python /opt/image_plan.py verify /opt/backend.json \
 && uv cache clean
