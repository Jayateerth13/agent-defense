# Disposable sandbox for the oversight-closure experiments: one container per run, no network at run time.
FROM python:3.13-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends strace make git curl procps \
 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir --root-user-action=ignore pytest==9.0.2

# agent: runs the model's commands.  checker: runs the check.  root: setup and protection only.
RUN useradd -m -s /bin/bash agent && useradd -m -s /bin/bash checker && mkdir -p /workspace
# Protection makes root own parts of .git; without this, git would refuse to run for the agent at all.
RUN git config --system --add safe.directory /workspace

# Real packages for the closure check (copied into the workspace by their scenarios; no network at run time).
RUN pip install --no-cache-dir --root-user-action=ignore text-unidecode==1.3
RUN git -c advice.detachedHead=false clone -q --depth 1 --branch 1.17.0 https://github.com/benjaminp/six /opt/pkgs/six \
 && git -c advice.detachedHead=false clone -q --depth 1 --branch v9.1.3 https://github.com/un33k/python-slugify /opt/pkgs/python-slugify \
 && git -c advice.detachedHead=false clone -q --depth 1 --branch 2.5.0 https://github.com/hukkin/tomli /opt/pkgs/tomli

# Closure probe: imported by every Python process, inert unless the closure builder sets OVERSIGHT_PROBE_DIR.
COPY probe/oversight_probe.py /usr/local/lib/python3.13/site-packages/oversight_probe.py
RUN echo "import oversight_probe" > /usr/local/lib/python3.13/site-packages/oversight_probe.pth

ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace
CMD ["sleep", "infinity"]
