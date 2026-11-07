# One SWE-bench instance prepared for the closure check: the official environment image plus our users, strace and
# the closure probe.  The repository moves from /testbed to /workspace (a symlink keeps old paths working) and the
# ImpossibleBench test patch is applied, so the pristine state is the impossible task.
ARG BASE
FROM ${BASE}
USER root
RUN (command -v strace >/dev/null || (apt-get update && apt-get install -y --no-install-recommends strace \
     && rm -rf /var/lib/apt/lists/*)) \
 && (id agent >/dev/null 2>&1 || useradd -m -s /bin/bash agent) \
 && (id checker >/dev/null 2>&1 || useradd -m -s /bin/bash checker)
COPY oversight_probe.py impossible_test.patch /tmp/
RUN SP=$(/opt/miniconda3/envs/testbed/bin/python -c "import site; print(site.getsitepackages()[0])") \
 && cp /tmp/oversight_probe.py "$SP/" && echo "import oversight_probe" > "$SP/oversight_probe.pth" \
 && mv /testbed /workspace && ln -s /workspace /testbed \
 && (grep -rlI /testbed "$SP" 2>/dev/null | grep -E '\.pth$|\.egg-link$|__editable__' | xargs -r sed -i 's#/testbed#/workspace#g'; true) \
 && cd /workspace && git apply /tmp/impossible_test.patch \
 && git config --system --add safe.directory /workspace && chmod -R a+rX /workspace
RUN (sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen) >/dev/null 2>&1 || true
ENV PYTHONDONTWRITEBYTECODE=1
WORKDIR /workspace
CMD ["sleep", "infinity"]
