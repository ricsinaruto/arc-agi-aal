FROM node:22.19.0-bookworm-slim AS node
FROM python:3.12.11-slim-bookworm
COPY --from=node /usr/local/bin/node /usr/local/bin/node
COPY --from=node /usr/local/lib/node_modules /usr/local/lib/node_modules
RUN ln -s /usr/local/lib/node_modules/npm/bin/npm-cli.js /usr/local/bin/npm
RUN apt-get update && apt-get install -y --no-install-recommends libseccomp2 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml /app/
RUN python -c "import tomllib,subprocess,sys; p=tomllib.load(open('pyproject.toml','rb'))['project']; subprocess.check_call([sys.executable,'-m','pip','install',*p['dependencies'],*p['optional-dependencies']['dev'],'hatchling'])"
COPY src /app/src
RUN pip install --no-deps --no-build-isolation .
COPY pi/package*.json /app/pi/
RUN cd /app/pi && npm ci --ignore-scripts
COPY pi /app/pi
COPY scripts/qwen /app/scripts/qwen
ENV OPENBLAS_NUM_THREADS=1
CMD ["python", "-m", "pytest", "tests", "-q"]
