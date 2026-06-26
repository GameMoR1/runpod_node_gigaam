FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=8000

WORKDIR /app

RUN apt-get update -y \
 && apt-get install -y --no-install-recommends ffmpeg ca-certificates git \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
COPY requirements-gigaam-deps.txt /app/requirements-gigaam-deps.txt
COPY constraints.txt /app/constraints.txt

RUN python -m pip install --upgrade pip setuptools wheel \
 && pip install -c /app/constraints.txt -r /app/requirements-gigaam-deps.txt \
 && python -c "import pathlib; src=pathlib.Path('/app/requirements.txt').read_text(encoding='utf-8').splitlines(); out=[]; \
  [out.append(l) for l in src if (not l.strip()) or l.lstrip().startswith('#') or (not l.strip().lower().startswith('gigaam'))]; \
  pathlib.Path('/tmp/requirements.no_gigaam.txt').write_text('\\n'.join(out)+'\\n', encoding='utf-8')" \
 && pip install -c /app/constraints.txt -r /tmp/requirements.no_gigaam.txt \
 && pip install --no-deps "gigaam @ git+https://github.com/salute-developers/GigaAM.git" \
 && python -c "import hydra; import gigaam; print('gigaam import ok')"

COPY . /app

EXPOSE 8000

CMD ["python", "run.py"]
