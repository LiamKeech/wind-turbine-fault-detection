FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py ./
COPY src/ ./src/

# data/ is intentionally NOT copied here — it's Git LFS-tracked and mounted
# as a volume at runtime (see docker-compose.yml) so the image doesn't bake
# in multi-GB CSVs/model weights or risk copying unresolved LFS pointers.
RUN mkdir -p data/raw data/processed

RUN useradd --create-home --uid 1000 app && chown -R app:app /app
USER app

ENTRYPOINT ["python", "main.py"]
CMD ["--help"]
