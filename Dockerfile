FROM node:22-bookworm-slim AS frontend-build
WORKDIR /app/webapp/frontend

COPY webapp/frontend/package.json webapp/frontend/package-lock.json ./
RUN npm ci

COPY webapp/frontend ./
RUN npm run build


FROM python:3.13-slim AS runtime
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY mmrag ./mmrag
COPY webapp ./webapp
COPY run_pipeline.py ./
COPY .env.example ./

COPY --from=frontend-build /app/webapp/frontend/dist ./webapp/frontend/dist

EXPOSE 10000

CMD ["python", "-m", "uvicorn", "webapp.app:app", "--host", "0.0.0.0", "--port", "10000"]
