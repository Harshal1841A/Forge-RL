FROM python:3.11-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y \
    curl git build-essential ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install frontend dependencies
COPY spatial-saas/package.json /app/spatial-saas/package.json
COPY spatial-saas/package-lock.json /app/spatial-saas/package-lock.json
RUN npm --prefix /app/spatial-saas ci

# Copy entire repo and build frontend
COPY . /app/
ENV NEXT_TELEMETRY_DISABLED=1
RUN npm --prefix /app/spatial-saas run build

# Install environment as package
RUN pip install --no-cache-dir .

EXPOSE 7860

ENV PORT=7860
ENV BACKEND_PORT=8000
ENV BACKEND_INTERNAL_URL=http://127.0.0.1:8000
ENV NEXT_PUBLIC_API_URL=

CMD ["python", "scripts/run_hf_space.py"]