FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl git build-essential ca-certificates gnupg \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Support both spatial-saas and spatial_saas directory names
COPY . /app/

# Build frontend — try both directory names
RUN if [ -d "spatial-saas" ]; then \
        npm --prefix spatial-saas ci && npm --prefix spatial-saas run build; \
    elif [ -d "spatial_saas" ]; then \
        npm --prefix spatial_saas ci && npm --prefix spatial_saas run build; \
    fi

RUN pip install --no-cache-dir -e . 2>/dev/null || true

EXPOSE 7860

# FastAPI on 7860 (OpenEnv port), Next.js on 3000 (internal)
ENV PORT=7860
ENV BACKEND_PORT=7860
ENV FRONTEND_PORT=3000
ENV NEXT_PUBLIC_API_URL=http://localhost:7860
ENV NEXT_TELEMETRY_DISABLED=1

CMD ["python", "scripts/run_hf_space.py"]