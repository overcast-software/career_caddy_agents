# syntax=docker/dockerfile:1
FROM python:3.13-slim AS base

# Set to "true" to install the Camoufox Firefox binary (~700 MB).
# Required by the browser-mcp service; not needed for the pipeline-only image.
ARG INSTALL_CAMOUFOX=false

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Base build tools (always needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
  && rm -rf /var/lib/apt/lists/*

# System libraries required by Camoufox/Firefox — only installed when INSTALL_CAMOUFOX=true
RUN if [ "$INSTALL_CAMOUFOX" = "true" ]; then \
    apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libxkbcommon0 \
    libpango-1.0-0 \
    libcairo2 \
    libasound2 \
    libx11-xcb1 \
    libxcb-dri3-0 \
    libgtk-3-0 \
    libdbus-glib-1-2 \
    libxt6 \
    && rm -rf /var/lib/apt/lists/*; \
  fi

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

ENV CAMOUFOX_DATA_DIR=/opt/camoufox

# Install Python dependencies — separate layer so source changes don't bust this cache
COPY pyproject.toml uv.lock /app/
RUN uv sync --frozen --no-dev --no-install-project

# Download Camoufox Firefox binary — only when INSTALL_CAMOUFOX=true
# Placed here (before COPY . /app) so it is only invalidated by lockfile changes
RUN if [ "$INSTALL_CAMOUFOX" = "true" ]; then uv run python -m camoufox fetch; fi

# Copy source and finish install
COPY . /app
RUN uv sync --frozen --no-dev

# No default CMD — overridden per-service in docker-compose
