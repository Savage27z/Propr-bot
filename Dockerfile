FROM python:3.11-slim

# Fonts for the PnL share card (DejaVu is what pnl.py tries first).
RUN apt-get update \
    && apt-get install -y --no-install-recommends fonts-dejavu-core ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Unbuffered stdout so Railway logs stream live.
ENV PYTHONUNBUFFERED=1

CMD ["python", "bot.py"]
