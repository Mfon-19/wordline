# Worldline web server: viewer, search API and the OpenCV 5 pipeline. Built for arm64 (AWS Graviton).
FROM python:3.14-slim

# libGL and GLib are needed by the opencv-python wheel; ffmpeg support is bundled in the wheel
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# CPU-only PyTorch first: the default Linux wheels pull in several GB of CUDA libraries we never use
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu \
        "$(grep '^torch==' requirements.txt)" "$(grep '^torchvision==' requirements.txt)" \
    && pip install --no-cache-dir -r requirements.txt

# Bake the search model into the image so the server starts without a network download
ENV HF_HOME=/app/.hf
COPY worldline/ worldline/
RUN python -c "from worldline import embed; embed.model()"

COPY web/ web/
RUN useradd --system --home /app worldline && chown -R worldline /app
USER worldline
EXPOSE 8765
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/api/health', timeout=4)"
CMD ["python", "-m", "worldline.server", "--host", "0.0.0.0", "--port", "8765"]
