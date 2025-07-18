# 1. The Base Image: Start with an official PyTorch image with CUDA support.
# This saves us from manually installing Python, CUDA, and PyTorch.
FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# 2. Set up the working environment and install system dependencies.
# - Set the working directory inside the container to /app.
# - Update package lists and install system libraries needed by our Python packages.
#   - libsndfile1 is for audio file handling (used by soundfile).
#   - ffmpeg is a robust utility for audio/video conversion (good to have).
#   - git is useful for various operations.
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# 3. Install Python dependencies using a cached layer.
# First, copy only the requirements file. This allows Docker to cache this layer.
# The expensive 'pip install' step will only re-run if requirements.txt changes,
# not every time you change your server.py code.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your application code into the container.
# This copies server.py and any other files into the /app directory.
COPY . .

# 5. Expose the port the server will run on.
# This tells Docker that the container listens on port 8000.
EXPOSE 8000

# 6. Define the command to run when the container starts.
# This executes your FastAPI server using uvicorn.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

