FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HOME=/cache/huggingface
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY task-3/requirements.txt task-3/requirements.txt
COPY task-4/requirements.txt task-4/requirements.txt
COPY task-5/requirements.txt task-5/requirements.txt
# The embedding encoder runs on CPU here; Ollama uses the Windows GPU.
RUN pip install --no-cache-dir 'torch>=2.2,<3' --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r task-5/requirements.txt
COPY task-3/common.py task-3/search.py task-3/
COPY task-4/rag.py task-4/prompts.py task-4/
COPY task-5/*.py task-5/
COPY task-5/questions.json task-5/malicious.txt task-5/
CMD ["python", "task-5/bot.py"]
