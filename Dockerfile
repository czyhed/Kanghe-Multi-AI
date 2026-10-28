FROM python:3.13-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY alembic ./alembic
COPY alembic.ini ./

EXPOSE 8000

# 基础设施地址通过环境变量传入，见 .env.example
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
