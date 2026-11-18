FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY code/ code/
CMD ["sh", "-c", "python code/generate_data.py && python code/model.py && python code/extra.py && python code/safeguard.py && python code/make_figures.py"]
