FROM python:3.10

# Set the working directory inside the container
WORKDIR /app

# Copy project files into the container
COPY . .

# Install required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Optionally generate a data drift report at startup
# Comment this line if you don't want to generate the report on container start
CMD ["sh", "-c", "python app.py data/clean_train.csv && uvicorn app:app --host 0.0.0.0 --port 8000"]

