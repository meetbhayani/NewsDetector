# Lightweight Python image
FROM python:3.11-slim

WORKDIR /app

# Copy only requirements first (cached layer)
COPY requirements.txt .

# Upgrade pip and install lightweight dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Streamlit environment variables
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ENABLECORS=false

# Run Streamlit
CMD ["streamlit", "run", "app.py"]
