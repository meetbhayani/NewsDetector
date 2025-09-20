# Use slim Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy lightweight requirements first to leverage caching
COPY requirements.txt .

# Upgrade pip and install lightweight dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Streamlit environment variables
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ENABLECORS=false

# Command to run Streamlit
CMD ["streamlit", "run", "app.py"]
