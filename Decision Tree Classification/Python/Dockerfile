# Multi-stage Dockerfile for Decision Tree Classification Application
# Stage 1: Base image with Python
FROM python:3.10-slim

# ============================================================================
# METADATA
# ============================================================================
LABEL maintainer="ML Classification Project"
LABEL description="Decision Tree Classification Web Application"
LABEL version="1.0"

# ============================================================================
# SET WORKING DIRECTORY
# ============================================================================
# Set the working directory inside the container
WORKDIR /app

# ============================================================================
# INSTALL SYSTEM DEPENDENCIES
# ============================================================================
# Update package list and install required system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ============================================================================
# COPY PROJECT FILES
# ============================================================================
# Copy requirements file first (for better Docker layer caching)
COPY requirements.txt .

# Copy application files
COPY app.py .
COPY decision_tree_classification.py .

# Copy the dataset
COPY Social_Network_Ads.csv .

# ============================================================================
# INSTALL PYTHON DEPENDENCIES
# ============================================================================
# Upgrade pip to latest version
RUN pip install --no-cache-dir --upgrade pip

# Install Python packages from requirements file
RUN pip install --no-cache-dir -r requirements.txt

# ============================================================================
# EXPOSE PORT
# ============================================================================
# Expose port 5000 (Flask default port)
EXPOSE 5000

# ============================================================================
# HEALTH CHECK
# ============================================================================
# Define a health check to verify the application is running
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# ============================================================================
# ENVIRONMENT VARIABLES
# ============================================================================
# Set Flask environment to production
ENV FLASK_ENV=production
ENV PYTHONUNBUFFERED=1

# ============================================================================
# RUN APPLICATION
# ============================================================================
# Command to run the Flask application
CMD ["python", "app.py"]
