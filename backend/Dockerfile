# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set build-time environment variables
# ARG S3_BUCKET
# ARG S3_ACCESS_KEY
# ARG S3_SECRET_ACCESS_KEY
# ARG REDIS_ENDPOINT

# Set environment variables
# ENV S3_BUCKET=${S3_BUCKET}
# ENV S3_ACCESS_KEY=${S3_ACCESS_KEY}
# ENV S3_SECRET_ACCESS_KEY=${S3_SECRET_ACCESS_KEY}
# ENV REDIS_ENDPOINT=${REDIS_ENDPOINT}

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt /app/

# Install the required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app/

# Expose the Flask port
EXPOSE 5000

# Start the Flask app
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
