# Use official Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir flask pandas numpy scikit-learn tensorflow

# Expose port
EXPOSE 5000

# Run the app
CMD ["python", "app.py"]

