# Use an official Python base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Install Jupyter and dependencies
RUN pip install --no-cache-dir notebook

# Copy requirements and install additional dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the notebooks into the container
COPY . .

# Expose the Jupyter notebook port
EXPOSE 8888

# Command to start Jupyter
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
