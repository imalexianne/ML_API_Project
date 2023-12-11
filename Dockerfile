# Use an official Python image
FROM python:3.8

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt ./

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

##copy model
COPY Model /app/Model

# Make port 8000 available to what fastapi will run on
EXPOSE 8000

##command to run fastapi when conatiner starts
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]