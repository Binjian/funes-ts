# set base image (host S)
FROM python:3.10


# set the working directory in the container
WORKDIR /app

#copy the dependcies tfiles to the working directory
COPY pyproject.toml .
COPY requirements.txt .

# copy the code to the working directory
COPY src .

RUN export HTTP_PROXY=http://127.0.0.1:20171
RUN export HTTPS_PROXY=http://127.0.0.1:20171

# install depnedencies
# RUN pip install -r requirements.txt

# RUN pip install --editable .