# Ubuntu Linux as the base image
FROM ubuntu:22.04

# Set UTF-8 encoding
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Install Python
RUN apt-get -y update && \
    apt-get -y upgrade
# The following line ensures that the subsequent install doesn't expect user input
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y install python3-pip python3-dev

# Install spaCy
RUN pip3 install --upgrade pip
#RUN pip3 install spacy
#RUN python3 -m spacy download en_core_web_sm
RUN pip3 install -U pip setuptools wheel
RUN pip3 install -U spacy
RUN python3 -m spacy download en_core_web_sm

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Add the files into container, under QA folder, modify this based on your need
RUN mkdir /QA
ADD ask /QA
ADD answer /QA
ADD Asking /QA/Asking
RUN chmod a+x QA/ask
RUN chmod a+x QA/answer
RUN chmod a+x QA/Asking

# Change the permissions of programs
CMD ["chmod 777 /QA/*"]

# Set working dir as /QA
WORKDIR /QA
ENTRYPOINT ["/bin/bash", "-c"]
