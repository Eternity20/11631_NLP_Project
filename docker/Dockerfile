# Ubuntu Linux as the base image
FROM nvidia/cuda:12.0.0-base-ubuntu20.04
CMD nvidia-smi

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

#RUN pip3 install -U pip setuptools wheel
#RUN pip3 install -U spacy

COPY requirements.txt requirements.txt
#RUN pip3 install torch==1.9.0+cu120 -f https://download.pytorch.org/whl/cu120/torch_stable.html
#RUN pip3 install torch==1.9.0+cu120 -f https://download.pytorch.org/whl/cu120/torch_stable.html
#RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html


RUN pip3 install -r requirements.txt
RUN python3 -m spacy download en_core_web_sm
RUN pip3 install pyJoules
RUN pip3 install pynvml

# Add the files into container, under QA folder, modify this based on your need
RUN mkdir /QA
ADD pretrained /QA/pretrained
ADD test_cpu_answer.py /QA
ADD test_cpu_ask.py /QA
ADD test_energy_ask.py /QA
ADD test_energy_answer.py /QA
ADD ask /QA
ADD answer /QA
ADD Asking /QA/Asking
ADD mydatasets /QA/mydatasets
ADD mymodels /QA/mymodels
ADD utils.py /QA/utils.py
ADD gpu_test.py /QA/gpu_test.py
RUN chmod a+x QA/ask
RUN chmod a+x QA/answer
RUN chmod a+x QA/Asking
RUN chmod a+x QA/mydatasets
RUN chmod a+x QA/mymodels
RUN chmod a+x QA/utils.py
RUN chmod a+x QA/test_cpu_answer.py
RUN chmod a+x QA/test_cpu_ask.py
RUN chmod a+x QA/test_energy_ask.py
RUN chmod a+x QA/test_energy_answer.py

# Change the permissions of programs
CMD ["chmod 777 /QA/*"]

# Set working dir as /QA
WORKDIR /QA
ENTRYPOINT ["/bin/bash", "-c"]
