FROM tensorflow/tensorflow:1.15.0-py3

ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true
ENV LANG=C.UTF-8
RUN mkdir /gpt-2

RUN apt update
RUN apt install git -y
RUN git clone --depth 1 https://github.com/openai/gpt-2.git /gpt-2

WORKDIR /gpt-2
RUN pip3 install -r requirements.txt

#Uncomment the model to download. Should match the model_to_use setting in gpt2-api.py
# RUN python3 download_model.py 124M
# RUN python3 download_model.py 355M
# RUN python3 download_model.py 774M
RUN python3 download_model.py 1558M

RUN apt install wget -y
ADD gpt2-api.py /gpt-2/src
ADD requirements.txt .
RUN pip install -r requirements.txt

WORKDIR /gpt-2/src
CMD python3 gpt2-api.py
