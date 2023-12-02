FROM ubuntu:22.04

RUN echo 'APT::Install-Suggests "0";' >> /etc/apt/apt.conf.d/00-docker
RUN echo 'APT::Install-Recommends "0";' >> /etc/apt/apt.conf.d/00-docker

RUN apt-get -y update
RUN apt-get install -y python3 python3-pip wget nginx ca-certificates build-essential git curl python-is-python3

RUN pip3 install --upgrade pip 
RUN pip3 install gunicorn pillow torch gunicorn gevent flask torchvision

        
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

# Set up the program in the image
COPY inference /opt/program

RUN ls /opt/program

RUN chmod 755 /opt/program
WORKDIR /opt/program
RUN chmod 755 serve

ENTRYPOINT ["python", "serve"]
