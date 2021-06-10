FROM jenkins/jenkins:lts

USER root
RUN wget https://dvc.org/deb/dvc.list -O /etc/apt/sources.list.d/dvc.list && apt update && apt install dvc
USER jenkins
RUN echo "Hola"
