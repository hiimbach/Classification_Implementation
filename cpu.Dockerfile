FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
COPY environment_cpu.yaml .
RUN conda env create -f environment_cpu.yaml .