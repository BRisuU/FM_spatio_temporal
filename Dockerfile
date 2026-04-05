FROM nvidia/cuda:12.0.1-devel-ubuntu20.04

# Install dependencies
RUN apt update && apt install -y python3 python3-pip git

# Copy the requirements file
COPY requirements.txt /workspace/

# Install the dependencies
RUN pip3 install -r /workspace/requirements.txt

WORKDIR /workspace
CMD ["bash"]