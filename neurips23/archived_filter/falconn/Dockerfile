# Use the official Ubuntu Jammy image as the base image
FROM neurips23

# Update the package list and install essential packages
RUN apt-get update && \
  apt-get install -y \
  python3 \
  python3-pip \
  swig

# Install any needed packages specified in requirements.txt
RUN pip3 install numpy
RUN pip3 install h5py
RUN pip3 install scipy

# Copy the local code to the container
RUN git clone https://github.com/duckth/FALCONN.git
WORKDIR /home/app/FALCONN
RUN make python_package_install
WORKDIR /home/app

# Specify the command to run on container start
RUN python3 -c 'import falconn'

