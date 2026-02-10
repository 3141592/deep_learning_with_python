#!/bin/bash
# input_dir = "~/src/data/images/"
# target_dir = "~/src/data/annotations/trimaps/"

# WGET the data
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz -P ~/tmp/
wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz -P ~/tmp/

# Extract to data directory
tar xvf ~/tmp/images.tar.gz -C ~/src/data/
tar xvf ~/tmp/annotations.tar.gz -C ~/src/data/

# Delete the archives
rm -rf ~/tmp/images.tar.gz*
rm -rf ~/tmp/annotations.tar.gz*

