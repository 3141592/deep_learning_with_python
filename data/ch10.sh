#!/bin/bash

# 10.2 A temperature-forecasting example
wget https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip -P ~/tmp
unzip -o ~/tmp/jena_climate_2009_2016.csv.zip -d ~/src/data/
rm ~/tmp/jena_climate_2009_2016.csv.zip

