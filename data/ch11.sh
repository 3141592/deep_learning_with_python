#!/bin/bash

# 11.3.1 Preparing the IMDB movie reviews data
# Download the dataset from the Standford page of Andrew Maas
curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar xvf aclImdb_v1.tar.gz -C ~/src/data/
rm -rf ~/src/data/aclImdb/train/unsup
rm aclImdb_v1.tar.gz

# "11.3.4", p. 335, Download GloVe word embeddings
wget http://nlp.stanford.edu/data/glove.6B.zip -P ~/tmp
unzip -o ~/tmp/glove.6B.zip -d ~/src/data/glove.6B
rm ~/tmp/glove.6B.zip

# 11.5.1 Beyond text classification: Sequence-to-sequence
wget http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip -P ~/tmp
unzip -o ~/tmp/spa-eng.zip -d ~/src/data
rm ~/tmp/spa-eng.zip
