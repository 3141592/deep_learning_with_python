#!/bin/bash
mkdir -p ~/src/data

kaggle competitions download -c dogs-vs-cats
mv dogs-vs-cats.zip ~/tmp/

mkdir ~/src/data/dogs-vs-cats/
unzip ~/tmp/dogs-vs-cats.zip -d ~/src/data/dogs-vs-cats/
unzip ~/src/data/dogs-vs-cats/train.zip -d ~/src/data/dogs-vs-cats/

rm ~/tmp/dogs-vs-cats.zip
