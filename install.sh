#!/bin/bash

cd esm2m
chmod 777 ./download.sh
bash ./download.sh

cd ..
java -jar FileUnion.jar ./uniprot/human/ ./uniprot_human_2023_07_10.tar.gz
rm -rf ./uniprot/human
mv uniprot_human_2023_07_10.tar.gz ./uniprot/

cd uniprot
tar -zxvf uniprot_human_2023_07_10.tar.gz
rm -f uniprot_human_2023_07_10.tar.gz
