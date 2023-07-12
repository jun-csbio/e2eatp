# E2EATP
A fast and high-accuracy protein-ATP binding residue prediction via protein language model

## Pre-requisite:
    - Python3, Java, Anaconda3
    - Linux system

## Installation:

*Download this repository at https://github.com/jun-csbio/e2eatp.git. Then, uncompress it and run the following command lines on Linux System.

~~~
  $ cd e2eatp-main
  $ chmod -R 777 ./install.sh
  $ ./install.sh
  $ java -jar FileUnion.jar ./uniprot/humain/ ./uniprot/uniprot_human_2023_07_10.tar.gz
  $ rm -rf ./uniprot/humain/
  $ tar zxvf ./uniprot/uniprot_human_2023_07_10.tar.gz
~~~

* If the package cannot directly work correct on your computational cluster, you should install the dependencies via running the following commands:

~~~
  $ cd e2eatp-main
  $ pip install -r requirements.txt
~~~

## Run
~~~
  $ python predict.py -sf example/results/ -seq_fa example/seq.fa
~~~

## Result

* The prediction result of E2EATP for each protein (e.g., 5DN1A) in your input fasta file (-seq_fa) could be found in the folder which you input as '-sf'.

## Update History:

- First release     2023-07-12

## References
[1] Bing Rao and Jun Hu. E2EATP: fast and high-accuracy protein-ATP Binding Residue Prediction via Protein Language Model Embedding. sumitted.
