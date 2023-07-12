# E2EATP
A fast and high-accuracy protein-ATP binding residue prediction via protein language model

Identifying the ATP-binding sites of proteins is fundamentally important to uncover the mechanisms of protein functions and explore drug discovery. Many computational methods are proposed to predict ATP-binding sites. However, due to the limitation of the quality of feature representation, the prediction performance still has a big room for improvements. In this study, we propose an end-to-end deep learning model, E2EATP, to dig out more discriminative information from protein sequence for improving the ATP-binding site prediction performance. Concretely, we employ a pre-trained deep learning-based protein language model (ESM2) to automatically extract high-latent discriminative representations of protein sequences relevant for protein functions. Based on ESM2, we design a residual convolutional neural network to learn protein-ATP binding site prediction model. Furthermore, a weighted focal loss function is used to reduce the negative impact of imbalanced data on the model training stage. Experimental results on the independent testing data set demonstrate that E2EATP could achieve a Matthew’s correlation coefficient value of 0.668 and the AUC score of 0.914, which are higher than that of most existing state-of-the-art prediction methods. <b>The speed (about 0.05 second per protein) of E2EATP is much faster than the other existing prediction method</b>. 

## Pre-requisite:
    - Python3, Java, Anaconda3
    - Linux system

## Installation:

*Download this repository at https://github.com/jun-csbio/e2eatp.git. Then, uncompress it and run the following command lines on Linux System.

~~~
  $ cd e2eatp-main
  $ chmod 777 ./install.sh
  $ ./install.sh
~~~

* If the package cannot work correctly on your computational cluster, you should install the dependencies via running the following commands:

~~~
  $ cd e2eatp-main
  $ pip install -r requirements.txt
~~~

## Run example
~~~
  $ python predict.py -sf example/results/ -seq_fa example/seq.fa
~~~

## Result

* The prediction result of E2EATP for each protein (e.g., 5DN1A) in your input fasta file (-seq_fa) could be found in the folder which you input as '-sf'.

## Predicted Database on Uniprot
* If you have install this repository/package, you can find that there is a predicted ATP-binding residues database for all human proteins in UniProt.

~~~
  $ cd e2eatp-main/uniprot/
~~~

## Update History:

- First release     2023-07-12

## References
[1] Bing Rao and Jun Hu. E2EATP: fast and high-accuracy protein-ATP Binding Residue Prediction via Protein Language Model Embedding. sumitted.
