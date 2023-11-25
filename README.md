# E2EATP
A fast and high-accuracy protein-ATP binding residue prediction via protein language model

Identifying the ATP-binding sites of proteins is fundamentally important to uncover the mechanisms of protein functions and explore drug discovery. Many computational methods are proposed to predict ATP-binding sites. However, due to the limitation of the quality of feature representation, the prediction performance still has a big room for improvements. In this study, we propose an end-to-end deep learning model, E2EATP, to dig out more discriminative information from protein sequence for improving the ATP-binding site prediction performance. Concretely, we employ a pre-trained deep learning-based protein language model (ESM2) to automatically extract high-latent discriminative representations of protein sequences relevant for protein functions. Based on ESM2, we design a residual convolutional neural network to learn protein-ATP binding site prediction model. Furthermore, a weighted focal loss function is used to reduce the negative impact of imbalanced data on the model training stage. Experimental results on the independent testing data set demonstrate that E2EATP could achieve a Matthew’s correlation coefficient value of 0.668 and the AUC score of 0.914, which are higher than that of most existing state-of-the-art prediction methods. <b>The speed (about 0.05 second per protein) of E2EATP is much faster than the other existing prediction method. We have predicted all 207,892 human proteins in UniProt (up to 2023-07-10)</b>. 

## Pre-requisite:
    - Python3, Java, Anaconda3
    - Linux system

## Installation:

* Download this repository at https://github.com/jun-csbio/e2eatp.git. Then, uncompress it and run the following command lines on Linux System.

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

* The prediction result file (e.g., "3J8YK.pred") of each protein (e.g., 3J8YK) in your input fasta file (-seq_fa) could be found in the folder which you input as "-sf".
* There are four columns in each prediction result file. The 1st column is the residue index. The 2nd column is the residue type. The 3rd column is the predicted probablity of the corresponding residue belonging to the class of ATP-binding residues. The 4th column is the prediction result ('B' and 'N' mean the predicted ATP-binding and non-ATP-binding residue, respectively). For example:

~~~
Index    AA    Prob.    State
    0     A    0.001    N
    1     E    0.000    N
    2     S    0.007    N
    3     N    0.001    N
    4     I    0.000    N
    5     K    0.000    N
    6     V    0.000    N
    7     M    0.003    N
    8     C    0.000    N
    9     R    0.984    B
   10     F    0.000    N
   11     R    0.993    B
   12     P    0.990    B
   13     L    0.001    N
   14     N    0.001    N
   15     E    0.000    N
   16     S    0.005    N
   17     E    0.000    N
   18     V    0.000    N
   19     N    0.001    N
~~~

## Predicted Database on Uniprot
* If you have installed this package, you can find that there is a predicted ATP-binding residues database for all 207,892 human proteins in UniProt (up to 2023-07-10).

~~~
  $ cd e2eatp-main/uniprot/
~~~

## Update History:

- Add a new data set     2023-11-26
- First release          2023-07-12

## Tips

* <b>This package is only free for academic use</b>. If you have any question, please email Jun Hu: junh_cs@126.com

## References
[1] Bing Rao, Xuan Yu, Jie Bai and Jun Hu. E2EATP: fast and high-accuracy protein-ATP Binding Residue Prediction via Protein Language Model Embedding. sumitted.
