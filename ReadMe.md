
# DEEPred 
## Descriptions of Folders and Files Under This Repository
* FastaFiles
    * Includes training and test sequences for each GO category. For example, the fasta file of training and test sequences for molecular function category is "MF_deepred_training_sequences.fasta.zip" and fasta file of CAFA benchmarking protein sequences for molecular function category is "mfo_cafa_only_annot_prot.fasta.zip".
* FirstRuns
    * Includes the GO terms trained in each model for feature selection and hyper-parameter optimization. File names include information about the level of GO terms, number of protein association range and the model number. For example, "MFGOTerms30_4_201_300_2.txt" includes the GO terms trained on the fourth level of GO DAG that have protein associations between 201 and 300 and this is the second model trained on the fourth level. Each files includes the trained GO terms and number of protein associations of the correponding GO terms in a tab-separated format.
* GOTermFiles
    * Includes the GO terms trained in each model. There are three zipped folders under this directory (one file for each GO category). Each unzipped folder includes a sub-folder named "5" and the model files are included in this folder. The format of the files are same as above (explained under FirstRuns).
* FeatureVectors (This folder is not available under repository. It should be downloaded from [here](http://goo.gl/Kd7FkU) .
    * This folder includes feature vectors that were used for training and testing of the system. For example, "Parsed_PAACFeatures_uniprot_training_test_set.txt" file contains PAAC feature vectors and "Parsed_BPSPMAPFeatures_CAFA2.txt" includes SPMAP feature vectors for BP CAFA benchmarking protein sequences.
* TrainTestDatasets
    * Includes training and testing UniProt identifiers.There are three zipped folders under this directory (one file for each GO category). The unzipped folders include two files (train and test) for each GO term trained in the corresposding category. Example:  train_GO:0043175.ids and test_GO:0043175.ids
* Annots
    * Includes annotation files. Manual experimental annotations are stored in  "all_categories_manual_experimental_annots_29_08_2017_Propagated.tsv" file and all annotations (including annotations with IEA evidence codes) are stored in "all_categories_all_annots_29_08_2017_Propagated.tsv" file.


         
## Dependencies
#### [python 3.5.1](https://www.python.org/downloads/release/python-351/)
#### [tensorflow 1.4.1](https://github.com/tensorflow/tensorflow/releases/tag/v1.4.1)
#### [numpy 1.13.3](https://pypi.python.org/pypi/numpy/1.13.3)


## How to run DEEPred
* Install dependencies and necessary libraries.
* Download DEEPred repository
* Download the compressed "FeatureVectors.zip" and "Annots.zip" files from [here](http://goo.gl/Kd7FkU) and put them under DEEPred folder. 
* Decompress the files under the following folders
    * FastaFiles
    * GOTermFiles
    * TrainTestDatasets
    * FeatureVectors
    * Annots
* Run DEEPred script (4_layer_train.py) by providing following command line arguments:
    * number of neurons at the first layer
    * number of neurons at the second layer
    * number of epochs
    * GO terms to be trained in a model (.txt file Could be any file under GOTermFiles)
    * GO category (MF, BP, CC)
    * Type of feature (could be PAAC, CTriad, MFSPMAP, BPSPMAP, CCSPMAP)
    * Learning rate
    * Mini-batch size
    * optimizer type (adam, momentum, rmsprop)
    * normalize_inputs (yes, no)
    * batch_normalization (yes, no)
    * learning_rate_decay (yes, no)
    * drop_out_rate


Example:
```
python 4_layer_train.py 1400 100 1000 MFGOTerms30_4_201_300_2.txt MF MFSPMAP 0.001 32 adam yes yes yes 0.6
```
## Output of the script
The prediction scores and the performance results for the test sequences are printed as the output.
## License
DEEPred
    Copyright (C) 2018 CanSyL

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.

