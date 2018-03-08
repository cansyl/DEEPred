
# DEEPred 
## Dependencies
#### python 3.5.1
#### tensorflow 1.4.1
#### numpy 1.13.3


## How to run DEEPred
* Install dependencies and necessary libraries.
* Download DEEPred repository
* Download the compressed "FeatureVectors.zip" and "Annots.zip" files from [here](goo.gl/Kd7FkU) and put them under DEEPred folder. 
* Decompress the files under the following folders
    * FastaFiles
    * GOTermFiles
    * TrainTestDatasets
    * FeatureVectors
    * Annots
* Run DEEPred script (4_layer_train.py for the 4 layered multi-task DNN or 5_layer_train.py for the 5 layered multi-task DNN ) by providing following command line arguments
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
    * number of neurons at the second layer (if you run 5_layer_train.py)

Example:
```
python 4_layer_train.py 1400 100 1000 MFGOTerms30_2_1001_2000.txt MF MFSPMAP 0.001 32 adam yes yes yes 0.6
```
* The list of GO terms trained in models (e.g. MFGOTerms30_2_1001_2000.txt in the above example) are located under "GOTermFiles" folder.
* Feature vectors that were used in training and testing are located under "FeatureVectors" folder. 
* "FastaFiles" folder includes all training and test sequences used in this study. 
* "TrainTestDatasets" includes training and testing UniProt identifiers for each GO term for all categories of GO.
* Manual experimental (all_categories_manual_experimental_annots_29_08_2017_Propagated.tsv) and all annotations (all_categories_all_annots_29_08_2017_Propagated.tsv) (includuding annotations with IEA evidence codes) are available under "Annots" folder.
* "FirstRuns" folder includes the GO terms trained at each model for feature selection and hyper-parameter optimization.

## License
DEEPred
    Copyright (C) 2018 CanSyL

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.

