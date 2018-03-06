
## DEEPred 
## Dependencies
#### python 3.5.1
#### tensorflow 1.4.1
#### numpy 1.13.3


## How to run DEEPred
* Bullet list
    * Nested bullet
    * Sub-nested bullet etc
* Bullet list item 2
Install dependencies and necessary libraries that are written above.
Download DEEPred repository
Decompress all the files under "FastaFiles", "GOTermFiles" and "TrainTestDatasets"

```
./runLinux.sh 
```


Above file (around 3 GB) should be downloaded from:

http://cansyl.metu.edu.tr/ECPred.html

## Installation

Extract the files using: <br />
```
tar -xvf ECPred.tar.gz  
```
After extraction the total size of the folder will be around 10 GB. <br />

Run runLinux.sh or runMac.sh from terminal according to your OS using one of these commands: <br />
```
./runLinux.sh 
```
or <br />
```
./runMac.sh
```
These bash scripts will install necessary libraries and tools.

## Usage

Run the following command on terminal to analyze the file "filename.fasta"  <br />
```
java -jar ECPred.jar filename.fasta
```
## Input

ECPred accepts one input fasta file which may contain up to 20 proteins.

## Output

"predictionResults_filename_Date-Time.tsv": <br />

A tsv file that contains the main, subfamily, sub-subfamily and substrate class predictions together with confidence scores for each prediction; alternatively, the output can be “non-enzyme” or “no prediction”.

## Data files

"ECNumberList.txt":  <br />

A text file containing the list of EC numbers that ECPred can predict.  <br />

"test.fasta":  <br />

An example input fasta file.  <br />

"predictionResults_input_20171128-172728.tsv":  <br />

An example output prediction file (for test.fasta).

## License
DEEPred
    Copyright (C) 2018 CanSyL

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.

