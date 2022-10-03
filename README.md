# Context-Aware Generative Models for Multi-Domain Proteins using Transformers

## Code Organization

### src folder

The folder src contains the source code of the paper:
- `ProteinsDataset.py` contains the function linked with the handling of the data.
- `ProteinsTransformer.py` contains the code for the Transformer model
- `MatchingLoss.py` contains our losses 
- `ardca.py` functions related to the arDCA model (wrapping the julia library ArDCA.jl)
- `DCA.py` functions related with the contact prediction
- `utils.py` the other functions

### files.config.json

shallow.config.json, large.config.json and large_renyi.config.json contains the hyperparameter of the shallow model, the large Transformer and the large Transformer using the entropic regularization. You can easily use a new set of hyperparameter by modifying one of this file or creating your own json file.

### models

models are saved in the models folder.

### data

We provide two datasets in the forlder data to test the code: PF00207_PF07677 & PF03171_PF14226.

### training file

The training is controlled from the `train.py` and the `train.sh`.
The arguments are:
	`--trainset` : path to train dataset  
	`--valset` : path to testset
	`--save` : path for saving the model
	`--load` : path to load to a model and continue training it
	`--modelconfig` : path to the json file with the hyperparameters
	`--outputfile` : output file where scores are written during training


## Run the code 
You can either use the python command:
```
python -m train --trainset "data/pMSA_PF00207_PF07677_train.csv" --valset "data/pMSA_PF00207_PF07677_val.csv" --save "models/saved_PF00207_PF07677.pth.tar" --load "" --modelconfig "shallow.config.json" --outputfile "output.txt"
```
or use the shell script `train.sh`.