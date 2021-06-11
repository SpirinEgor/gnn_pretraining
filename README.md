# GNN Pre-training & fine-tuning
Pre-training and fine-tuning GNN model on source code 

To correct usage install all requirements:
```shell
pip install -r requirements.txt
```

## Data preparation

[src.data.preprocess](src/data/preprocess) module provide all necessary functionality to prepare data for further training.

Source code should be provided in single file with the following format:
- Examples are separated between each other by special symbol: `␢`
- Inside example source code and filename (or other label) are separated by special symbol: `₣`.

_TODO: Add information about extracting from GitHub repos_

### Code2graph

To represents code as graphs we use the approach presented in [Typilus](https://arxiv.org/abs/2004.10657).
The implementation is taken from the [fork](https://github.com/JetBrains-Research/typilus)
of original implementation with essential bug fixes.

Use [preprocess.py](src/data/preprocess/preprocess.py) to convert your data into graphs:
```shell
PYTHONPATH="." python src/data/preprocess/preprocess.py
    -d <path to file with data>
    -t <path to destination folder>
    --vocabulary
```
`--vocabulary` flag used to collect information about tokens appearance in code.

The output of preprocessing is a 3 gzipped JSONL file. Each file correspond to separate holdout (`train`, `val`, `test`). Each line is a standalone JSON that describes one graph.

### Preprocessed data

All source and preprocessed data can be obtained from this table:
| name  	| source                                                                                                                  	| preprocessed                                                                                                               	| holdout sizes (train/val/test) 	| # tokens 	|
|-------	|-------------------------------------------------------------------------------------------------------------------------	|----------------------------------------------------------------------------------------------------------------------------	|--------------------------------	|----------	|
| dev   	| [s3 link](https://s3.eu-west-1.amazonaws.com/datasets.ml.labs.aws.intellij.net/gnn_pretraining/dev/dev.txt) (3.6Mb)     	| [s3 link](https://s3.eu-west-1.amazonaws.com/datasets.ml.labs.aws.intellij.net/gnn_pretraining/dev/dev.tar.gz) (15Mb)      	| 552/185/192                    	| 12269    	|
| small 	| [s3 link](https://s3.eu-west-1.amazonaws.com/datasets.ml.labs.aws.intellij.net/gnn_pretraining/small/small.txt) (287Mb) 	| [s3 link](https://s3.eu-west-1.amazonaws.com/datasets.ml.labs.aws.intellij.net/gnn_pretraining/small/small.tar.gz) (1.2Gb) 	| 44683/14892/14934              	| 213875   	|

## Model pre-training

We use [PyTorch Lightning](https://www.pytorchlightning.ai) to implement all necessary modules for training. Thus they can be easily reused in other research works. Currently, we supported next pretraining schemes:
- Predicting `Node` and `Edge` types using `GINEConv` operator from the [Strategies for Pre-training Graph Neural Networks](https://arxiv.org/abs/1905.12265) paper. For each graph we randomly masked `Node` and `Edge` types with special token and trained model to restore them back. To start experiment run:
```shell
PYTHONPATH="." python src/train.py -c <path to YAML config file> 
```
[src.models.modules.gine_conv_encoder](./src/models/modules/gine_conv_encoder.py) contains descibed encoder model and [src.models.gine_conv_masking_pretraining](./src/models/gine_conv_masking_pretraining.py) contains complete Lightning module with pretraining description.

### Configuration

Complete configuration of model is defined by YAML config. Examples of config are stored in [config](./configs) folder.
