# GNN Pre-training & fine-tuning
[WIP] Pre-training and fine-tuning GNN model on source code 

To correct usage install all requirements:
```shell
pip install -r requirements.txt
```

## Data preparation

[src.data.preprocess](src/data/preprocess) module provide all necessary functionality to prepare data for further training.

Source code should be provided in single file with the following format:
- Examples are separated between each other by special symbol: `␢`
- Inside example source code and filename (or other label) are separated by special symbol: `₣`.

You can download already prepared source code using these links:
- [small set of python files from GitHub](https://s3-eu-west-1.amazonaws.com/datasets.ml.labs.aws.intellij.net/gnn_pretraining/train_small.txt.tar.gz)

### Code2graph

To represents code as graphs we use the approach presented in [Typilus](https://arxiv.org/abs/2004.10657).
The implementation is taken from the [fork](https://github.com/JetBrains-Research/typilus)
of original implementation with essential bug fixes.

Use [preprocess.py](src/data/preprocess/preprocess.py) to convert your data into graphs:
```shell
PYTHONPATH="." python data/preprocess/preprocess.py -d <path to file with data>
```

The output of preprocessing is a gzipped JSONL file. Each line is a standalone JSON that describes one graph.
You can download preprocessed data using these links:
- [small set of python files from GitHub]()
