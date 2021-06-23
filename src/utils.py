from warnings import filterwarnings

PAD = "<PAD>"
UNK = "<UNK>"
MASK = "<MASK>"
BOS = "<BOS>"
EOS = "<EOS>"


def filter_warnings():
    # "The dataloader does not have many workers which may be a bottleneck."
    filterwarnings("ignore", category=UserWarning, module="pytorch_lightning.trainer.data_loading", lineno=102)
    filterwarnings("ignore", category=UserWarning, module="pytorch_lightning.utilities.data", lineno=41)
    # "Please also save or load the state of the optimizer when saving or loading the scheduler."
    filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler", lineno=216)  # save
    filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler", lineno=234)  # load
