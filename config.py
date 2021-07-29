import os
import math
import pickle
import torch
from torch import optim
from torchvision import transforms as trans
from deepvac import config, AttrDict, new, interpret, fork
from data.dataloader import FileLineEncoderDataset
from modules.model_transformer_encoder import TransformerEncoderNet

torch.backends.cudnn.benchmark=True
config = new('TransformerTrain')
## -------------------- global ------------------
config.file_path_vocab = 'data/ancient_chinese.txt'
config.file_path_train = 'data/ancient_chinese_train.txt'
config.file_path_val = 'data/ancient_chinese_val.txt'

config.pin_memory = True if torch.cuda.is_available() else False
config.seq_len = 14
config.train_batch_size = 256
config.val_batch_size = 8
config.embedding_size = 512
config.nhid = 2048
config.nlayers = 6
config.nhead = 8
config.dropout = 0.1

## -------------------- loader ------------------
config.num_workers = 3
FileLineEncoderDataset.build_vocab_from_file(config.file_path_vocab)
config.ntokens = FileLineEncoderDataset.get_token_num()
config.core.TransformerTrain.train_dataset = FileLineEncoderDataset(config, config.file_path_train, config.seq_len)
config.core.TransformerTrain.train_loader = torch.utils.data.DataLoader(config.core.TransformerTrain.train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=config.pin_memory)
config.core.TransformerTrain.val_dataset = FileLineEncoderDataset(config, config.file_path_val, config.seq_len)
config.core.TransformerTrain.val_loader = torch.utils.data.DataLoader(config.core.TransformerTrain.val_dataset, batch_size=config.val_batch_size, shuffle=False, num_workers=config.num_workers, pin_memory=config.pin_memory)

## ------------------ common ------------------
config.core.TransformerTrain.epoch_num = 200
config.core.TransformerTrain.save_num = 1
config.core.TransformerTrain.model_path = None
config.core.TransformerTrain.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config.core.TransformerTrain.output_dir = 'output'
config.core.TransformerTrain.log_every = 10
config.core.TransformerTrain.disable_git = False
config.core.TransformerTrain.model_reinterpret_cast = True
config.core.TransformerTrain.cast_state_dict_strict = False
# load script and quantize model path
#config.core.TransformerTrain.jit_model_path = "<your-script-or-quantize-model-path>"

## -------------------- training ------------------
## -------------------- tensorboard ------------------
# config.core.TransformerTrain.tensorboard_port = "6007"
# config.core.TransformerTrain.tensorboard_ip = None

## -------------------- script and quantize ------------------
# config.cast.TraceCast = AttrDict()
# config.cast.TraceCast.model_dir = "./script.pt"
# config.cast.TraceCast.static_quantize_dir = "./script.sq"
# config.cast.TraceCast.dynamic_quantize_dir = "./quantize.sq"

## -------------------- net and criterion ------------------
config.core.TransformerTrain.net = TransformerEncoderNet(config.ntokens, config.embedding_size, config.nhead, config.nhid, config.nlayers, config.dropout)
config.core.TransformerTrain.criterion = torch.nn.CrossEntropyLoss()

## -------------------- optimizer and scheduler ------------------
config.core.TransformerTrain.optimizer = torch.optim.SGD(config.core.TransformerTrain.net.parameters(), lr=0.001)
config.core.TransformerTrain.scheduler = torch.optim.lr_scheduler.MultiStepLR(config.core.TransformerTrain.optimizer, milestones=[20,40,60,80,100,120,140,160,180], gamma=0.27030 )

## ------------------ ddp --------------------
# config.dist_url = 'tcp://localhost:27030'
# config.world_size = 2
config.core.TransformerTest = config.core.TransformerTrain.clone()
config.core.TransformerTest.model_reinterpret_cast = False