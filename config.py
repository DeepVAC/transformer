import os
import math
import pickle
import torch
from torch import optim
from torchvision import transforms as trans
from deepvac import AttrDict, new, interpret, fork
from data.dataloader import FileLineEncoderDataset
from modules.model_transformer_encoder import TransformerEncoderNet

torch.backends.cudnn.benchmark=True
encoder_config = new('TransformerEncoderTrain')
## -------------------- global ------------------
encoder_config.file_path_vocab = 'data/ancient_chinese.txt'
encoder_config.file_path_train = 'data/ancient_chinese_train.txt'
encoder_config.file_path_val = 'data/ancient_chinese_val.txt'

encoder_config.pin_memory = True if torch.cuda.is_available() else False
encoder_config.seq_len = 14
encoder_config.train_batch_size = 256
encoder_config.val_batch_size = 8
encoder_config.embedding_size = 512
encoder_config.nhid = 2048
encoder_config.nlayers = 6
encoder_config.nhead = 8
encoder_config.dropout = 0.1

## -------------------- loader ------------------
encoder_config.num_workers = 3
FileLineEncoderDataset.build_vocab_from_file(encoder_config.file_path_vocab)
encoder_config.ntokens = FileLineEncoderDataset.get_token_num()
encoder_config.core.TransformerEncoderTrain.train_dataset = FileLineEncoderDataset(encoder_config, encoder_config.file_path_train, encoder_config.seq_len)
encoder_config.core.TransformerEncoderTrain.train_loader = torch.utils.data.DataLoader(encoder_config.core.TransformerEncoderTrain.train_dataset, batch_size=encoder_config.train_batch_size, shuffle=True, num_workers=encoder_config.num_workers, pin_memory=encoder_config.pin_memory)
encoder_config.core.TransformerEncoderTrain.val_dataset = FileLineEncoderDataset(encoder_config, encoder_config.file_path_val, encoder_config.seq_len)
encoder_config.core.TransformerEncoderTrain.val_loader = torch.utils.data.DataLoader(encoder_config.core.TransformerEncoderTrain.val_dataset, batch_size=encoder_config.val_batch_size, shuffle=False, num_workers=encoder_config.num_workers, pin_memory=encoder_config.pin_memory)

## ------------------ common ------------------
encoder_config.core.TransformerEncoderTrain.epoch_num = 200
encoder_config.core.TransformerEncoderTrain.save_num = 1
encoder_config.core.TransformerEncoderTrain.model_path = None
encoder_config.core.TransformerEncoderTrain.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder_config.core.TransformerEncoderTrain.output_dir = 'output'
encoder_config.core.TransformerEncoderTrain.log_every = 10
encoder_config.core.TransformerEncoderTrain.disable_git = False
encoder_config.core.TransformerEncoderTrain.model_reinterpret_cast = True
encoder_config.core.TransformerEncoderTrain.cast_state_dict_strict = False
## -------------------- net and criterion ------------------
encoder_config.core.TransformerEncoderTrain.net = TransformerEncoderNet(encoder_config.ntokens, encoder_config.embedding_size, encoder_config.nhead, encoder_config.nhid, encoder_config.nlayers, encoder_config.dropout)
encoder_config.core.TransformerEncoderTrain.criterion = torch.nn.CrossEntropyLoss()

## -------------------- optimizer and scheduler ------------------
encoder_config.core.TransformerEncoderTrain.optimizer = torch.optim.SGD(encoder_config.core.TransformerEncoderTrain.net.parameters(), lr=0.001)
encoder_config.core.TransformerEncoderTrain.scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_config.core.TransformerEncoderTrain.optimizer, milestones=[20,40,60,80,100,120,140,160,180], gamma=0.27030 )

encoder_config.core.TransformerTest = encoder_config.core.TransformerEncoderTrain.clone()
encoder_config.core.TransformerTest.model_reinterpret_cast = False