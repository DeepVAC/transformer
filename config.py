import os
import math
import pickle
import torch
from torch import optim
from torchvision import transforms as trans
from deepvac import AttrDict, new, interpret, fork
from data.dataloader import FileLineTranslationDataset, collate_fn, PAD_IDX
from modules.model import TransformerEncoderNet, TransformerNet

torch.backends.cudnn.benchmark=True
encoder_config = new('TransformerEncoderTrain')
## -------------------- global ------------------
encoder_config.file_path_vocab = 'data/ancient_chinese.txt'
encoder_config.file_path_train = 'data/ancient_chinese_train.txt'
encoder_config.file_path_val = 'data/ancient_chinese_val.txt'

encoder_config.pin_memory = True if torch.cuda.is_available() else False
encoder_config.seq_len = 14
encoder_config.embedding_size = 512
encoder_config.nhid = 2048
encoder_config.nlayers = 6
encoder_config.nhead = 8
encoder_config.dropout = 0.1

## -------------------- loader ------------------
encoder_config.num_workers = 3
encoder_config.core.TransformerEncoderTrain.train_dataset = FileLineTranslationDataset(encoder_config, encoder_config.file_path_train, encoder_config.file_path_vocab,[encoder_config.seq_len,0])
encoder_config.core.TransformerEncoderTrain.train_loader = torch.utils.data.DataLoader(encoder_config.core.TransformerEncoderTrain.train_dataset, batch_size=256, shuffle=True, num_workers=encoder_config.num_workers, pin_memory=encoder_config.pin_memory)
encoder_config.core.TransformerEncoderTrain.val_dataset = FileLineTranslationDataset(encoder_config, encoder_config.file_path_val, encoder_config.file_path_vocab,[encoder_config.seq_len,0])
encoder_config.core.TransformerEncoderTrain.val_loader = torch.utils.data.DataLoader(encoder_config.core.TransformerEncoderTrain.val_dataset, batch_size=8, shuffle=False, num_workers=encoder_config.num_workers, pin_memory=encoder_config.pin_memory)

## ------------------ common ------------------
encoder_config.core.TransformerEncoderTrain.epoch_num = 200
encoder_config.core.TransformerEncoderTrain.save_num = 1
encoder_config.core.TransformerEncoderTrain.model_path = None
encoder_config.core.TransformerEncoderTrain.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder_config.core.TransformerEncoderTrain.output_dir = 'output'
encoder_config.core.TransformerEncoderTrain.log_every = 10
encoder_config.core.TransformerEncoderTrain.disable_git = False
encoder_config.core.TransformerEncoderTrain.model_reinterpret_cast = False
encoder_config.core.TransformerEncoderTrain.cast_state_dict_strict = False
## -------------------- net and criterion ------------------
encoder_config.ntokens = encoder_config.core.TransformerEncoderTrain.train_dataset.get_token_num()
encoder_config.core.TransformerEncoderTrain.net = TransformerEncoderNet(encoder_config.ntokens, encoder_config.embedding_size, encoder_config.nhead, encoder_config.nhid, encoder_config.nlayers, encoder_config.dropout)
encoder_config.core.TransformerEncoderTrain.criterion = torch.nn.CrossEntropyLoss()

## -------------------- optimizer and scheduler ------------------
encoder_config.core.TransformerEncoderTrain.optimizer = torch.optim.SGD(encoder_config.core.TransformerEncoderTrain.net.parameters(), lr=0.001)
encoder_config.core.TransformerEncoderTrain.scheduler = torch.optim.lr_scheduler.MultiStepLR(encoder_config.core.TransformerEncoderTrain.optimizer, milestones=[20,40,60,80,100,120,140,160,180], gamma=0.27030 )

encoder_config.core.TransformerTest = encoder_config.core.TransformerEncoderTrain.clone()
encoder_config.core.TransformerTest.model_reinterpret_cast = False


transformer_config = new('TransformerTrain')
## -------------------- global ------------------
transformer_config.file_path_vocab = ['data/ancient_chinese.txt', 'data/morden_chinese.txt']
transformer_config.file_path_train = ['data/ancient_chinese_train.txt','data/morden_chinese_train.txt']
transformer_config.file_path_val = ['data/ancient_chinese_val.txt', 'data/morden_chinese_val.txt']

transformer_config.pin_memory = True if torch.cuda.is_available() else False
transformer_config.max_len_list = [64,128]

## -------------------- loader ------------------
transformer_config.num_workers = 3
transformer_config.core.TransformerTrain.train_dataset = FileLineTranslationDataset(transformer_config, transformer_config.file_path_train,transformer_config.file_path_vocab, transformer_config.max_len_list)
transformer_config.core.TransformerTrain.train_loader = torch.utils.data.DataLoader(transformer_config.core.TransformerTrain.train_dataset, batch_size=128, collate_fn=collate_fn, shuffle=True, num_workers=transformer_config.num_workers, pin_memory=transformer_config.pin_memory)
transformer_config.core.TransformerTrain.val_dataset = FileLineTranslationDataset(transformer_config, transformer_config.file_path_val, transformer_config.file_path_vocab,transformer_config.max_len_list)
transformer_config.core.TransformerTrain.val_loader = torch.utils.data.DataLoader(transformer_config.core.TransformerTrain.val_dataset, batch_size=8, collate_fn=collate_fn, shuffle=False, num_workers=transformer_config.num_workers, pin_memory=transformer_config.pin_memory)

## ------------------ common ------------------
transformer_config.core.TransformerTrain.epoch_num = 200
transformer_config.core.TransformerTrain.save_num = 1
transformer_config.core.TransformerTrain.model_path = None
transformer_config.core.TransformerTrain.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transformer_config.core.TransformerTrain.output_dir = 'output'
transformer_config.core.TransformerTrain.log_every = 10
transformer_config.core.TransformerTrain.disable_git = False
transformer_config.core.TransformerTrain.model_reinterpret_cast = False
transformer_config.core.TransformerTrain.cast_state_dict_strict = False
## -------------------- net and criterion ------------------
transformer_config.src_ntokens = transformer_config.core.TransformerTrain.train_dataset.get_token_num(0)
transformer_config.tgt_ntokens = transformer_config.core.TransformerTrain.train_dataset.get_token_num(1)
transformer_config.core.TransformerTrain.net = TransformerNet(num_encoder_layers=6,num_decoder_layers=6,emb_size=512, nhead=8, src_vocab_size=transformer_config.src_ntokens, tgt_vocab_size=transformer_config.tgt_ntokens, dim_feedforward = 512, dropout = 0.1)
transformer_config.core.TransformerTrain.criterion = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

## -------------------- optimizer and scheduler ------------------
transformer_config.core.TransformerTrain.optimizer = torch.optim.Adam(transformer_config.core.TransformerTrain.net.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
transformer_config.core.TransformerTrain.scheduler = torch.optim.lr_scheduler.MultiStepLR(transformer_config.core.TransformerTrain.optimizer, milestones=[100,140,180], gamma=0.27030 )

transformer_config.core.TransformerTest = transformer_config.core.TransformerTrain.clone()
transformer_config.core.TransformerTest.model_reinterpret_cast = False