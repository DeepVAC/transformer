import sys
import os
import math
import time
import numpy as np
import cv2
import torch
from torch import nn
from torch import optim
from torch.utils.data import dataloader
from deepvac import LOG, DeepvacTrain
from modules.utils import generate_square_subsequent_mask, getPosAndPaddingMask
from data.dataloader import UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX

class TransformerEncoderTrain(DeepvacTrain):
    def __init__(self, deepvac_config):
        super(TransformerEncoderTrain, self).__init__(deepvac_config)
        self.src_mask = generate_square_subsequent_mask(deepvac_config.seq_len).to(self.config.device)
        self.ntokens = deepvac_config.ntokens

    def debugOutput(self):
        input = self.config.sample.transpose(0,1)
        output = self.config.output.transpose(0,1)
        output = torch.argmax(output, dim=2)
        LOG.logI("input shape: {}".format(input.shape))
        LOG.logI("output shape: {}".format(output.shape))

        for t in input:
            src = self.config.train_dataset.index2string(t.tolist())
            LOG.logI("SRC: {}".format(src))
        
        for t in output:
            target = self.config.train_dataset.index2string(t.tolist())
            LOG.logI("TARGET: {}".format(target))

    def doForward(self):
        self.config.sample = self.config.sample.transpose(0,1)
        self.config.target = self.config.target.reshape(-1)
        self.config.output = self.config.net(self.config.sample, self.src_mask)
        if self.config.is_train:
            self.config.output = self.config.output.view(-1, self.ntokens)
        else:
            self.debugOutput()

    def doLoss(self):
        if not self.config.is_train:
            return
        super(TransformerEncoderTrain, self).doLoss()

class TransformerTrain(DeepvacTrain):
    def doForward(self):
        tgt_input = self.config.target[:-1, :]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = getPosAndPaddingMask(self.config.sample, tgt_input)
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = src_mask.to(self.config.device), tgt_mask.to(self.config.device), src_padding_mask.to(self.config.device), tgt_padding_mask.to(self.config.device)
        logits = self.config.net(self.config.sample, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        self.config.output = logits.reshape(-1, logits.shape[-1])
        self.config.target = self.config.target[1:, :].reshape(-1)

    def processAccept(self):
        src = "凡日月所照、江河所至，皆为汉土。"
        src = src.strip()
        sample_token = torch.tensor([self.config.train_dataset.word_dict[0][w] for w in src ], dtype=torch.long)
        sample_token = torch.cat([torch.tensor([BOS_IDX]), sample_token, torch.tensor([EOS_IDX]) ])
        src = sample_token.view(-1, 1)
        num_tokens = src.shape[0]
        src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
        tgt_tokens = self.greedy_decode(src, src_mask, max_len=(num_tokens +5)*2, start_symbol=BOS_IDX).flatten()
        result = self.config.train_dataset.index2string(tgt_tokens.tolist())
        result = result.replace("BOS_IDX", "").replace("EOS_IDX", "")
        LOG.logI("accept test result: {}".format(result) )

    def greedy_decode(self, src, src_mask, max_len, start_symbol):
        src = src.to(self.config.device)
        src_mask = src_mask.to(self.config.device)
        memory = self.config.net.encode(src, src_mask)
        ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(self.config.device)
        for i in range(max_len-1):
            memory = memory.to(self.config.device)
            tgt_mask = (generate_square_subsequent_mask(ys.size(0)).type(torch.bool)).to(self.config.device)
            out = self.config.net.decode(ys, memory, tgt_mask)
            out = out.transpose(0, 1)
            prob = self.config.net.generator(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            next_word = next_word.item()
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
            if next_word == EOS_IDX:
                break
        return ys

if __name__ == "__main__":
    from config import encoder_config, transformer_config
    # train = TransformerEncoderTrain(encoder_config)
    train = TransformerTrain(transformer_config)
    train()
