import sys
import os
import math
import time
import numpy as np
import cv2
import torch
from torch import nn
from torch import optim
from deepvac import LOG, DeepvacTrain
from modules.utils import generate_square_subsequent_mask
from data.dataloader import FileLineEncoderDataset

class TransformerTrain(DeepvacTrain):
    def __init__(self, deepvac_config):
        super(TransformerTrain, self).__init__(deepvac_config)
        self.src_mask = generate_square_subsequent_mask(deepvac_config.seq_len).to(self.config.device)
        self.ntokens = deepvac_config.ntokens

    def debugOutput(self):
        input = self.config.sample.transpose(0,1)

        output = self.config.output.transpose(0,1)
        output = torch.argmax(output, dim=2)
        LOG.logI("input shape: {}".format(input.shape))
        LOG.logI("output shape: {}".format(output.shape))

        for t in input:
            src = FileLineEncoderDataset.index2string(t.tolist())
            LOG.logI("SRC: {}".format(src))
        
        for t in output:
            target = FileLineEncoderDataset.index2string(t.tolist())
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
        super(TransformerTrain, self).doLoss()

if __name__ == "__main__":
    from config import config
    train = TransformerTrain(config)
    train()
