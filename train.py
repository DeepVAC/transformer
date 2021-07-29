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

class TransformerTrain(DeepvacTrain):
    def __init__(self, deepvac_config):
        super(TransformerTrain, self).__init__(deepvac_config)
        self.src_mask = generate_square_subsequent_mask(deepvac_config.seq_len).to(self.config.device)
        self.ntokens = deepvac_config.ntokens

    def doForward(self):
        self.config.sample = self.config.sample.transpose(0,1)
        self.config.target = self.config.target.reshape(-1)
        output = self.config.net(self.config.sample, self.src_mask)
        self.config.output = output.view(-1, self.ntokens)

if __name__ == "__main__":
    from config import config
    train = TransformerTrain(config)
    train()
