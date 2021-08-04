from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence
from deepvac import LOG
from deepvac.datasets import DatasetBase

UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
KEY_TOKENS = {'UNK_IDX':0, 'PAD_IDX':1, 'BOS_IDX':2, 'EOS_IDX':3}

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(src_sample)
        tgt_batch.append(tgt_sample)
    src_batch = pad_sequence(src_batch, padding_value=KEY_TOKENS['PAD_IDX'])
    tgt_batch = pad_sequence(tgt_batch, padding_value=KEY_TOKENS['PAD_IDX'])
    return src_batch, tgt_batch

class FileLineTranslationDataset(DatasetBase):
    def __init__(self, deepvac_config, file_path, vocab_file_path, max_len_list=[14, 20]):
        super(FileLineTranslationDataset, self).__init__(deepvac_config)
        self.word_dict = [{},{}]
        self.word_list = [ [], [] ]
        self.build_vocab_from_file(vocab_file_path)

        if not isinstance(file_path, list):
            file_path = [file_path]
            
        self.file_path = file_path
        self.max_len_list = max_len_list
        self.sample_list = []
        self.sample_token_list = []

        if len(self.file_path) == 1:
            self.buildDatasetFromOneFile()
        elif len(self.file_path) == 2:
            self.buildDatasetFromTwoFiles()
        else:
            LOG.logE("illegal file_path value: {}".format(file_path), exit=True)

    def build_vocab_from_file(self, vocab_file_path):
        if not isinstance(vocab_file_path, list):
            vocab_file_path = [vocab_file_path]
        
        if len(vocab_file_path) > len(self.word_dict):
            LOG.logE("assert filepath num {} <= max {}".format(len(vocab_file_path), len(self.word_dict) ), exit=True)

        for i,f in enumerate(vocab_file_path):
            char_counter = Counter()
            with open(f, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    for s in line:
                        char_counter[s] += 1
            char_list = char_counter.most_common()
            offset = len(KEY_TOKENS)
            self.word_dict[i] = {word[0]: i+offset for i, word in enumerate(char_list)}
            self.word_dict[i].update(KEY_TOKENS)
            self.word_list[i] = [None] * len(self.word_dict[i])
            for k,v in self.word_dict[i].items():
                self.word_list[i][v] = k

    def get_token_num(self, idx=0):
        size = len(self.word_dict[idx])
        if size == 0:
            LOG.logE("you must build vocab via build_vocab_from_file() API first.", exit=True)
        return size
    
    def index2string(self, idx_list, idx=0):
        s = ''
        for i in idx_list:
            s += self.word_list[idx][i]
        return s

    def buildDatasetFromOneFile(self):
        if len(self.word_dict[0]) == 0:
            LOG.logE("you must build vocab via build_vocab_from_file() API first.", exit=True)

        with open(self.file_path[0], 'r', encoding='utf-8') as f:
            for original_s in f:
                s = original_s.strip().replace('，','').replace('。','')
                if len(s) < self.max_len_list[0] + 1:
                    continue
                for i in range(0, len(s)- self.max_len_list[0] - 1):
                    sample = [s[i:i+self.max_len_list[0]], s[i+1: i+self.max_len_list[0]+1] ] 
                    self.sample_list.append(sample)

        #tokennize
        for sample in self.sample_list:
            sample_token = torch.tensor([self.word_dict[0][w] for w in sample[0] ], dtype=torch.long)
            target_token = torch.tensor([self.word_dict[0][w] for w in sample[1] ], dtype=torch.long)
            self.sample_token_list.append([sample_token, target_token])

    def buildDatasetFromTwoFiles(self):
        if len(self.word_dict[0]) == 0 or len(self.word_dict[1]) == 0:
            LOG.logE("you must build vocab via build_vocab_from_file() API first.", exit=True)

        with open(self.file_path[0], 'r', encoding='utf-8') as f1, open(self.file_path[1], 'r', encoding='utf-8') as f2:
            for original_s, original_t in zip(f1, f2):
                s = original_s.strip()
                s = s.replace('，','').replace('。','')
                t = original_t.strip()
                if len(s) > self.max_len_list[0] or len(t) > self.max_len_list[1]:
                    continue

                sample = [s, t]
                print("gemfield: {} ----> {}".format(s,t))
                self.sample_list.append(sample)

        #tokennize
        for sample in self.sample_list:
            sample_token = torch.tensor([self.word_dict[0][w] for w in sample[0] ], dtype=torch.long)
            sample_token = torch.cat([torch.tensor([BOS_IDX]), sample_token, torch.tensor([EOS_IDX]) ])
            target_token = torch.tensor([self.word_dict[1][w] for w in sample[1] ], dtype=torch.long)
            target_token = torch.cat([torch.tensor([BOS_IDX]), target_token, torch.tensor([EOS_IDX])])
            self.sample_token_list.append([sample_token, target_token])

    def auditConfig(self):
        pass

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        return self.sample_token_list[index]

#test
if "__main__" == __name__:
    from deepvac import new
    config = new()
    token_files = ["/gemfield/hostpv/github/transformer/data/ancient_chinese.txt", "/gemfield/hostpv/github/transformer/data/morden_chinese.txt"]
    data = FileLineTranslationDataset(config, ["/gemfield/hostpv/github/transformer/data/ancient_chinese.txt", "/gemfield/hostpv/github/transformer/data/morden_chinese.txt"],token_files,max_len_list=[50,100])
    # data = FileLineTranslationDataset(config, "/gemfield/hostpv/github/transformer/data/ancient_chinese.txt", max_len_list=[14,0])
