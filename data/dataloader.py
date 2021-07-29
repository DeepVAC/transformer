import torch
from deepvac import LOG
from deepvac.datasets import DatasetBase

class FileLineEncoderDataset(DatasetBase):
    word_set = set()
    word_dict = {}
    word_list = []

    @staticmethod
    def build_vocab_from_file(vocab_file_path):
        with open(vocab_file_path, 'r', encoding='utf-8') as f:
            for s in f:
                s = s.strip()
                FileLineEncoderDataset.word_set.update(set(s))
        FileLineEncoderDataset.word_list = list(FileLineEncoderDataset.word_set)
        FileLineEncoderDataset.word_dict = {word: i for i, word in enumerate(FileLineEncoderDataset.word_list)}

    @staticmethod
    def get_token_num():
        size = len(FileLineEncoderDataset.word_dict)
        if size == 0:
            LOG.logE("you must build vocab via build_vocab_from_file() API first.", exit=True)
        return size
    
    @staticmethod
    def index2string(idx_list):
        s = ''
        for i in idx_list:
            s += FileLineEncoderDataset.word_list[i]
        return s

    def __init__(self, deepvac_config, file_path, seq_len=14):
        super(FileLineEncoderDataset, self).__init__(deepvac_config)
        self.file_path = file_path
        self.seq_len = seq_len
        self.sample_list = []
        self.sample_token_list = []
        if len(self.word_dict) == 0:
            LOG.logE("you must build vocab via build_vocab_from_file() API first.", exit=True)

        with open(self.file_path, 'r', encoding='utf-8') as f:
            for original_s in f:
                s = original_s.strip().replace('，','').replace('。','')
                if len(s) < self.seq_len + 1:
                    continue
                for i in range(0, len(s)- seq_len - 1):
                    sample = [s[i:i+seq_len], s[i+1: i+seq_len+1] ] 
                    self.sample_list.append(sample)

        #tokennize
        for sample in self.sample_list:
            sample_token = torch.tensor([self.word_dict[w] for w in sample[0] ], dtype=torch.long)
            target_token = torch.tensor([self.word_dict[w] for w in sample[1] ], dtype=torch.long)
            sample_token = [sample_token, target_token]
            self.sample_token_list.append(sample_token)

    def auditConfig(self):
        pass

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        return self.sample_token_list[index]

class FileLineFromTwoFilesTranslationDataset(DatasetBase):
    def __init__(self, deepvac_config, src_file_path, tgt_file_path, src_max_len=10000, tgt_max_len=10000):
        super(FileLineFromTwoFilesTranslationDataset, self).__init__(deepvac_config)
        self.src_file_path = src_file_path
        self.tgt_file_path = tgt_file_path
        self.src_list = []
        self.tgt_list = []

        with open(self.src_file_path, 'r', encoding='utf-8') as f1, open(self.tgt_file_path, 'r', encoding='utf-8') as f2:
            for s, t in zip(f1, f2):
                if len(s.strip()) + 1 >= src_max_len or len(t.strip()) + 1 >= tgt_max_len:
                    continue
                self.src_list.append(s.strip())
                self.tgt_list.append(t.strip())

#test
if "__main__" == __name__:
    from deepvac import new
    config = new()
    #data = FileLineFromTwoFilesTranslationDataset(config, "/gemfield/hostpv/github/transformer/ancient_chinese.txt", "/gemfield/hostpv/github/transformer/morden_chinese.txt")
    data = FileLineEncoderDataset(config, "/gemfield/hostpv/github/transformer/data/ancient_chinese.txt", seq_len=14)
