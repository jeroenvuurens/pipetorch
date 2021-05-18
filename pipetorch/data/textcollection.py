from torchtext.data.utils import get_tokenizer, ngrams_iterator
from collections import Counter
from torchtext.vocab import Vocab, build_vocab_from_iterator
from torch.utils.data import DataLoader, IterableDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import random_split
from .databunch import Databunch
import torch
import copy

class TextCollection:
    def __init__(self, train_iter, valid_iter, test_iter=None, language='basic_english', min_freq=1, vocab=None, 
                 labels = None, batch_size = 32, shuffle=True, specials=('<unk>', '<pad>')):
        self._train_iter = train_iter
        self._valid_iter = valid_iter
        self._test_iter = test_iter
        self.language = language
        self.min_freq = min_freq
        self.vocab = vocab
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.specials = specials
        self.__collate = None
    
    @classmethod
    def from_iter(cls, train_iter, valid_iter, test_iter=None, language='basic_english', min_freq=1, vocab=None, 
                 labels = None, batch_size = 32, shuffle=True, specials=('<unk>', '<pad>'), collate=None):
        r = cls(train_iter, valid_iter, test_iter=test_iter, language=language, min_freq=min_freq, vocab=vocab, 
                 labels = labels, batch_size = batch_size, shuffle=shuffle, specials=specials)
        if collate is not None:
            r = r.collate(collate)
        return r
    
    @property
    def _collate(self):
        return self.__collate
    
    def collate(self, collate):
        if collate == 'offset':
            return OffsetTextCollection(self._train_iter, self._valid_iter, test_iter=self._test_iter, 
                    language=self.language, min_freq=self.min_freq, vocab=self.vocab, labels = self.labels, 
                    batch_size = self.batch_size, shuffle=self.shuffle, specials=self.specials)
        if collate == 'pad':
            return PaddedTextCollection(self._train_iter, self._valid_iter, test_iter=self._test_iter,  
                    language=self.language, min_freq=self.min_freq, vocab=self.vocab, labels = self.labels, 
                    batch_size = self.batch_size, shuffle=self.shuffle, specials=self.specials)
        r = TextCollection(self._train_iter, self._valid_iter, test_iter=self._test_iter,  
                    language=self.language, min_freq=self.min_freq, vocab=self.vocab, labels = self.labels, 
                    batch_size = self.batch_size, shuffle=self.shuffle, specials=self.specials)
        r.__collate = collate
        return r   
    
    def split(self, valid_perc, test_perc=None):
        r = copy.copy(self)
        test_count = 0 if test_perc is None else round(test_perc * len(r._train_iter))
        valid_count = round(valid_perc * len(r._train_iter))
        train_count = len(r._train_iter) - test_count - valid_count
        print(len(r._train_iter), train_count, valid_count, test_count)
        train_iter, valid_iter, test_iter = random_split(list(r._train_iter), [train_count, valid_count, test_count])
        r._train_iter = train_iter
        r._valid_iter = valid_iter
        if test_perc is not None:
            r._test_iter = test_iter
        return r
        
    @property
    def language(self):
        return self._language
        
    @language.setter
    def language(self, value):
        if value is not None:
            self._language = value
        
    @property
    def min_freq(self):
        return self._min_freq
        
    @min_freq.setter
    def min_freq(self, value):
        if value is not None:
            self._min_freq = value

    @property
    def batch_size(self):
        return self._batch_size
        
    @batch_size.setter
    def batch_size(self, value):
        if value is not None:
            self._batch_size = value

    @property
    def shuffle(self):
        return self._shuffle
        
    @shuffle.setter
    def shuffle(self, value):
        if value is not None:
            self._shuffle = value

    @property
    def train(self):
        try:
            return self._train
        except:
            self._train = TextDataSet(self._train_iter)
            return self._train
        
    @property
    def valid(self):
        try:
            return self._valid
        except:
            self._valid = TextDataSet(self._valid_iter)
            return self._valid
        
    @property
    def test(self):
        try:
            return self._test
        except:
            self._test = TextDataSet(self._test_iter)
            return self._test
    
    @property
    def tokenizer(self):
        try:
            return self._tokenizer
        except:
            self._tokenizer = get_tokenizer(self.language)
            return self._tokenizer
        
    @property
    def vocab(self):
        try:
            return self._vocab
        except:
            self._build_vocab()
            return self._vocab
    
    @property
    def labels(self):
        try:
            return self._labels
        except:
            self._build_vocab()
            return self._labels

    @labels.setter
    def labels(self, value):
        if value is not None:
            self._labels = value
            
    @vocab.setter
    def vocab(self, value):
        if value is not None:
            self._vocab = value
    
    def decode_sentence_index(self, words, index):
        return self.decode_sentence(words[index])
    
    def decode_sentence(self, words):
        return ' '.join([ self.vocab.itos[w] for w in words])

    def encode_sentence(self, sentence):
        return self.vocab.lookup_indices(self.tokenizer(sentence))

    def _build_vocab(self):
        counter_line = Counter()
        labels = Counter()
        for (label, line) in self.train:
            counter_line.update(self.tokenizer(line))
            labels[label] += 1
        self._vocab = Vocab(counter_line, specials=self.specials, min_freq=self.min_freq)
        self._labels = LabelSet(labels)
        
    def to_databunch(self, batch_size=None, shuffle=True, vocab=None, labels=None, min_freq=None, language=None, offset=None, balance=False, collate=None, **kwargs):
        r = copy.copy(self)
        r.batch_size = batch_size
        r.shuffle = shuffle
        r.vocab = vocab
        r.labels = labels
        r.min_freq = min_freq
        r.language = language
        r.offset = offset
        if collate is not None:
            r = r.collate(collate)
        return Databunch(None, r.train, r.valid, r.test, batch_size=r.batch_size, valid_batch_size=r.batch_size, shuffle=r.shuffle, collate=r._collate, **kwargs)
    
class PaddedTextCollection(TextCollection):
    def _collate(self, batch):
        label_list, text_list = [], []
        for (_label, _text) in batch:
            label_list.append(self.labels[_label])
            processed_text = torch.tensor(self.encode_sentence(_text), dtype=torch.int64)
            text_list.append(processed_text)
        label_list = torch.tensor(label_list, dtype=torch.int64)
        text_list = pad_sequence(text_list, batch_first=True, padding_value=self.vocab.stoi['<pad>'])
        return text_list, label_list

    def decode_sentence_index(self, words, index):
        return super().decode_sentence(words[index].squeeze())
        
    def decode_sentences(self, words):
        return [ self.decode_sentence(words, i) for i in range(len(words))]
    
class OffsetTextCollection(TextCollection):
    def _collate(self, batch):
        label_list, text_list, offsets = [], [], [0]
        for (_label, _text) in batch:
            label_list.append(self.labels[_label])
            processed_text = torch.tensor(self.encode_sentence(_text), dtype=torch.int64)
            text_list.append(processed_text)
            offsets.append(processed_text.size(0))
        label_list = torch.tensor(label_list, dtype=torch.int64)
        offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
        text_list = torch.cat(text_list)
        return text_list, offsets, label_list

    def decode_sentence_index(self, words, offsets, index):
        start = offsets[index]
        end = offsets[index+1] if index < len(offsets)-1 else len(words)
        return super().decode_sentence(words[start:end])
        
    def decode_sentences(self, words, offsets):
        return [ self.decode_sentence(words, offsets, i) for i in range(len(offsets))]
    
class LabelSet:
    def __init__(self, labels):
        self.labels = labels
        self._itol = list(labels.keys())
        self._ltoi = {l:i for i, l in enumerate(self._itol)}
        
    def __getitem__(self, label):
        return self._ltoi[label]
    
    def __len__(self):
        return len(self._itol)
    
    def lookup_labels(self, labels):
        return [self._ltoi[l] for l in labels]
    
    def lookup_ints(self, ints):
        return [self._itol[i] for i in ints]

class TextDataSet:
    def __init__(self, it):
        self.data = list(it)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data.__getitem__(index)