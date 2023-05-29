#from torchtext.data.utils import get_tokenizer
from collections import Counter
#from torchtext.vocab import Vocab, build_vocab_from_iterator, FastText, GloVe
from torch.utils.data import DataLoader, IterableDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import random_split
from .databunch import Databunch
from .kagglereader import Kaggle
from .helper import path_shared, path_user, read_torchtext
import numpy as np
import torch
import copy

class TextCollection:
    """
    Setup an in memory text collection
    
    Args:
        train: TextDataSet
        valid: TextDataSet (None)
        test: TextDataSet (None)
        language: str ('basic_english')
            the langage used by the tokenizer
        min_freq: int (1)
            the minimum document frequency for words to be added to the vocabulary
        vocab: (None)
            a vocabulary to use. If None, a new vocabulary will be made from the trainset
        labels: LabelSet (None)
            a set of labes to use. If None, a LabelSet is constructed from the train set
        specials: tuple of str ('<unk>', 'pad')
            special tokens that are added to the vocabulary. Most commonly, <unk> is used
            for words that are not in the vocabulary and <pad> for making data points
            the same size.
        collate: callable (None)
            a function to prepare a batch, for example by padding all sentences to equal
            length.
    """
    
    def __init__(self, train, test=None, valid=None, language='basic_english', min_freq=1, vocab=None, 
                 labels = None, specials=('<unk>', '<pad>'), collate=None):
        self._vocab = vocab
        self.test = test
        self.train = train
        self.valid = valid
        self.language = language
        self.min_freq = min_freq
        self.labels = labels
        self.specials = specials
        self.__collate = collate
    
    @classmethod
    def from_iter(cls, train_iter, test_iter=None, valid_iter=None, **kwargs):
        """
        Reads a TextCollection from iterators. The most common use is from TorchText
        DataSets.
        
        Arguments:
            train_iter: str
                iterator that is used as the train set, e.g. 'train.csv'
            test_iter: str (None)
                iterator that is used as the test set, e.g. 'test.csv'
            valid_iter: str (None)
                iterator that is used as the valid set, e.g. 'valid.csv'
            **kwargs: see the TextCollection constructor for additional arguments
                      such as language, min_freq, vocab, labels, special, collate 
        Returns: TextCollection
        """
        train = TextDataSet.from_iter(train_iter)
        test = None if test_iter is None else TextDataSet.from_iter(test_iter)
        valid = None if valid_iter is None else TextDataSet.from_iter(valid_iter)
        return cls(train, test=test, valid=valid, **kwargs)
    
    @classmethod
    def from_csv(cls, train_filename, test_filename=None, valid_filename=None, **kwargs):
        """
        Reads a TextCollection from csv files. 
        
        Typically, the files are assumed to be organized in lines with the format: label, text
        Although often these files have a .csv extension, the files are not true csv files.
        
        Example:
            read_from_csv('train.csv', test_filename='test.csv', min_freq=1)
            to combine a train and test set in a single TextCollection

        Arguments:
            train_filename: str
                the filename that is used as the train set, e.g. 'train.csv'
            test_filename: str (None)
                the filename that is used as the test set, e.g. 'test.csv'
            valid_filename: str (None)
                the filename that is used as the valid set, e.g. 'valid.csv'
            **kwargs: see the TextCollection constructor for additional arguments
                      such as language, min_freq, vocab, labels, special, collate 
        Returns: TextCollection
        """
        train = TextDataSet.from_csv(train_filename)
        test = None if test_filename is None else TextDataSet.from_csv(test_filename)
        valid = None if valid_filename is None else TextDataSet.from_csv(valid_filename)
        return cls(train, test=test, valid=valid, **kwargs)

    @classmethod
    def from_kaggle(cls, dataset, train=None, test=None, valid=None, shared=True, force=False, **kwargs):
        """
        Reads a TextCollection from a Kaggle dataset. The files are forwarded to from_csv().
        
        The downloaded dataset is automatically stored so that the next time
        it is read from file rather than downloaded. 
        The dataset is stored by default in a folder
        with the dataset name in `~/.pipetorchuser`. 

        If the dataset is not cached, this functions requires a valid .kaggle/kaggle.json file, that you can 
        create manually or with the function `create_kaggle_authentication()`.

        Note: there is a difference between a Kaggle dataset and a Kaggle competition. For the latter, 
        you have to use `read_from_kaggle_competition`.

        Example:
            read_from_kaggle('yufengdev/bbc-text-categorization')
                to read/download `https://www.kaggle.com/datasets/yufengdev/bbc-text-categorization`

        Arguments:
            dataset: str
                the username/dataset part of the kaggle url, e.g. uciml/autompg-dataset for 
            train: str (None)
                the filename that is used as the train set, e.g. 'train.csv'
            test: str (None)
                the filename that is used as the test set, e.g. 'test.csv'
            valid: str (None)
                the filename that is used as the valid set, e.g. 'valid.csv'
            shared: bool (False)
                save the dataset in ~/.pipetorch instead of ~/.pipetorchuser, allowing to share downloaded
                files between users.
            force: bool (False)
                when True, the dataset is always downloaded
            **kwargs: see the TextCollection constructor for additional arguments
                      such as language, min_freq, vocab, labels, special, collate 
        Returns: TextCollection
        """
        k = Kaggle(dataset, shared=shared)
        if force:
            k.remove_user()
        train = k.file(train)
        if test is not None:
            test = k.file(test)
        if valid is not None:
            valid = k.file(valid)
        return cls.from_csv(train, test_filename=test, valid_filename=valid, **kwargs)

    @classmethod
    def from_torchtext(cls, func, min_freq=1, collate='pad', **kwargs): 
        """
        Reads a TextCollection from a torchtext data function.
        
        Arguments:
            func: callable
                from torchtext.datasets, a a function that behaves similarly and returns
                a train and test iterator for the dataset. By default the datasets are
                cached in the .pipetorchuser folder or used from the .pipetorch folder
                if present.
            **kwargs: see the TextCollection constructor for additional arguments
                      such as language, min_freq, vocab, labels, special, collate 
        Returns: TextCollection
        """
        try:
            train_iter, test_iter = read_torchtext( func )
        except:
            raise ValueError(f'cannot convert given function {func} to an iterator. Is it a torchtext data function?')
        return cls.from_iter(train_iter=train_iter, test_iter=test_iter, min_freq=min_freq, collate=collate, **kwargs)
            
    @property
    def _collate(self):
        return self.__collate
    
    def collate(self, collate):
        """
        Return a textcollection for which the dataset is prepared as:
        'pad': the sentences ae padded with a '<pad>' token to have equal length
        'offset': a batch is a list of tokenids and a seperate list of offsets indicate where a new sentence starts.
        ...: a cusrom collation function
        
        There are several options in TorchText to train with either padded of offset datasets.
        """
        if collate == 'offset':
            return OffsetTextCollection(self.train, test=self.test, valid=self.valid, 
                    language=self.language, min_freq=self.min_freq, vocab=self._vocab, labels = self.labels, 
                    specials=self.specials)
        if collate == 'pad':
            return PaddedTextCollection(self.train, test=self.test, valid=self.valid,  
                    language=self.language, min_freq=self.min_freq, vocab=self._vocab, labels = self.labels, 
                    specials=self.specials)
        r = TextCollection(self.train, test=self.test, valid=self.valid, 
                    language=self.language, min_freq=self.min_freq, vocab=self._vocab, labels = self.labels, 
                    specials=self.specials)
        r.__collate = collate
        return r   
    
    def split(self, *perc, test_perc=None, valid_perc=None):
        """
        Creates out-of-sample test and valid sets.
        
        Args:
            perc, test_perc, valid_perc: float[0-1]
                fraction of the data that is used resp. for the test and valid set.
                When a fixed test set is used, the first positional argument is used as
                valid_perc, otherwise as test_perc.

        Returns: TextCollection
            a shallow copy of this TextCollection that is configured to 
            split the data in a train, test and/or valid set

        Note: you cannot resplit a textcollection that was already split, or if it has a fixed validation or test set.
        """
        assert test_perc is None or self.test is None, 'You cannot specify a test_perc if a fixed test set is given'
        assert self.valid is None, 'You cannot resplit a text collection that was already split or that has a fixed validation set'
        assert self.vocab_not_exists, 'You cannot split a text collection when a vocabulary is already built'
        assert len(perc) < 2 or test_perc is None or valid_perc is None, 'Too many percentages'
        assert len(perc) < 3, 'Too many percentages'
        if len(perc) > 0:
            if test_perc is None:
                if self.test:
                    valid_perc = perc[0]
                else:
                    test_perc = perc[0]
            else:
                valid_perc = perc[0]
            if len(perc) == 2:
                test_perc, valid_perc = 1
                
        r = copy.copy(self)
        test_count = 0 if test_perc is None else round(test_perc * len(r.train))
        valid_count = 0 if valid_perc is None else round(valid_perc * len(r.train))
        train_count = len(r.train) - test_count - valid_count
        #print(len(r.train), train_count, valid_count, test_count)
        #print(len(r.train), sum([train_count, valid_count, test_count]))
        r.train, r.test, r.valid = random_split(r.train, [train_count, test_count, valid_count])
        return r
        
    @property
    def language(self):
        return self._language
        
    @language.setter
    def language(self, value):
        assert self.vocab_not_exists, 'You cannot specify a language after the vocabulary was built'
        if value is not None:
            self._language = value
            try:
                del self._tokenizer
                del self._vocab
                del self._labels
            except: pass
        
    @property
    def min_freq(self):
        return self._min_freq
        
    @min_freq.setter
    def min_freq(self, value):
        assert self.vocab_not_exists, 'You cannot specify a min_freq after the vocabulary was built'
        if value is not None:
            self._min_freq = value
    
    @property
    def tokenizer(self):
        """
        The tokenizer used to transform a text into a list of words/tokens.
        """
        try:
            return self._tokenizer
        except:
            from torchtext.data.utils import get_tokenizer
            self._tokenizer = get_tokenizer(self.language)
            return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value):
        assert self.vocab_not_exists, 'You cannot change the tokenizer after the vocabulary was built'
        self._tokenizer = value
        
    @property
    def vocab_not_exists(self):
        return self._vocab is None
        
    @property
    def vocab(self):
        """
        A (generated) PyTorch Vocab that maps all tokens in the training set to numbers.
        """
        if self._vocab is None:
            self._build_vocab()
        return self._vocab
    
    @vocab.setter
    def vocab(self, value):
        self._vocab = value
    
    @property
    def labels(self):
        """
        Collection of groud truth labels for the TextCollection
        """
        if self._labels is None:
            self._build_vocab()
        return self._labels

    @labels.setter
    def labels(self, value):
        self._labels = value
            
    def decode_sentence_index(self, words, index):
        """
        returns a list of tokens for the text at index in words.
        """
        return self.decode_sentence(words[index])
    
    def decode_sentence(self, tokenids):
        # lookup_tokens on vocab transforms List(int) -> List(str)
        return self.vocab.lookup_tokens(list(tokenids))

    def encode_sentence(self, sentence):
        # lookup_indices on vocab transforms List(str) -> List(int)
        return self.vocab.lookup_indices(self.tokenizer(sentence))

    def token_iterator(self, dataset):
        """
        returns: iterator over lists of tokenids that represent the original sentences
        may be overriden by subclassing TextCollection
        """
        def yield_tokens(dataset):
            for _, text in dataset:
                yield self.tokenizer(text)
        return yield_tokens(dataset)
    
    def _build_vocab(self):
        from torchtext.vocab import build_vocab_from_iterator

        labels = Counter([ l for l, _ in self.train ])
        self._vocab = build_vocab_from_iterator(self.token_iterator(self.train), specials=self.specials, special_first=True, min_freq=self.min_freq)
        self._vocab.set_default_index(self._vocab['<unk>'])
        self._labels = LabelSet(labels)
        
    def to_databunch(self, batch_size=32, shuffle=True, balance=False, **kwargs):
        """
        Return: a databunch (object with a dataloader for the train, valid, (test) set)
        batch_size (None): the batch_size used by the DataLoader
        shuffle (True): whether the samples are shuffled between epochs
        balance (False): if the training set is to be balanced (only works for binary classification)
        **kwargs: any other named argument that Databunch accepts
        
        Note: creating a databunch does not modify the original TextCollection, instead a shallow copy is made
        and creating the databunch usually triggers split, balance and creation of the vocabulary to be triggered.
        You can access the shallow copy of the textcollection through db.textcollection and the generated 
        vocabulary through db.vocab.
        """
        r = copy.copy(self)
        db = Databunch(None, r.train, r.test, r.valid, batch_size=batch_size, test_batch_size=batch_size, shuffle=shuffle, balance=balance, collate=r._collate, **kwargs)
        db.textcollection = r
        db.vocab = r.vocab
        if r._using_pretrained():
            db.pretrained_embeddings = r.pretrained_embeddings
            db.load_pretrained_embeddings_in_layer = r.load_pretrained_embeddings_in_layer
        return db
    
    def GloVe(self, name='6B', cache='/data/datasets', max_vectors=None):
        """
        Setup the use of pretrained GloVe embeddings. There are 4 sets available for download:
        name (6B): 6B, 42B, 840B or twitter.27B
        cache: shared folder to store downloaded embeddings. Do not change unless you know what you are doing
        since embeddings take up a lot of storage.
        max_vectors: use only the indicated number of vectors to save RAM. Since the tokens are sorted on frequency,
        using only the n most frequently appearing tokens works well in most cases.
        """
        from torchtext.vocab import GloVe
        r = copy.copy(self)
        r._pretrained = GloVe
        r._pretrained_params = {'name':name, 'cache':cache, 'max_vectors':max_vectors}
        return r
    
    def _using_pretrained(self):
        try:
            self._pretrained
            return True
        except:
            return False
    
    def pretrained_embeddings(self, dim):
        assert self._using_pretrained(), 'Can only prepare a table of pretrained embeddings if you configure using GloVe'
        self._pretrained_params['dim'] = dim
        self._embeddings = self._pretrained(**self._pretrained_params).get_vecs_by_tokens(self.vocab.get_itos())
        return self._embeddings
    
    def load_pretrained_embeddings_in_layer(self, layer):
        dim = embedding_layer.embedding_dim
        embedding_layer.weight.data = self.pretrained_embeddings(dim)              
        
class PaddedTextCollection(TextCollection):
    def _collate(self, batch):
        label_list, text_list = [], []
        for (_label, _text) in batch:
            label_list.append(self.labels[_label])
            processed_text = torch.tensor(self.encode_sentence(_text), dtype=torch.int64)
            text_list.append(processed_text)
        label_list = torch.tensor(label_list, dtype=torch.int64)
        text_list = pad_sequence(text_list, batch_first=True, padding_value=self.vocab['<pad>'])
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
    @classmethod
    def from_iter(cls, it):
        r = cls()
        r.data = list(it)
        return r
        
    @classmethod
    def from_csv(cls, filename, header=True, delimiter=','):
        r = cls()
        with open(filename) as fin:
            r.data = [ l.split(delimiter, 1) for l in fin ]
        if header:
            r.data = r.data[1:]
        return r
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data.__getitem__(index)
