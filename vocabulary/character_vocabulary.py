import string

_MAXLEN = 32

class CharacterVocabulary:
    PAD = '<PAD>'
    UNK = '<UNK>'
    START = '<START>'
    END = '<END>'
    specials = [PAD,UNK,START,END]
    PAD_IDX = specials.index(PAD)
    UNK_IDX = specials.index(UNK)
    START_IDX = specials.index(START)
    END_IDX = specials.index(END)

    vocab = specials + list(string.printable)
    char_to_int = {c:i for i,c in enumerate(vocab)}
    int_to_char = {i:c for c,i in char_to_int.items()}

    def __getitem__(self,chr:str) -> int:
        return self.char_to_int.get(chr,self.UNK_IDX)

    @property
    def size(self):
        return len(self.char_to_int)


character_vocab = CharacterVocabulary()

def _pad_sequence(sequence, maxlen: int = _MAXLEN):
    diff = maxlen - len(sequence)
    if diff<0:
        raise Exception('length of sequence must be less than max length')
    return sequence+[character_vocab.PAD_IDX]*diff

def encode_character_sequence(sequence):
    encoded = [character_vocab.START_IDX]
    encoded += [character_vocab[c] for c in sequence]
    encoded += [character_vocab.END_IDX]
    encoded = encoded[0:_MAXLEN]
    encoded = _pad_sequence(encoded)
    return encoded
