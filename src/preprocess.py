from .interfaces import Vocabulary, Tokenizer, ModelConfig, SpecialTokens
from collections import Counter
import re
import numpy as np

class SimpleTokenizer(Tokenizer):

    def __init__(self, config: ModelConfig):
        self.special_tokens: SpecialTokens = config.special_tokens
        self.lowercase = config.lowercase
        self.remove_punctuation = config.remove_punctuation

        self.punctuation_pattern = config.punctuation_pattern
        self.sentence_pattern = config.split_sentence_pattern
        self.token_pattern = config.split_token_pattern

    def normalize(self, sentence: str) -> str:

        if self.lowercase:
            sentence = sentence.lower()

        if self.remove_punctuation:
            sentence = re.sub(
                self.punctuation_pattern,
                ' ',
                sentence
            )

        sentence = re.sub(r'\s+', ' ', sentence).strip()

        return sentence

    def tokenize(self, paragraph: str) -> list[str]:

        result = [self.special_tokens.bop_token]

        sentences = self.sentence_pattern.split(paragraph)

        for sentence in sentences:

            sentence = self.normalize(sentence)

            if not sentence:
                continue

            tokens = self.token_pattern.findall(sentence)

            if not tokens:
                continue

            result.append(self.special_tokens.bos_token)
            result.extend(tokens)
            result.append(self.special_tokens.eos_token)

        result.append(self.special_tokens.eop_token)

        return result


class SimpleVocabulary(Vocabulary):
    """
    A Simple implementation of the Vocabulary interface for Vietnamese text.
    Function: Token -> Lookup table -> ID.

    Input:  List of sentences (tokenized)
    Output: Vocabulary (token to ID mapping, ID to token list, token count list)
    """

    def __init__(self, config: ModelConfig):
        self.special_tokens: SpecialTokens = config.special_tokens

        self.vocab_size = config.vocab_size

        self._reset()

    def _reset(self):
        self.token_to_id = {}
        self.tokens = []
        self.token_counts = []

    def build_vocabulary(self, tokenized_paragraphs: list[list[str]]) -> list[str]:
        """
        Builds the vocabulary from a list of tokenized paragraphs.
        Params: [Corpura] -> [Paragraph (tokenized)]
        """
        self._reset()
        
        for _, token in self.special_tokens.__dict__.items():
            self.add_token(token, count=0)

        for paragraph in tokenized_paragraphs:
            counter = Counter(paragraph)
            for token, count in counter.items():
                self.add_token(token, count)

        n = len(self.special_tokens.__dict__)

        # sort tokens by frequency
        sorted_pairs = sorted(
            zip(
                self.tokens[n:],
                self.token_counts[n:]
            ),
            key=lambda x: x[1],
            reverse=True
        )

        # Select top tokens
        total_count = sum(self.token_counts)
        sorted_pairs = sorted_pairs[:self.vocab_size-n]

        # Rebuild token_to_id and tokens list
        self.tokens = self.tokens[:n] + [token for token, _ in sorted_pairs]
        self.token_counts = self.token_counts[:n] + [count for _, count in sorted_pairs]
        self.token_to_id = {
            token: idx
            for idx, token in enumerate(self.tokens)
        }

        # Add the discard counting back as <UNK>
        discard_count = total_count - sum(self.token_counts)
        self.add_token(self.special_tokens.unk_token, discard_count)

    def token_distribution(self):
        count = np.array(self.token_counts)
        n = len(self.special_tokens.__dict__)
        count[0:n] = 0

        return count / count.sum()

    def add_token(self, token: str, count: int = 1):
        if token in self.token_to_id:
            id = self.token_to_id[token]
            self.token_counts[id] += count
        else:
            id = len(self.tokens)
            self.token_to_id[token] = id
            self.tokens.append(token)
            self.token_counts.append(count)

    def to_id(self, token: str) -> int:
        """Returns the ID of a token."""
        return self.token_to_id.get(token, self.token_to_id[self.special_tokens.unk_token]) 
    
    def to_token(self, id: int) -> str:
        """Returns the token corresponding to an ID."""
        if 0 <= id < len(self.tokens):
            return self.tokens[id]
        else:
            return self.special_tokens.unk_token
        
    def __len__(self) -> int:
        return len(self.tokens)
    
    @property
    def pad_idx(self):
        return self.to_id(self.special_tokens.pad_token)