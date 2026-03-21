# Suppress warnings
import os, pathlib
import string
from ai_surgery.data_paths import get_data_root

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

DATA_ROOT = get_data_root()

class Vectorizer:
    def standardize(self, text):
        text = text.lower()
        return "".join(char for char in text
                       if char not in string.punctuation)

    def tokenize(self, text):
        text = self.standardize(text)
        return text.split()

    def make_vocabulary(self, dataset):
        self.vocabulary = {"": 0, "[UNK]": 1}
        for text in dataset:
            text = self.standardize(text)
            tokens = self.tokenize(text)
            for token in tokens:
                if token not in self.vocabulary:
                    self.vocabulary[token] = len(self.vocabulary)
        self.inverse_vocabulary = dict (
                (v, k) for k, v in self.vocabulary.items())

    def encode(self, text):
        text = self.standardize(text)
        tokens = self.tokenize(text)
        return [self.vocabulary.get(token, 1) for token in tokens]

    def decode(self, int_sequence):
        return " ".join(
                self.inverse_vocabulary.get(i, "[UNK]") for i in int_sequence)

vectorizer = Vectorizer()
dataset = [
        "I write, erase, rewrite",
        "Erase again, and then",
        "A poppy blooms."
]
print("dataset: ", dataset)
vectorizer.make_vocabulary(dataset)

test_sentence = "I write, rewrite, and still rewrite again"
print("test_sentence: ", test_sentence)

encoded_sentence = vectorizer.encode(test_sentence)
print("encoded_sentence: ", encoded_sentence)

decoded_sentence = vectorizer.decode(encoded_sentence)
print("decoded_sentence: ", decoded_sentence)

#
# In practice you'll work with the Keras TextVectorization layer.
print("In practice you'll work with the Keras TextVectorization layer.")
from tensorflow.keras.layers import TextVectorization
text_vectorization = TextVectorization(
    output_mode="int"
)
text_vectorization.adapt(dataset)

#
# Listing 11.1 Displaying the vocabulary
print("Listing 11.1 Displaying the vocabulary")

print("text_vectorization.get_vocabulary():")
print(text_vectorization.get_vocabulary())

#
# For demonstration, encode and decode an example sentence.
print("For demonstration, encode and decode an example sentence.")
vocabulary = text_vectorization.get_vocabulary()
encoded_sentence = text_vectorization(test_sentence)
print("encoded_sentence: ", encoded_sentence)
inverse_vocab = dict(enumerate(vocabulary))
deocded_sentence = " ".join(inverse_vocab[int(i)] for i in encoded_sentence)
print("deocded_sentence: ", deocded_sentence)














