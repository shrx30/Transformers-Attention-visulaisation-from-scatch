import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import re

# Tokenizer
class Tokenizer:
    def __init__(self):
        self.word_to_index = {'<PAD>': 0, '<UNK>': 1}
        self.index_to_word = {0: '<PAD>', 1: '<UNK>'}

    def tokenize(self, sentence):
        return re.findall(r'\b\w+\b', sentence.lower())

    def build_vocab(self, sentences):
        words = set()
        for sentence in sentences:
            words.update(self.tokenize(sentence))
        for index, word in enumerate(words, 2):
            self.word_to_index[word] = index
            self.index_to_word[index] = word

    def encode(self, sentence):
        return [self.word_to_index.get(word, 1) for word in self.tokenize(sentence)]

    def decode(self, indices):
        return [self.index_to_word.get(idx, '<UNK>') for idx in indices]

# Attention Model
class SimpleSelfAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        Q, K, V = self.q(x), self.k(x), self.v(x)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(Q.shape[-1])
        attn_probs = self.softmax(attn_scores)
        return torch.matmul(attn_probs, V), attn_probs

# Generate Attention Map
def generate_attention(sentence):
    tokenizer = Tokenizer()
    tokenizer.build_vocab([sentence])
    input_indices = tokenizer.encode(sentence)
    vocab_size = len(tokenizer.word_to_index)
    d_model = 16

    x = torch.randn(len(input_indices), d_model)
    model = SimpleSelfAttention(d_model)
    _, attention_map = model(x)

    return attention_map.detach().numpy(), tokenizer.decode(input_indices)

# Streamlit UI
st.title("Attention Map Visualizer")
sentence = st.text_input("Enter a sentence:", "Transformers are amazing!")
if sentence:
    attn_map, tokens = generate_attention(sentence)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(attn_map, annot=False, cmap='Blues', xticklabels=tokens, yticklabels=tokens, ax=ax)
    plt.xticks(rotation=45, fontsize=10)
    plt.yticks(fontsize=10)
    st.pyplot(fig)
