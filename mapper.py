#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 12:28:17 2024

@author: pedrolopez
"""

# We'll calculate the embeddings of some topics:
topics = [
   # male studies,
   "single dose", "prostate cancer", "male subjects", "muscle damage", 
   "treatment group",
   # female studies,
   "postmenopausal women", "breast cancer", "african americans", 
   "laser ablation", "tumor cells"
]
n = len(topics) # 7

from sentence_transformers import SentenceTransformer

# Retrieve the model and check its embedding dimension.
model = SentenceTransformer('all-MiniLM-L12-v2')
d = model.get_sentence_embedding_dimension()
assert d == 384

import numpy as np
from scipy.spatial.distance import cosine

# Calculate embeddings for all topics in parallel
embed = np.array(model.encode(topics))
assert embed.shape == (n, d) # 7 * 384

# Pairwise cosine similarity among all rows
similar = np.array([[
   1.0-cosine(embed[i], embed[j]) for j in range(n)]
   for i in range(n)])
assert similar.shape == (n, n) # 7 * 7, symmetric