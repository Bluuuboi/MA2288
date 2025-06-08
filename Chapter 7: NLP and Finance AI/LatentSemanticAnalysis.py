import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
import os
import certifi
os.environ['SSL_CERT_FILE'] = certifi.where()

dataset = fetch_20newsgroups(shuffle=True, random_state=42)
documents = dataset.data
print(len(documents))
print(dataset.target_names)

#Data Preprocessing

