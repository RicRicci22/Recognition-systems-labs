import torch
import pandas as pd
import json
from transformers import AutoTokenizer

def create_vocabulary(data_path:str, min_freq:int=10):
    data = pd.read_csv(data_path)
    abstracts = data["ABSTRACT"].tolist()
    abstracts = [abstract.strip().lower().replace("\n", " ") for abstract in abstracts]
    # Tokenize the text in words
    all_abs_tokenized = [abstract.split() for abstract in abstracts]
    # Flatten the list
    all_abs_flattented = [word for text in all_abs_tokenized for word in text]
    # Get unique words
    unique_words = set(all_abs_flattented)
    # Go back and filter
    appearence = {word:0 for word in unique_words}
    for title in all_abs_tokenized:
        for word in title:
            appearence[word] += 1
    
    # Filter the words
    filtered_words = [word for word in unique_words if appearence[word]>=min_freq]
    
    i2v = {i+1:word for i, word in enumerate(filtered_words)}
    v2i = {word:i for i, word in i2v.items()}
    print(f"Vocabulary size: {len(i2v)}")
    # Save the vocabulary
    with open("i2w.json", "w") as f:
        json.dump(i2v, f)
    with open("w2i.json", "w") as f:
        json.dump(v2i, f)
        
def convert_texts_to_indices(texts:list, word2idx:dict, pad_idx:int=0):
    # Given a list of titles, convert them to indices using the vocabulary. Pad the sequences to the same length
    
    # Input:
    # tokenized_titles: List of titles
    # pad_idx: Index to pad the sequences
    # word2idx: Dictionary mapping words to indices
    tokenized_texts = [text.split() for text in texts]
    max_len = max([len(tokenized_text) for tokenized_text in tokenized_texts])
    batch_titles = torch.zeros((len(tokenized_texts), max_len), dtype=torch.long) + pad_idx # Initialize with padding
    for i, title in enumerate(tokenized_texts):
        for j, word in enumerate(title):
            try:
                batch_titles[i, j] = word2idx[word]
            except:
                pass # We leave the padding token
        
    return batch_titles
    
def convert_texts_to_indices_bert(texts:list, max_len:int=512):
    # Given a list of titles, convert them to indices using the vocabulary. Pad the sequences to the same length
    
    # Input:
    # tokenized_titles: List of titles
    # pad_idx: Index to pad the sequences
    # word2idx: Dictionary mapping words to indices
    tokenizer = AutoTokenizer.from_pretrained("distilbert/distilbert-base-uncased")
    batch_titles = tokenizer(texts, padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
        
    return batch_titles


if __name__=="__main__":
    create_vocabulary("train.csv")
    v2i = json.load(open("w2i.json"))
    batch = convert_texts_to_indices_bert(["mathematics world there", "good morning"], 512)
    print(batch)
    