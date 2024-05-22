import torch
import pandas as pd
import json
import string
import nltk

def create_vocabulary(data_path:str, min_freq:int=1):
    data = pd.read_csv(data_path)
    abstracts = data["TITLE"].tolist()
    # CLEAN AND PREPROCESS THE TITLES
    # 1. Lowercase the text
    abstracts = [abstract.strip().lower() for abstract in abstracts]
    # 2. Remove punctuation
    PUNC_TO_REMOVE = string.punctuation
    abstracts = [abstract.translate(str.maketrans('', '', PUNC_TO_REMOVE)) for abstract in abstracts]
    # 3. Lemmatize the text
    lemma = nltk.wordnet.WordNetLemmatizer()
    abstracts = [lemma.lemmatize(abstract) for abstract in abstracts]
    
    # Tokenize the text in words
    all_abs_tokenized = [abstract.split() for abstract in abstracts]
    # Flatten the list
    all_abs_flattented = [word for text in all_abs_tokenized for word in text]
    # Get unique words
    unique_words = set(all_abs_flattented)
    # Create a dictionary with key the word and value the number of appearences
    appearence = {word:0 for word in unique_words}
    for title in all_abs_tokenized:
        for word in title:
            appearence[word] += 1
    
    # Filter the words based on appearence
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
    # Apply the same preprocessing steps
    # 1. Lowercase the text
    texts = [text.strip().lower() for text in texts]
    # 2. Remove punctuation
    PUNC_TO_REMOVE = string.punctuation
    texts = [text.translate(str.maketrans('', '', PUNC_TO_REMOVE)) for text in texts]
    # 3. Lemmatize the text
    lemma = nltk.wordnet.WordNetLemmatizer()
    texts = [lemma.lemmatize(text) for text in texts]

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


if __name__=="__main__":
    create_vocabulary("train.csv", min_freq=5)
    w2i = json.load(open("w2i.json"))
    batch = convert_texts_to_indices(texts=["mathematics world there", "good morning!"], word2idx=w2i, pad_idx=0)
    print(batch)