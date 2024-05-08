from torch.utils.data import Dataset, DataLoader
import pandas
import torch

class MultiLabelDataset(Dataset):
    def __init__(self, data_path:str, split:str="train"):
        self.data = pandas.read_csv(data_path)
        self.split = split
        # Sanity check
        self.data = self.data.dropna()
        print(f"Data loaded successfully, total number of {self.split} samples: {len(self.data)}")
        self.labels = ["Computer Science", "Physics", "Mathematics", "Statistics", "Quantitative Biology", "Quantitative Finance"]
    
    def __len__(self):
        return len(self.data)
    
    def __getnlabels__(self):
        return self.labels
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        title = row["TITLE"].lower()
        abstract = row["ABSTRACT"].lower()
        # Get the labels
        cs = row["Computer Science"]
        p = row["Physics"]
        m = row["Mathematics"]
        s = row["Statistics"]
        qb = row["Quantitative Biology"]
        qf = row["Quantitative Finance"]
        labels = torch.tensor([cs, p, m, s, qb, qf], dtype=torch.float)
        
        return title, abstract, labels

def custom_collate(batch):
    titles, abstracts, labels = zip(*batch)
    return titles, abstracts, torch.stack(labels)
    
if __name__=="__main__":
    dataset = MultiLabelDataset("train.csv")
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=custom_collate)
    batch = next(iter(dataloader))
    titles, abstracts, labels = batch