import torch

class SimpleRNNModel(torch.nn.Module):
    def __init__(self, embedding_dim:int, hidden_dim:int, output_dim:int, pad_idx:int, vocab_size:int):
        super(SimpleRNNModel, self).__init__()
        # Create the embedding table. This is used to convert word indices to word embeddings.
        self.embedding = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=pad_idx)
        # Create the RNN layer. 
        self.rnn = torch.nn.RNN(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True)
        # Create the output linear layer
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        
    def forward(self, texts):
        # The goal of this forward function is to encode the texts into fixed sized representations
        # Input
        # texts: [batch_size, seq_len]
        # Output: 
        # output: [batch_size, hidden_dim]
        
        texts_lengths = (texts!=0).sum(dim=1)-1
        texts_embedded = self.embedding(texts)
        
        # print(texts_embedded.shape)
        # texts_embedded: [batch_size, seq_len, embedding_dim]
        output, _ = self.rnn(texts_embedded)
        # Pluck the hidden state of the last valid token
        to_out = torch.stack([output[i, texts_lengths[i], :] for i in range(output.shape[0])])
        
        # Focus on the last hidden state 
        out = self.fc(to_out)
        
        return out
        
        

if __name__=="__main__":
    model = SimpleRNNModel(embedding_dim=100, hidden_dim=256, output_dim=10, pad_idx=0, vocab_size=10000)
    batch = torch.randint(0, 2, (2, 10))
    out = model(batch)
    print(out.shape)