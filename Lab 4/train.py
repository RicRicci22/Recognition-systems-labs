import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

# Use pickle to save the loss array
import pickle


# Let's try with an RNN


class RNNLM(nn.Module):
    def __init__(self, vocab_size, hidden_size=32):
        super(RNNLM, self).__init__()
        self.embedding_table = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        # Classifier to predict the next token
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs):
        # Embed the inputs
        embeddings = self.embedding_table(inputs)
        # Pass the embeddings through the RNN
        outputs, _ = self.rnn(embeddings)
        # Pass the outputs through the classifier
        logits = self.linear(outputs)

        return logits

    def generate(self, inputs, max_new_tokens=10):
        # Generate next tokens given the inputs (B, L, C)
        for _ in range(max_new_tokens):
            # Embed the inputs
            embeddings = self.embedding_table(inputs)
            # Feed the RNN with the embeddings
            _, hidden = self.rnn(embeddings)
            # Remove the first dimension (get only the last hidden state for each element in the batch)
            hidden = hidden.squeeze(0)
            # Project to the output classes
            logits = self.linear(hidden)
            # Convert to probabilities
            probs = F.softmax(logits, dim=1)
            # Get the next token
            next_token = torch.multinomial(probs, num_samples=1)
            # Concatenate to input
            inputs = torch.cat((inputs, next_token), dim=1)

        return inputs


def get_batch(data, batch_size=8, context_length=8):
    # Get batch_size random indices in the data
    random_idx = random.sample(range(len(data) - context_length), batch_size)
    # Pluck the next character after each random index
    inputs = torch.zeros((batch_size, context_length), dtype=torch.long)
    targets = torch.zeros((batch_size, context_length), dtype=torch.long)

    for i in range(batch_size):
        inputs[i, :] = data[random_idx[i] : random_idx[i] + context_length]
        targets[i, :] = data[random_idx[i] + 1 : random_idx[i] + context_length + 1]

    return inputs, targets


if __name__ == "__main__":
    with open("gameofthrones.txt", "r", encoding="utf8") as file:
        text = file.read()

    chars = sorted(list(set(text)))
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    encode = lambda s: [
        char_to_idx[c] for c in s
    ]  # s is the input string that I want to encode
    decode = lambda l: "".join(
        [idx_to_char[i] for i in l]
    )  # l is the input list of indices that I want to decode
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(text))
    train_data = data[:n]
    val_data = data[n:]

    context_length = (
        8  # This is maximum number of tokens that are allowed to fit in the context.
    )

    # Hyperparameters
    learning_rate = 0.01
    momentum = 0.9
    batch_size = 32
    combinations = [
        (32, 32),
        (32, 64),
        (32, 128),
        (32, 256),
        (128, 32),
        (128, 64),
        (128, 128),
        (128, 256),
        (512, 32),
        (512, 64),
        (512, 128),
        (512, 256),
    ]
    num_iterations = 100000
    device = "cuda:0"

    # Start training loop

    for c, h in combinations:
        print("Training with context length: ", c, " and hidden size: ", h)
        # Get the model
        model = RNNLM(len(chars), hidden_size=h)
        # Send the model to the device
        model.to(device)
        # Get the optimizer
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
        # Get loss function
        loss_fn = CrossEntropyLoss()

        loss_train = 0
        loss_array = []
        for i in tqdm(range(num_iterations)):
            # Get a batch
            inputs, targets = get_batch(
                train_data, batch_size=batch_size, context_length=c
            )
            # Send the inputs and targets to device
            inputs = inputs.to(device)
            targets = targets.to(device)
            # Get the predictions
            predictions = model(inputs)
            # We have to reshape the predictions and the targets to use cross entropy
            B, L, C = predictions.shape
            predictions = predictions.view(B * L, C)
            targets = targets.view(B * L)
            # Compute the loss
            loss = loss_fn(predictions, targets)
            # Accumulate the loss
            loss_train += loss.item()
            # Zero the gradients
            optimizer.zero_grad()
            # Compute the gradients
            loss.backward()
            # Update the parameters
            optimizer.step()

            if (i % 1000) == 0 and i != 0:
                loss_array.append(loss_train / 1000)
                loss_train = 0

        # Save the loss list
        with open(
            "loss_rnn_context" + str(c) + "_hiddensize" + str(h) + ".pkl", "wb"
        ) as file:
            pickle.dump(loss_array, file)
