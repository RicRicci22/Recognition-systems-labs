from tqdm import tqdm
import torch
import pickle
import os

def train(model, trainloader, valloader, epochs, optimizer, loss_fn, device, log_every=100, save_directory=None):
  # Store the training and validation accuracies and losses in lists for plotting
  acc_train = []
  loss_train = []
  acc_val = []
  loss_val = []

  # Start the training loop
  for epoch in range(epochs):
    loss_training = 0
    accuracy_training = 0 
    print("Epoch:", epoch)
    for i, data in enumerate(tqdm(trainloader)):
      imgs, labels = data
      # Send data to GPU
      imgs = imgs.to(device)
      labels = labels.to(device)
      predictions = model(imgs)
      loss = loss_fn(predictions.squeeze(),labels.view(-1))
      # Zero the gradients in the optimizer
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()
      # Accumulate the 
      loss_training += loss.item()
      # CALCULATE THE TRAINING ACCURACY
      predictions = torch.sigmoid(predictions.detach().squeeze()) # Turn the predictions into probabilities by applying the sigmoid function
      accuracy_training += (sum(torch.round(predictions)==labels.view(-1))/imgs.shape[0]).item()
      # Save the metrics for plotting
      if i%log_every == 0 and i!=0:
        loss_train.append(loss_training/(i+1))
        acc_train.append(accuracy_training/(i+1))
      
    # Test on validation images 
    loss_validation = 0
    accuracy_validation = 0
    with torch.no_grad(): # Use no_grad to avoid storing the gradients and speed up calculations
      model.eval() # Put the model in evaluation mode
      for i, data in enumerate(tqdm(valloader)):
        imgs, labels = data
        # Send data to GPU
        imgs = imgs.to(device)
        labels = labels.to(device)
        predictions = model(imgs)
        loss = loss_fn(predictions.squeeze(),labels.view(-1))
        loss_validation += loss.item()
        # Calculate the validation accuracy 
        predictions = torch.sigmoid(predictions.detach().squeeze())
        accuracy_validation += (sum(torch.round(predictions)==labels.view(-1))/imgs.shape[0]).item()
        if i%10 == 0 and i!=0:
            loss_val.append(loss_validation/(i+1))
            acc_val.append(accuracy_validation/(i+1))
        
      # Put the model in training mode again
      model.train()

    print("\nTraining loss epoch ",epoch,":",loss_training/len(trainloader))
    print("Training accuracy epoch ",epoch,":",accuracy_training/len(trainloader))
    print("Validation loss epoch ",epoch,":",loss_validation/len(valloader))
    print("Validation accuracy epoch ",epoch,":",accuracy_validation/len(valloader))

  if save_directory is not None:
      os.makedirs(save_directory, exist_ok=True)
      # Save the model 
      torch.save(model.state_dict(), os.path.join(save_directory, f"model_{epoch+1}_epochs.pt"))
      with open(os.path.join(save_directory,"acc_train.pkl"),"wb") as f:
          pickle.dump(acc_train,f)
      with open(os.path.join(save_directory,"loss_train.pkl"),"wb") as f:
          pickle.dump(loss_train,f)
      with open(os.path.join(save_directory,"acc_val.pkl"),"wb") as f:
          pickle.dump(acc_val,f)
      with open(os.path.join(save_directory,"loss_val.pkl"),"wb") as f:
          pickle.dump(loss_val,f)