import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm


class FNN(nn.Module):

    # Adapted from: https://github.com/rohanrao619/Deep_Neural_Networks_from_Scratch/blob/master/Deep_Neural_Networks_from_Scratch_using_NumPy.ipynb

    def __init__(self, 
                input_size, 
                output_size, 
                hidden_dims, 
                output_type="classification", 
                initializer="xavier", 
                activation="lrelu", 
                leaky_relu_slope=0.1, 
                device=None):
        
        super().__init__()

        # Set up device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_type = output_type
        self.layers = nn.ModuleList()
        
        # Layer dimensions
        layer_dims = [input_size] + hidden_dims + [output_size] #output[100, 500, 10]
        #print(layer_dims)

        # Define layers
        for i in range(len(layer_dims) - 1):
            layer = nn.Linear(layer_dims[i], layer_dims[i+1])
            # Weight initialization
            if initializer == 'xavier':
                nn.init.xavier_uniform_(layer.weight)
            elif initializer == 'he':
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            else:
                nn.init.normal_(layer.weight)
            self.layers.append(layer)

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(leaky_relu_slope)
        else:
            self.activation = nn.Identity()

        # Move model to the correct device
        self.to(self.device)

    def forward(self, X):
        X = X.to(self.device)
        for layer in self.layers[:-1]:  # Apply activation after each layer except the output
            X = self.activation(layer(X))
        # Output layer
        X = self.layers[-1](X)
        if self.output_type == "classification":
            return X # Just return raw logits without softmax
        return X


    def train_model(self, 
                    X_train, Y_train, 
                    X_val, Y_val, 
                    optimizer="adam", 
                    learning_rate=0.01, 
                    epochs=10, 
                    batch_size=32,
                    accumulation_steps=1):
        
        # Wrap data in DataLoader for efficient loading
        train_data = DataLoader(TensorDataset(X_train, Y_train), 
                                batch_size=batch_size, 
                                shuffle=True, 
                                pin_memory=True)
        val_data = DataLoader(TensorDataset(X_val, Y_val), 
                            batch_size=batch_size, 
                            pin_memory=True)
        
        # Set up optimizer
        if optimizer == "adam":
            optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer == "sgd":
            optim = torch.optim.SGD(self.parameters(), lr=learning_rate)
        elif optimizer == "rmsprop":
            optim = torch.optim.RMSprop(self.parameters(), lr=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")
        
        # Loss function
        criterion = nn.CrossEntropyLoss() if self.output_type == "classification" else nn.MSELoss()
        criterion.to(self.device)
        
        # Enable mixed precision training
        scaler = torch.cuda.amp.GradScaler()

        # Lists to store losses
        losses_train = []
        losses_val = []

        # Training loop
        for epoch in range(epochs):
            self.train()
            epoch_train_loss = 0
            
            with tqdm(train_data, desc=f"Epoch {epoch+1}/{epochs}", unit="batch") as t:
                for batch_idx, (batch_x, batch_y) in enumerate(t):
                    # Move data to GPU, avoid blocking
                    batch_x = batch_x.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)

                    # Zero gradients only for first step of accumulation
                    if batch_idx % accumulation_steps == 0:
                        optim.zero_grad()

                    # Mixed precision forward pass and loss computation
                    with torch.cuda.amp.autocast():
                        outputs = self(batch_x)
                        loss = criterion(outputs, batch_y) / accumulation_steps

                    # Backward pass and gradient scaling
                    scaler.scale(loss).backward()

                    # Step optimizer only after accumulation steps
                    if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_data):
                        scaler.step(optim)
                        scaler.update()

                    # Accumulate training loss
                    epoch_train_loss += loss.item() * len(batch_x)

                    # Free up memory
                    del batch_x, batch_y, outputs, loss
                    torch.cuda.empty_cache()

            # Normalize training loss
            epoch_train_loss /= len(X_train)
            losses_train.append(epoch_train_loss)

            # Validation loop
            self.eval()
            epoch_val_loss = 0
            with torch.no_grad():
                for val_x, val_y in val_data:
                    val_x = val_x.to(self.device, non_blocking=True)
                    val_y = val_y.to(self.device, non_blocking=True)

                    # Forward pass
                    val_outputs = self(val_x)
                    val_loss = criterion(val_outputs, val_y)

                    # Accumulate validation loss
                    epoch_val_loss += val_loss.item() * len(val_x)

                    # Free up memory
                    del val_x, val_y, val_outputs, val_loss
                    torch.cuda.empty_cache()
            
            epoch_val_loss /= len(X_val)
            losses_val.append(epoch_val_loss)

            # Print epoch statistics
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")

        # Return the lists of losses
        return losses_train, losses_val


    # def train_model(self, 
    #             X_train, #(samples, nodes * window)
    #             Y_train, 
    #             X_val, 
    #             Y_val, 
    #             optimizer="adam", 
    #             learning_rate=0.01, 
    #             epochs=100, 
    #             batch_size=32):
    
    #     # Move data to the appropriate device
    #     X_train, Y_train = X_train.to(self.device), Y_train.to(self.device)
    #     X_val, Y_val = X_val.to(self.device), Y_val.to(self.device)
        
    #     # Set up optimizer
    #     if optimizer == "adam":
    #         optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
    #     elif optimizer == "sgd":
    #         optim = torch.optim.SGD(self.parameters(), lr=learning_rate)
    #     elif optimizer == "rmsprop":
    #         optim = torch.optim.RMSprop(self.parameters(), lr=learning_rate)
    #     else:
    #         raise ValueError(f"Unsupported optimizer: {optimizer}")
        
    #     # Loss function
    #     criterion = nn.CrossEntropyLoss() if self.output_type == "classification" else nn.MSELoss()
    #     criterion.to(self.device)
        
    #     # Initialize variables to accumulate losses
    #     train_losses = []
    #     val_losses = []
        
    #     for epoch in range(epochs):
    #         # Reset epoch losses
    #         epoch_train_loss = 0
    #         epoch_val_loss = 0
            
    #         # Shuffle and batch processing
    #         permutation = torch.randperm(X_train.size(0))
    #         for i in range(0, X_train.size(0), batch_size):
    #             indices = permutation[i:i+batch_size]
    #             batch_x, batch_y = X_train[indices], Y_train[indices]
                
    #             # Zero gradients, forward pass, backward pass, and optimizer step
    #             optim.zero_grad()
    #             outputs = self(batch_x)
    #             loss = criterion(outputs, batch_y)
    #             loss.backward()
    #             optim.step()
                
    #             epoch_train_loss += loss.item()
            
    #         # Normalize training loss
    #         epoch_train_loss /= len(X_train) / batch_size
            
    #         # Calculate validation loss
    #         self.eval() #This ensures layers like dropout or batch normalization behave correctly.
    #         with torch.no_grad():
    #             # process X_val in batches, similar to training.
    #             for i in range(0, X_val.size(0), batch_size):
    #                 val_indices = torch.arange(i, min(i+batch_size, X_val.size(0)))
    #                 val_batch_x, val_batch_y = X_val[val_indices], Y_val[val_indices]
    #                 val_outputs = self(val_batch_x)
    #                 val_loss = criterion(val_outputs, val_batch_y)
    #                 epoch_val_loss += val_loss.item()
    #         self.train()
            
    #         # Normalize validation loss
    #         epoch_val_loss /= len(X_val) / batch_size
            
    #         # Print epoch status
    #         print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss:.4f}, Test Loss: {epoch_val_loss:.4f}")
            
    #         # Accumulate total losses
    #         train_losses.append(epoch_train_loss)
    #         val_losses.append(epoch_val_loss)
        
    #     # Return total loss for both training and validation
    #     return train_losses, val_losses

    def predict(self, X):
        X = X.to(self.device)
        with torch.no_grad():
            outputs = self(X)
        if self.output_type == "classification":
            return torch.argmax(outputs, dim=1)
        return outputs