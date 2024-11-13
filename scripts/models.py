import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F
import networkx as nx


class GATModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads=50, device = None):
        super().__init__()

        # Set device (if not provided, use GPU if available, otherwise use CPU)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # GAT Layer: multi-head attention with num_heads=50
        self.gat = pyg_nn.GATConv(input_dim, #the number of features each node has).
                                hidden_dim,  #The output dimension of each node after applying the GAT layer
                                heads=num_heads,
                                concat=False #This means that the attention heads will be averaged rather than concatenated. 
                                )
        
        # TODO:
        # add pooling
        
        # Linear Layer for classification (50 classes)
        self.fc = nn.Linear(hidden_dim, output_dim)

        # TODO:
        # add non linearity

    def forward(self, data):
        # data is a Batch object that contains a batch of graphs
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
           #data.x: (num_nodes, input_dim)
            # (2, num_edges) 
        
        # Apply the GAT layer to all graphs in the batch
        # The GAT layer returns the attention weights (a tensor) and the updated node features
        #x, attention_weights = self.gat(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        x = self.gat(x, edge_index, edge_attr=edge_attr)

        # Store attention weights for visualization purposes
        #self.attention_weights = attention_weights  # Shape: (num_heads, num_edges)
        
        # Mean pooling: Perform mean pooling across the batch of graphs
        #This operation pools the node features (x) into a single graph-level representation for each graph in the batch.
        x = pyg_nn.global_mean_pool(x, data.batch.to(self.device)) #(num_graphs_in_batch, hidden_dim)
        
        # Fully connected layer for classification
        x = self.fc(x)
        
        return x #return logits

    def get_attention_weights(self):
        # Returns the attention weights for visualization
        return self.attention_weights


def visualize_attention_weights(model, data, head=0):
    # Get the attention weights from the model
    attention_weights = model.get_attention_weights()  # Shape: (num_heads, num_edges)
    
    # Extract attention weights for the specified head
    head_attention_weights = attention_weights[head].detach().cpu().numpy()  # (num_edges,)
    
    # Create a graph (you can choose the first graph in the batch for visualization)
    edge_index = data.edge_index.cpu().numpy()  # Edge index of the graph (2, num_edges)
    
    # Create a NetworkX graph object
    G = nx.Graph()
    
    # Add nodes
    for node in range(data.x.size(0)):
        G.add_node(node)
    
    # Add edges with weights (attention weights)
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        G.add_edge(src, dst, weight=head_attention_weights[i])
    
    # Draw the graph, with edge width proportional to attention weight
    pos = nx.spring_layout(G)  # Layout for the nodes
    edge_widths = [G[u][v]['weight'] * 10 for u, v in G.edges()]  # Scale edge weights for visualization
    
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=12, font_weight='bold',
            width=edge_widths, edge_color=edge_widths, edge_cmap=plt.cm.Blues)
    plt.title(f"Visualization of Attention Weights for Head {head}")
    plt.show()


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
                    X_train, #(samples, nodes * window)
                    Y_train, 
                    X_val, 
                    Y_val, 
                    optimizer="adam", 
                    learning_rate=0.01, 
                    epochs=100, 
                    batch_size=32):
        
        # Move data to the appropriate device
        X_train, Y_train = X_train.to(self.device), Y_train.to(self.device)
        X_val, Y_val = X_val.to(self.device), Y_val.to(self.device)
        
        # Set up optimizer
        if optimizer == "adam":
            optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer == "sgd":
            optim = torch.optim.SGD(self.parameters(), lr=learning_rate)
        elif optimizer == "rmsprop":
            optim = torch.optim.RMSprop(self.parameters(), lr=learning_rate)
        
        # Loss function
        criterion = nn.CrossEntropyLoss() if self.output_type == "classification" else nn.MSELoss()
        
        # Initialize variables to accumulate losses
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Reset epoch losses
            epoch_train_loss = 0
            epoch_val_loss = 0
            
            # Batch processing
            permutation = torch.randperm(X_train.size(0))
            for i in range(0, X_train.size(0), batch_size):
                indices = permutation[i:i+batch_size]
                batch_x, batch_y = X_train[indices], Y_train[indices]
                
                # Zero gradients, forward pass, backward pass, and optimizer step
                optim.zero_grad()
                outputs = self(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optim.step()
                
                epoch_train_loss += loss.item()
            
            # Calculate validation loss after each epoch
            with torch.no_grad():
                val_outputs = self(X_val)
                val_loss = criterion(val_outputs, Y_val)
                epoch_val_loss += val_loss.item()
            
            # Optional: print epoch status
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {epoch_train_loss / len(X_train):.4f}, Val Loss: {epoch_val_loss / len(X_val):.4f}")
            
            # Accumulate total losses
            train_losses.append(epoch_train_loss)
            val_losses.append(epoch_val_loss)
        
        # Return total loss for both training and validation
        return train_losses, val_losses

    def predict(self, X):
        X = X.to(self.device)
        with torch.no_grad():
            outputs = self(X)
        if self.output_type == "classification":
            return torch.argmax(outputs, dim=1)
        return outputs
    

class Transformer(nn.Module):
    def __init__(self,
                input_dim, 
                num_heads, 
                num_layers, 
                hidden_dim, 
                num_classes, 
                dropout=0.1
            ):

        super().__init__()

        # [CLS] token embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim)) 
            # 1cls tokes x 1 batch x dim ROW (number of timepoits for each timeseries)
            # nn.Parameter --> make it a learnable parameter, meaning the model can update this token's values during training.

        # Row (time series) embedding layer
        self.row_embedding = nn.Linear(input_dim, hidden_dim)
            # linear layer that projects each row (a time series for one ROI) from input_dim (number of time steps) to hidden_dim (transformer’s internal dimension).

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, 
                                                   nhead=num_heads, 
                                                   dropout=dropout
                                                   )
        # stacks multiple copies of encoder_layer to create a deeper transformer.
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, 
                                                        num_layers=num_layers
                                                        )

        # Classification head
        self.fc = nn.Linear(hidden_dim, num_classes)
            #linear layer that maps the final embedding of the [CLS] token (dimension hidden_dim) to num_classes

    def forward(self, x):
        # Input x has shape (batch_size, num_rows, time_steps)
        # - batch_size: Number of matrices in the batch.
        # - num_rows: Number of rows (each corresponding to an ROI or time series) in each matrix.
        # - time_steps: Number of time steps in each time series (length of each row).

        batch_size, num_rows, time_steps = x.shape

        # Flatten each row's time series into a single vector by keeping all time steps in sequence
        # After reshaping, x has shape (batch_size, num_rows, num_timesteps)
        x = x.reshape(batch_size, num_rows, -1)

        # Expand the [CLS] token for each item in the batch and concatenate it with the rows.
        # This [CLS] token will act as the summary representation for the entire matrix.
        # After expanding, `cls_tokens` has shape (batch_size, 1, num_timesteps)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # Concatenate the [CLS] token with the input x along the row dimension.
        # The resulting shape of x is (batch_size, num_rows + 1, num_timesteps)
        x = torch.cat((cls_tokens, x), dim=1)

        # Embed each row (including the [CLS] token) to the transformer's hidden dimension
        # Shape of x after embedding: (batch_size, num_rows + 1, hidden_dim)
        x = self.row_embedding(x)

        # Pass the embedded rows through the transformer encoder to get contextualized embeddings
        # Shape of x after transformer encoder: (batch_size, num_rows + 1, hidden_dim)
        x = self.transformer_encoder(x)

        # Extract the embedding of the [CLS] token, which is the first row in x
        # Shape of cls_embedding: (batch_size, hidden_dim)
        cls_embedding = x[:, 0, :]

        # Pass the [CLS] token's embedding through the classification head to get final class scores
        # Shape of out: (batch_size, num_classes)
        out = self.fc(cls_embedding)

        return out # logits


