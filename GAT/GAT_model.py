import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt
import torch.optim as optim
import gc
from tqdm import tqdm


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
        
        # Linear Layer for classification (50 classes)
        self.fc = nn.Linear(hidden_dim, output_dim)

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

    def get_attention_weights(self, data):
        # It is like a shoerter forward pass
        # to use only on trined dataset
        # to pass one graph at a time

        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x, attention_weights = self.gat(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)

        # Returns the attention weights for visualization
        return attention_weights


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


def GAT_train(
    model, 
    train_loader, 
    test_loader, 
    num_epochs=10, 
    learning_rate=0.001, 
    accumulation_steps=4  # Number of batches to accumulate gradients
):
    # Find device where the model is
    device = next(model.parameters()).device

    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    # Define the loss function (CrossEntropyLoss for multi-class classification)
    criterion = torch.nn.CrossEntropyLoss()

    # Define the optimizer (Adam)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_test_accuracy = 0
    best_model_state = None

    for epoch in range(num_epochs):
        # ---- Training Phase ----
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train_samples = 0

        optimizer.zero_grad()  # Zero gradients before starting an epoch

        # Progress bar for the training epoch
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", unit="batch", leave=True) as batch_bar:
            for batch_idx, batch in enumerate(batch_bar):
                # Move batch to device
                batch = batch.to(device)

                # Forward pass: Get predictions
                out = model(batch)  # Ensure model outputs logits (not softmax)

                # Compute the loss
                loss = criterion(out, batch.y)

                # Backward pass: Compute gradients
                loss = loss / accumulation_steps  # Scale the loss
                loss.backward()

                # Accumulate loss for logging
                total_train_loss += loss.item() * accumulation_steps

                # Calculate accuracy
                _, predicted = torch.max(out, dim=1)  # Predicted classes
                correct_train += (predicted == batch.y).sum().item()
                total_train_samples += batch.y.size(0)

                # Perform optimizer step after accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                    optimizer.step()
                    optimizer.zero_grad()

                # GPU memory monitoring
                if device.type == "cuda":
                    current_memory = torch.cuda.memory_allocated(device) / 1e6  # Convert bytes to MB
                    peak_memory = torch.cuda.max_memory_allocated(device) / 1e6
                    batch_bar.set_postfix(
                        loss=loss.item() * accumulation_steps,
                        mem_used=f"{current_memory:.2f}MB",
                        peak_mem=f"{peak_memory:.2f}MB"
                    )

                # Clean up memory
                del batch, out, loss
                torch.cuda.empty_cache()  # Clear unused memory

        # Calculate average train loss and accuracy
        epoch_train_loss = total_train_loss / len(train_loader)
        epoch_train_accuracy = correct_train / total_train_samples
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)

        print(f"\tEpoch {epoch+1}/{num_epochs} [Train], Loss: {epoch_train_loss:.4f}, Accuracy: {epoch_train_accuracy:.4f}")

        # ---- Evaluation Phase ----
        model.eval()  # Set model to evaluation mode
        total_test_loss = 0
        correct_test = 0
        total_test_samples = 0

        with torch.no_grad():  # Disable gradient computation for evaluation
            with tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Test]", unit="batch", leave=True) as batch_bar:
                for batch in batch_bar:
                    # Move batch to device
                    batch = batch.to(device)

                    # Forward pass: Get predictions
                    out = model(batch)

                    # Compute the loss
                    loss = criterion(out, batch.y)

                    # Accumulate loss for logging
                    total_test_loss += loss.item()

                    # Calculate accuracy
                    _, predicted = torch.max(out, dim=1)  # Predicted classes
                    correct_test += (predicted == batch.y).sum().item()
                    total_test_samples += batch.y.size(0)

                    # Clean up memory
                    del batch, out, loss

        # Calculate average test loss and accuracy
        epoch_test_loss = total_test_loss / len(test_loader)
        epoch_test_accuracy = correct_test / total_test_samples
        test_losses.append(epoch_test_loss)
        test_accuracies.append(epoch_test_accuracy)

        print(f"\tEpoch {epoch+1}/{num_epochs} [Test], Loss: {epoch_test_loss:.4f}, Accuracy: {epoch_test_accuracy:.4f}\n")

        # ---- Update Best Model Based on Test Accuracy ----
        if epoch_test_accuracy >= best_test_accuracy:
            best_test_accuracy = epoch_test_accuracy
            best_model_state = model.state_dict()

    # Load the best model state (based on test accuracy)
    model.load_state_dict(best_model_state)

    # Convert lists to standard Python types for JSON serialization
    results_dict = {
        "train_losses": [float(x) for x in train_losses],
        "test_losses": [float(x) for x in test_losses],
        "train_accuracies": [float(x) for x in train_accuracies],
        "test_accuracies": [float(x) for x in test_accuracies],
        "best_test_accuracy": float(best_test_accuracy),
    }

    return model, results_dict

def GAT_eval(model, dataloader):
    """
    Evaluate the trained model on the provided dataloader and return the predicted labels.

    Args:
        model (torch.nn.Module): The trained model.
        dataloader (DataLoader): The dataloader for the dataset to evaluate on.

    Returns:
        torch.Tensor: The predicted labels for the dataset.
    """
    # Find device where the model is
    device = next(model.parameters()).device

    model.eval()  # Set model to evaluation mode
    all_predictions = []

    # No need to calculate gradients during evaluation
    with torch.no_grad():
        for batch in dataloader:
            # Move the batch to the same device as the model
            batch = batch.to(device)

            # Forward pass: Get predictions
            outputs = model(batch)

            # Get the predicted class with the highest logit value
            _, predicted_labels = torch.max(outputs, dim=1)

            # Collect all the predictions in the list
            all_predictions.extend(predicted_labels.cpu().numpy())  # Convert tensor to numpy array and extend list

    return [int(x) for x in all_predictions]