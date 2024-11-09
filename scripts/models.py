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