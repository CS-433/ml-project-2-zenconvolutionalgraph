from torch_geometric.nn import global_mean_pool, global_max_pool
import torch.nn as nn
import torch
import torch_geometric.nn as pyg_nn
import torch.nn.functional as F
from tqdm import tqdm
import pickle as pkl
class SimpleGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super().__init__()
        self.conv1 = pyg_nn.GCNConv(input_dim, hidden_dim1)
        self.bn1 = pyg_nn.BatchNorm(hidden_dim1)
        #self.conv2 = pyg_nn.GCNConv(hidden_dim1, hidden_dim2)
        self.linear = nn.Linear(hidden_dim1, output_dim)
        self.dropout = nn.Dropout(p=0.5)
        #he init
        nn.init.kaiming_uniform_(self.linear.weight)



    def forward(self, x, edge_index, batch):
        #print(x.shape)
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        #x = global_max_pool(x, batch)
        #print(x.shape)
        #print(edge_index.min(), edge_index.max())
        #x = F.relu(x)
        #x = self.conv2(x, edge_index)
        
        #print(x.shape)
        x = F.relu(x)
        x = self.dropout(x)
        #print(x.shape)
        x = global_max_pool(x, batch)
        x = self.linear(x)
        #print(x.shape)
        return x


def GCN_train(model, optimizer, loss_fn, train_loader, test_loader, device, 
        num_epochs=10):
    #gc.collect()
    y_test = []
    best_prediction = []
    y_train = []
    best_pred_y_train = []
    best_test_acc = 0
    best_model_state = None
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        #print(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}")
        predicted_label = []
        pred_y_train = []
        pred_y_test = []
        # train the model
        
        model.train()
        total_loss = 0
        train_acc = 0
        train_sample = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        for i , batch in enumerate(progress_bar):
            batch = batch.to(device)
            #print(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}")
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = loss_fn(out, batch.y)
            if epoch == 0:
                y_train.append(batch.y)
            #print("Capturing:", torch.cuda.is_current_stream_capturing())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            
            #calculate train accuracy
            _, predicted = torch.max(out, dim=1)
            pred_y_train.append(predicted)
            #progress_bar.set_postfix({"Batch Loss": loss.item(), "out": (predicted==batch.y).sum().item()})
            #progress_bar.set_postfix({"Batch Loss": loss.item(), "out": predicted})
            train_acc += (predicted==batch.y).sum().item()
            train_sample += batch.y.size(0)
            #clean memory
            del batch
            #torch.cuda.empty_cache()

        # summary on training one epoch
        epoch_train_loss = total_loss/ len(train_loader)
        epoch_train_accuracy = train_acc/ train_sample
        train_losses.append(epoch_train_loss)
        train_accuracies.append(epoch_train_accuracy)
        print(f"\tEpoch {epoch+1}/{num_epochs}, Loss: {epoch_train_loss}, Train accuracy: {epoch_train_accuracy}\n")
        #print(torch.cuda.memory_summary(device=None, abbreviated=False))


        #evaluate the model
        model.eval()
        total_test_loss = 0
        test_acc = 0
        test_sample = 0 

        progress_bar_test = tqdm(test_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=False)
        with torch.no_grad():
            for i, batch in enumerate(progress_bar_test):
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = loss_fn(out, batch.y)
                if epoch == 0:
                    y_test.append(batch.y)

                total_test_loss += loss.item()

                _, predicted = torch.max(out, dim=1)
                progress_bar_test.set_postfix({"Batch Loss": loss.item(), "out": (predicted==batch.y).sum().item()})
                predicted_label.append(predicted)
                test_acc += (predicted==batch.y).sum().item()
                test_sample += batch.y.size(0)

                del batch, out, loss
        
        #summary of one epoch on training set
        epoch_test_loss = total_test_loss / len(test_loader)
        epoch_test_accuracy = test_acc / test_sample
        test_losses.append(epoch_test_loss)
        test_accuracies.append(epoch_test_accuracy)

        print(f"\tEpoch {epoch+1}/{num_epochs}, Loss: {epoch_test_loss:.4f}, test accuracy: {epoch_test_accuracy:.4f}\n")

        #Get best model from test accuracy
        if epoch_test_accuracy > best_test_acc:
            best_test_acc = epoch_test_accuracy
            best_prediction = predicted_label
            best_model_state = model.state_dict()
            best_pred_y_train = pred_y_train
        best_model_state = model.state_dict()
        model.load_state_dict(best_model_state)
        all_labels = torch.cat(best_prediction).cpu().numpy()
    # Convert lists to standard Python types for JSON serialization
        results_dict = {
        "train_losses": [float(x) for x in train_losses],
        "test_losses": [float(x) for x in test_losses],
        "train_accuracies": [float(x) for x in train_accuracies],
        "test_accuracies": [float(x) for x in test_accuracies],
        "best_test_accuracy": float(best_test_acc),
        "pred_y_test" : torch.cat(best_prediction).cpu().numpy(),
        #"pred_y_test" : torch.cat(predicted_label).cpu().numpy(),
        "y_test":torch.cat(y_test).cpu().numpy(),
        "pred_y_train": torch.cat(best_pred_y_train).cpu().numpy(),
        #"pred_y_train": torch.cat(pred_y_train).cpu().numpy(),
        "y_train": torch.cat(y_train).cpu().numpy()

        }
        with open('/home/zhzhou/GNN_E/data/results/GCN/GCNModel_result_sup_subject.pkl','wb') as f:
            pkl.dump(results_dict,f)
        print('Result saved!')
    return model, results_dict