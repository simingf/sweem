import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# SWEEM code imports
import data
from model import PathwayModuleAtt
import checkpoint

def get_pathway_module():
    """
    Returns the pathway_module data as a numpy array.

    Args:
    None

    Returns:
    pathway_module (numpy array): Pathway module data.
    """
    # load in pathway module data
    pathway_module = pd.read_csv("./Data/PM_OmicsData/pathway_module.csv", index_col=0)

    # convert to numpy array
    pathway_module = pathway_module.to_numpy().T
    # print(pathway_module.shape) #(4801,860)

    return pathway_module

def train(model, settings, optimizer, criterion, train_dataloader, test_dataloader, device):
    pathway_module = torch.Tensor(get_pathway_module()).to(device)
    
    epoch_train_losses = []
    epoch_val_losses   = []

    for epoch in range(settings["train"]["epochs"]):
        epoch_train_loss = 0
        epoch_val_loss   = 0

        model.train()
        for i, (batchX, batchY) in enumerate(train_dataloader):        
            batchX = batchX.to(device)
            rna = batchX[:, :4801]
            scna = batchX[:, 4801:9602]
            methy = batchX[:, 9602:]
            time = batchY[:,0].reshape(-1, 1).to(device)
            event = batchY[:,1].reshape(-1, 1).to(device)
            
            rna = torch.matmul(rna, pathway_module)
            scna = torch.matmul(scna, pathway_module)
            methy = torch.matmul(methy, pathway_module)
            
            x = torch.cat((rna, scna, methy), dim=1)
            outputs = model(x)
            
            loss = criterion(outputs, event)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_train_loss += loss.item()
            
        model.eval()
        with torch.no_grad():
            for i, (batchX, batchY) in enumerate(test_dataloader):
                batchX = batchX.to(device)
                rna = batchX[:, :4801]
                scna = batchX[:, 4801:9602]
                methy = batchX[:, 9602:]
                time = batchY[:,0].reshape(-1, 1).to(device)
                event = batchY[:,1].reshape(-1, 1).to(device)
                
                rna = torch.matmul(rna, pathway_module)
                scna = torch.matmul(scna, pathway_module)
                methy = torch.matmul(methy, pathway_module)
                
                x = torch.cat((rna, scna, methy), dim=1)
                outputs = model(x)

                loss = criterion(outputs, event)
                epoch_val_loss += loss.item()

        # Save and print losses
        epoch_train_loss /= len(train_dataloader)
        epoch_val_loss /= len(test_dataloader)
        epoch_train_losses.append(epoch_train_loss)
        epoch_val_losses.append(epoch_val_loss)
        if epoch % settings["train"]["epoch_mod"] == 0:
            print(f"Epoch {epoch + 1} training loss: {epoch_train_loss}")
            print(f"Epoch {epoch + 1} validation loss: {epoch_val_loss}")
    
    return epoch_train_losses, epoch_val_losses

if __name__ == "__main__":

    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running on", device)
    
    # settings
    settings = {
        "model": {
            "hidden_dim": 128,
            "device": device
        },
        "train": {
            "batch_size": 16,
            "lr": 0.00001,
            "l2": 1e-5,
            "epochs": 201,
            "epoch_mod": 10
        }
    }

    # dataloaders
    train_dataloader, test_dataloader = data.get_train_test(path='./Data/PM_OmicsData/data.csv', batch_size=settings["train"]["batch_size"])

    ## FIRST TIME SETUP ##

    # # model
    # model = PathwayModuleAtt(**settings["model"])
    # model.to(device)

    # # optimizer
    # optimizer = optim.Adam(model.parameters(), lr=settings["train"]["lr"], weight_decay=settings["train"]["l2"])

    ## RETRAINING MODEL ##
    model, settings, optimizer, epoch_train_losses, epoch_val_losses = checkpoint.load("./PM.model", PathwayModuleAtt, device, optim.Adam, inference=False)

    # binary cross entropy loss
    criterion = nn.BCELoss()

    # train the model
    epoch_train_losses, epoch_val_losses = train(model, settings, optimizer, criterion, train_dataloader, test_dataloader, device)

    # Sanity Check
    model.eval()
    with torch.no_grad():
        pathway_module = torch.Tensor(get_pathway_module()).to(device)
        for (batchX, batchY) in test_dataloader:
            batchX = batchX.to(device)
            rna = batchX[:, :4801]
            scna = batchX[:, 4801:9602]
            methy = batchX[:, 9602:]
            time = batchY[:,0].reshape(-1, 1).to(device)
            event = batchY[:,1].reshape(-1, 1).to(device)
            
            rna = torch.matmul(rna, pathway_module)
            scna = torch.matmul(scna, pathway_module)
            methy = torch.matmul(methy, pathway_module)
            
            x = torch.cat((rna, scna, methy), dim=1)
            outputs = model(x)
            
            # concat torch tensors
            table = torch.cat((time, event, outputs), 1)
            
            # print row by row
            print("Sanity Check:")
            print("time, event, predicted")
            for row in table:
                print(row.tolist())
            break

    # save and load for training
    checkpoint.save("./PM.model", model, settings, optimizer, epoch_train_losses, epoch_val_losses, inference=False)

    model, settings, optimizer, epoch_train_losses, epoch_val_losses = checkpoint.load("./PM.model", PathwayModuleAtt, device, optim.Adam, inference=False)

    # save and load for inference
    checkpoint.save("./PM_inf.model", model, settings, inference=True)

    model, settings = checkpoint.load("./PM_inf.model", PathwayModuleAtt, device, inference=True)