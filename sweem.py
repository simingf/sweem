import torch
import torch.nn as nn
import torch.optim as optim

# SWEEM code imports
import data
from model import SWEEM
from loss import temp_loss, neg_par_log_likelihood
from train import train
import checkpoint

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on", device)

## FIRST TIME SETUP ##
# # model
# model = SWEEM(**settings["model"])
# model.to(device)

# # settings
# settings = {
#     "model": {
#         "rna_dim": 5540,
#         "scna_dim": 5507,
#         "methy_dim": 4846,
#         "use_rna": True,
#         "use_scna": True,
#         "use_methy": False,
#         "hidden_dim": 64,
#         "self_att": False,
#         "cross_att": False,
#         "device": device
#     },
#     "train": {
#         "batch_size": 32,
#         "lr": 0.0001,
#         "l2": 1e-5,
#         "epochs": 5,
#         "epoch_mod": 1
#     }
# }

# # optimizer
# optimizer = optim.Adam(model.parameters(), lr=settings["train"]["lr"], weight_decay=settings["train"]["l2"])

## RETRAINING MODEL ##
model, settings, optimizer, epoch_train_losses, epoch_val_losses = checkpoint.load("./sweem.model", SWEEM, device, optim.Adam, inference=False)

# settings adjustments
settings["train"]["batch_size"] = 380
settings["train"]["lr"] = 0.00001
settings["train"]["epochs"] = 101
settings["train"]["epoch_mod"] = 10
print(settings)

# binary cross entropy loss
# criterion = nn.BCELoss()
# cox proportional hazard loss
criterion = neg_par_log_likelihood

# dataloaders
train_dataloader, test_dataloader = data.get_train_test(path='./Data/OmicsData/data.csv', batch_size=settings["train"]["batch_size"])

# train the model
epoch_train_losses, epoch_val_losses = train(model, settings, optimizer, criterion, train_dataloader, test_dataloader, device)

# Sanity Check
model.eval()
with torch.no_grad():
    for (batchX, batchY) in test_dataloader:
        batchX = batchX.to(device)
        rna = batchX[:, :5540].to(device)
        scna = batchX[:, 5540:11047].to(device)
        methy = batchX[:, 11047:].to(device)
        time = batchY[:,0].reshape(-1, 1).to(device)
        event = batchY[:,1].reshape(-1, 1).to(device)
        outputs = model(event, rna=rna, scna=scna, methy=methy)
        
        # concat torch tensors
        table = torch.cat((time, event, outputs), 1)
        
        rows_to_print = 10
        # print row by row
        print("Sanity Check:")
        print("time, event, predicted")
        for row in table[:rows_to_print]:
            print(row.tolist())
        break

# save and load for training
checkpoint.save("./sweem.model", model, settings, optimizer, epoch_train_losses, epoch_val_losses, inference=False)

model, settings, optimizer, epoch_train_losses, epoch_val_losses = checkpoint.load("./sweem.model", SWEEM, device, optim.Adam, inference=False)

# save and load for inference
checkpoint.save("./sweem_inf.model", model, settings, inference=True)

model, settings = checkpoint.load("./sweem_inf.model", SWEEM, device, inference=True)