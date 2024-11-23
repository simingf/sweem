import torch

def train(model, settings, optimizer, criterion, train_dataloader, test_dataloader, device):
    epoch_train_losses = []
    epoch_val_losses   = []

    for epoch in range(settings["train"]["epochs"]):
        epoch_train_loss = 0
        epoch_val_loss   = 0

        model.train()
        for i, (batchX, batchY) in enumerate(train_dataloader):        
            batchX = batchX.to(device)
            rna = batchX[:, :5540]
            scna = batchX[:, 5540:11047]
            methy = batchX[:, 11047:]
            time = batchY[:,0].reshape(-1, 1).to(device)
            event = batchY[:,1].reshape(-1, 1).to(device)

            outputs = model(event, rna=rna, scna=scna, methy=methy)
            
            loss = criterion(outputs, time, event)
            # loss = criterion(outputs, event)
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_train_loss += loss.item()
            
        model.eval()
        with torch.no_grad():
            for i, (batchX, batchY) in enumerate(test_dataloader):
                batchX = batchX.to(device)
                rna = batchX[:, :5540]
                scna = batchX[:, 5540:11047]
                methy = batchX[:, 11047:]
                time = batchY[:,0].reshape(-1, 1).to(device)
                event = batchY[:,1].reshape(-1, 1).to(device)
                outputs = model(event, rna=rna, scna=scna, methy=methy)

                loss = criterion(outputs, time, event)
                # loss = criterion(outputs, event)
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