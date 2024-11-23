import torch
from model import SWEEM
import checkpoint
import data
import metrics
from PM import get_pathway_module

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on", device)

model, settings = checkpoint.load("./sweem_inf.model", SWEEM, device, inference=True)

train_dataloader, test_dataloader = data.get_train_test(path='./Data/OmicsData/data.csv', batch_size=settings["train"]["batch_size"])

risk_scores = torch.Tensor([]).to(device)
times = torch.Tensor([]).to(device)
events = torch.Tensor([]).to(device)

model.eval()
with torch.no_grad():
    pathway_module = torch.Tensor(get_pathway_module()).to(device)
    for (batchX, batchY) in test_dataloader:
        batchX = batchX.to(device)
        rna = batchX[:, :5540].to(device)
        scna = batchX[:, 5540:11047].to(device)
        methy = batchX[:, 11047:].to(device)
        time = batchY[:,0].reshape(-1, 1).to(device)
        event = batchY[:,1].reshape(-1, 1).to(device)
        outputs = model(event, rna=rna, scna=scna, methy=methy)
        
        risk_scores = torch.cat((risk_scores, outputs), 0)
        times = torch.cat((times, time), 0)
        events = torch.cat((events, event), 0)

c_index = metrics.concordance_index(risk_scores, events, times).item()
print(c_index)