import torch
from model import PathwayModuleAtt
import checkpoint
import data
import metrics
from PM import get_pathway_module

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on", device)

model, settings = checkpoint.load("./PM_inf.model", PathwayModuleAtt, device, inference=True)

train_dataloader, test_dataloader = data.get_train_test(path='./Data/PM_OmicsData/data.csv', batch_size=settings["train"]["batch_size"])

risk_scores = torch.Tensor([]).to(device)
times = torch.Tensor([]).to(device)
events = torch.Tensor([]).to(device)

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
        
        risk_scores = torch.cat((risk_scores, outputs), 0)
        times = torch.cat((times, time), 0)
        events = torch.cat((events, event), 0)

c_index = metrics.concordance_index(risk_scores, events, times).item()
print(c_index)