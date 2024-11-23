import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttResNet(nn.Module):
    def __init__(self, input_dim):
        super(SelfAttResNet, self).__init__()
        self.input_dim = input_dim
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.softmax = nn.Softmax(dim=1)
        self.dense = nn.Linear(input_dim + input_dim, input_dim)
        
    def forward(self, x):
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        scores = torch.mm(queries, keys.transpose(0,1)) / (self.input_dim ** 0.5)
        attention = self.softmax(scores)   
        weighted = torch.mm(attention, values)
        cat = torch.cat((x, weighted), dim=1)
        out = self.dense(cat)
        out = F.relu(out)
        return out

class SWEEM(nn.Module):
    def __init__(self, 
                 rna_dim, 
                 scna_dim, 
                 methy_dim, 
                 use_rna,
                 use_scna,
                 use_methy, 
                 hidden_dim, 
                 self_att, 
                 cross_att,
                 device):
        super(SWEEM, self).__init__()
        self.use_rna = use_rna
        self.use_scna = use_scna
        self.use_methy = use_methy
        self.self_att = self_att
        self.cross_att = cross_att
        self.device = device
        
        if self_att:
            if use_rna:
                self.rna_att = SelfAttResNet(rna_dim) 
            if use_scna:
                self.scna_att = SelfAttResNet(scna_dim)
            if use_methy:
                self.methyl_att = SelfAttResNet(methy_dim)
        
        total_dim = 0
        if use_rna:
            total_dim += rna_dim
        if use_scna:
            total_dim += scna_dim
        if use_methy:
            total_dim += methy_dim
        
        if cross_att:
            self.cross_att = SelfAttResNet(total_dim)

        self.dense1 = nn.Linear(total_dim + 1, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, 1)

    def forward(self, event, **kwargs):
        if self.use_rna:
            rna = kwargs['rna']
            rna = torch.nn.Dropout(p=0.8)(rna)
        if self.use_scna:
            scna = kwargs['scna']
            scna = torch.nn.Dropout(p=0.8)(scna)
        if self.use_methy:
            methy = kwargs['methy']
            methy = torch.nn.Dropout(p=0.8)(methy)
            
        cat = torch.tensor([]).to(self.device)
        
        if self.self_att:
            if self.use_rna:
                rna = self.rna_att(rna)
                rna = torch.nn.Dropout(p=0.5)(rna)
                cat = torch.cat((cat, rna), dim=1)
            if self.use_scna:
                scna = self.scna_att(scna)
                scna = torch.nn.Dropout(p=0.5)(scna)
                cat = torch.cat((cat, scna), dim=1)
            if self.use_methy:
                methy = self.methyl_att(methy)
                methy = torch.nn.Dropout(p=0.5)(methy)
                cat = torch.cat((cat, methy), dim=1)
        else:
            if self.use_rna:
                cat = torch.cat((cat, rna), dim=1)
            if self.use_scna:
                cat = torch.cat((cat, scna), dim=1)
            if self.use_methy:
                cat = torch.cat((cat, methy), dim=1)

        if self.cross_att:
            cat = self.cross_att(cat)
            cat = torch.nn.Dropout(p=0.5)(cat)
            
        cat = torch.cat((cat, event), dim=1)
        out = self.dense1(cat)
        out = torch.nn.Dropout(p=0.5)(out)
        out = F.relu(out)
        out = self.dense2(out)
        out = F.sigmoid(out)
        return out

class PathwayModuleAtt(nn.Module):
    def __init__(self, hidden_dim, device):
        super(PathwayModuleAtt, self).__init__()
        self.device = device
        self.self_att = SelfAttResNet(2580)
        self.dense1 = nn.Linear(2580, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        assert(x.shape[1] == 2580)
        out = self.self_att(x)
        out = self.dense1(x)
        out = torch.nn.Dropout(p=0.5)(out)
        out = F.relu(out)
        out = self.dense2(out)
        out = F.sigmoid(out)
        return out