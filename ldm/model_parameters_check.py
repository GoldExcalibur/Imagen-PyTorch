import torch 
import torch.nn as nn 

## 确定模型训练中参数的异常值
class ModelParametersCheck:
    def __init__(self, model, T):
        self.model = model
        self.history = dict()
        self.T = T


    def forward(self):
        for (name,p) in self.model.named_parameters():
            if p.requires_grad == True:
                key_name = '{}_mean'.format(name)
                value = torch.mean(p).detach().cpu().numpy().tolist()
                
                if key_name in self.history.keys():
                    self.history[key_name] = self.history[key_name].append(value)
                    if len(self.history[key_name]) > self.T:
                        self.history[key_name] = self.history[key_name][-self.T:-1]
                else:
                    self.history[key_name] = [value]
                    #print(self.history[key_name])


                key_name = '{}_std'.format(name)
                value = torch.std(p).detach().cpu().numpy().tolist()
                if key_name in self.history.keys():
                    
                    self.history[key_name] = self.history[key_name].append(value)
                    if len(self.history[key_name]) > self.T:
                        self.history[key_name] = self.history[key_name][-self.T:-1]
                else:
                    self.history[key_name] = [value]

                key_name = '{}_min'.format(name)
                value = torch.min(p).detach().cpu().numpy().tolist()
                if key_name in self.history.keys():
                    self.history[key_name] = self.history[key_name].append(value)
                    if len(self.history[key_name]) > self.T:
                        self.history[key_name] = self.history[key_name][-self.T:-1]
                else:
                    self.history[key_name] = [value]

                key_name = '{}_max'.format(name)
                value = torch.max(p).detach().cpu().numpy().tolist()
                if key_name in self.history.keys():
                    self.history[key_name] = self.history[key_name].append(value)
                    if len(self.history[key_name]) > self.T:
                        self.history[key_name] = self.history[key_name][-self.T:-1]
                else:
                    self.history[key_name] = [value]

