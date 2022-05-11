import torch
import torch.nn as nn
import torch.nn.functional as F

class Contrast_loss_point_cloud(nn.Module):
        def __init__(self, temperature=0.07):
            super(Contrast_loss_point_cloud, self).__init__()
            self.temp = temperature
        def forward(self, features, labels_all=None):
            all_loss = []
            for features_map,labels in zip(features,labels_all):
                labels = labels.unsqueeze(0)
                normalize_vectors = F.normalize(features_map.T,dim = 1)
                dot_products = torch.matmul(normalize_vectors, normalize_vectors.T)
                mask = torch.eq(labels, labels.T).float()
                mask_not = torch.logical_not(mask)
                #
                negetives = (mask_not * dot_products).sum(1) 
#                 negetives = torch.div(negetives,temperature)
                #
                posetives = (mask * dot_products).sum(1) / mask.sum(1)
#                 posetives = torch.div(posetives,temperature)
                diviation = - torch.log(posetives / (posetives + negetives))
                loss = torch.mean(diviation)
                all_loss.append(loss)
            print(all_loss)
            all_loss = torch.stack(all_loss)
            return torch.mean(all_loss)