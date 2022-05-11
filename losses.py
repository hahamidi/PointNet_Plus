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
                # print(features_map.size())
                # print(labels.size())
                normalize_vectors = F.normalize(features_map.T,dim = 1)
                dot_products = torch.matmul(normalize_vectors, normalize_vectors.T)
                mask = torch.eq(labels, labels.T).float()
                mask_not = torch.logical_not(mask)
                print("dot_products",dot_products)
                print("----------------------------------------------------------")
                print("mask_not",mask_not)
                print("----------------------------------------------------------")
                print("mask",mask)
                print("----------------------------------------------------------")
                #
                negetives = (mask_not * dot_products).sum(1) 
#                 negetives = torch.div(negetives,temperature)
                #
                posetives = (mask * dot_products).sum(1) / mask.sum(1)
                # print(negetives,posetives)
#                 posetives = torch.div(posetives,temperature)
                print("p:",posetives)
                print("n:",negetives)
                print("----------------------------------------------------------")
                diviation = posetives / (posetives + negetives)
                print("=>",diviation)
                diviation = - torch.log(diviation)
                print("*=>",diviation)
                loss = torch.mean(diviation)
                all_loss.append(loss)
            # print(all_loss)
            all_loss = torch.stack(all_loss)
            return torch.mean(all_loss)