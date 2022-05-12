import torch
import torch.nn as nn
import torch.nn.functional as F

class Contrast_loss_point_cloud(nn.Module):
        def __init__(self, temperature=0.07):
            super(Contrast_loss_point_cloud, self).__init__()
            self.temp = temperature
            self.device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))

        def forward(self, features, labels_all=None):
            all_loss = []
            for features_map,labels in zip(features,labels_all):

                labels = labels.unsqueeze(0)

                normalize_vectors = F.normalize(features_map.T,dim = 1)       
                dot_products = torch.matmul(normalize_vectors, normalize_vectors.T) 
                dot_products = torch.exp(dot_products)
                dot_products = torch.div(dot_products,self.temp)
                dot_products = dot_products - torch.diag(torch.diagonal(dot_products, 0))
                
                mask = torch.eq(labels, labels.T).float()
                mask_not = torch.logical_not(mask)

                
                posetives = (mask * dot_products).sum(1) / mask.sum(1)
                negetives = (mask_not * dot_products).sum(1) / mask_not.sum(1)
                print(posetives,negetives)
                

                diviation = posetives / (posetives + negetives)
                print(diviation)
                
                diviation = - torch.log(diviation)
                print(diviation)
                loss = torch.mean(diviation)
                print(loss)
                print("------------------------------------------")
                all_loss.append(loss)
            all_loss = torch.stack(all_loss)
            return torch.mean(all_loss)
        
if __name__ == '__main__':    
    device = (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    feature = torch.rand(2,128,4096).to(device)
    labels = torch.randint(0, 10, (3,4096)).to(device)
    loss_class = Contrast_loss_point_cloud().to(device)
    loss_class(feature,labels)