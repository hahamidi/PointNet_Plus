

# {'gpus': [0], 'optimizer': {'weight_decay': 0.0, 'lr': 0.001, 'lr_decay': 0.5, 'bn_momentum': 0.5, 'bnm_decay': 0.5,
#  'decay_step': 300000.0}, 'task_model': {'class': 'model_ssg.PointNet2SemSegSSG', 'name': 'sem-ssg'},
#   'model': {'use_xyz': True}, 'distrib_backend': 'dp', 'num_points': 4096, 'epochs': 50, 'batch_size': 24}



import os
import sys

pointnet2_dir = os.path.split(os.path.abspath(__file__))[0]
main_dir = "/".join(pointnet2_dir.split("/")[0:-1])
pointnet2_ops_lib_dir = main_dir+"/pointnet2_ops_lib/" 

sys.path.insert(0,main_dir)
sys.path.insert(0,pointnet2_ops_lib_dir)

import hydra
import omegaconf
import torch
import numpy as np
import time
from tqdm import tqdm, trange
from data.Indoor3DSemSegLoader import fakeIndoor3DSemSeg,Indoor3DSemSeg
from torch.utils.data import DataLoader
from losses import Contrast_loss_point_cloud
from openTSNE import TSNE
import matplotlib.pyplot as plt


# from torch.utils.tensorboard import SummaryWriter
# writer = SummaryWriter()

# from pytorch_lightning.loggers import TensorBoardLogger
# from surgeon_pytorch import Inspect,get_layers

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def hydra_params_to_dotdict(hparams):
    def _to_dot_dict(cfg):
        res = {}
        for k, v in cfg.items():
            if isinstance(v, omegaconf.DictConfig):
                res.update(
                    {k + "." + subk: subv for subk, subv in _to_dot_dict(v).items()}
                )
            elif isinstance(v, (str, int, float, bool)):
                res[k] = v

        return res

    return _to_dot_dict(hparams)



def show_embeddings(tsne_embs_i, lbls, highlight_lbls=None, imsize=8, cmap=plt.cm.tab20):
    tsne = TSNE(metric='cosine', n_jobs=-1)
    tsne_embs = tsne.fit(tsne_embs_i)
    fig,ax = plt.subplots(figsize=(imsize,imsize))
    colors = cmap(np.array(lbls))
    ax.scatter(tsne_embs[:,0], tsne_embs[:,1], c=colors, cmap=cmap, alpha=1 if highlight_lbls is None else 0.1)
    fig.savefig('to.png') 
    print(ax)


class Trainer:
    def __init__(self,
                 model: torch.nn.Module,
                 device: torch.device,
                 criterion: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 training_DataLoader: torch.utils.data.Dataset,
                 validation_DataLoader: torch.utils.data.Dataset = None,
                 lr_scheduler: torch.optim.lr_scheduler = None,
                 epochs: int = 100,
                 epoch: int = 0,
                 notebook: bool = False,
                 save_best_model : int = 1,
                 load_checkpoint : bool = False
                 ):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.training_DataLoader = training_DataLoader
        self.validation_DataLoader = validation_DataLoader
        self.device = device
        self.epochs = epochs
        self.epoch = epoch
        self.notebook = notebook
        self.save_best_model = save_best_model
        self.load_checkpoint = load_checkpoint
        
    
        self.training_loss = [0]
        self.validation_loss = [0]
        self.learning_rate = [0]
        self.last_model = pointnet2_dir + "/checkpoints/27_001.pth.tar"
        self.validation_acc = [0]
        self.training_acc = [0]

    def save_checkpoint(self,state,filename = "chechpoint.pth.tar"):
        print("**************saving model****************")
        print(pointnet2_dir)
        filename =pointnet2_dir+"/checkpoints/"+filename
        torch.save(state,filename)
        self.last_model = filename


    def load_from_checkpoint(self , checkpoint = "" ):
        print("++++++++++++++loading_model++++++++++++++++")
        if checkpoint == "":
            checkpoint =  torch.load(self.last_model)
        self.model.load_state_dict(checkpoint["state_dict"])
        # self.optimizer = 
        



    def run_trainer(self):


        if self.notebook:
            from tqdm.notebook import tqdm, trange
        else:
            from tqdm import tqdm, trange

        progressbar = trange(self.epochs, desc='Progress')
        # writer = SummaryWriter("loss_lr_logs")
        if self.load_checkpoint == True:
            self.load_from_checkpoint(torch.load(self.last_model))


        for i in progressbar:
            """Epoch counter"""
            self.epoch += 1  # epoch counter

            """Training block"""
            self._train()


            """Validation block"""
            if self.validation_DataLoader is not None:
                self._validate()
            

            """Learning rate scheduler block"""
            if self.lr_scheduler is not None:
                if self.validation_DataLoader is not None and self.lr_scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                    self.lr_scheduler.batch(self.validation_loss[i])  # learning rate scheduler step with validation loss
                else:
                    self.lr_scheduler.batch()  # learning rate scheduler step
            logs = {"train_loss":self.training_loss[-1],"val_loss":self.validation_loss[-1],"lr":self.learning_rate[-1]}
            logs_acc = {"training_acc":self.training_acc[-1],"val_acc":self.validation_acc[-1]}
            # writer.add_scalars("train/loss",logs, self.epoch)
            print("---------------------------------------------------------------------------------")
            print("epoch_num:",i,"\n")
            print("=>",logs,"\n","=>",logs_acc)
            print("---------------------------------------------------------------------------------")
        
            
            if self.epoch % 5 == 0:

                state = {'epoch': self.epoch,
                                'state_dict': self.model.state_dict(),
                                'optimizer': self.optimizer.state_dict()}
                self.save_checkpoint(state,filename= f"acc: {self.validation_acc[-1]:.4f} chechpoint.pth.tar")

            
        # writer.close()
        return self.training_loss, self.validation_loss, self.learning_rate

    def _train(self):



        self.model.train()  # train mode
        train_losses = []  # accumulate the losses here
        train_acc = []
        batch_iter = self.training_DataLoader

        for (x, y) in batch_iter:


            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)
            self.optimizer.zero_grad()  # zerograd the parameters
            out = self.model(input)  # one forward pass
            # print(out.size())
            loss = self.criterion(out, target)  # calculate loss
            
            loss_value = loss.item()
            # print(loss_value)
            train_losses.append(loss_value)
            loss.backward()  # one backward pass
            self.optimizer.step()  # update the parameters
            # with torch.no_grad():
            #     acc = (torch.argmax(out, dim=1) == target).float().mean()
            # train_acc.append(acc.item())
            break


            # print(f'Training: (loss {loss_value:.4f})') 
        # print(out[0].size())
        # print(target[0].size())
        # print(out[0])
        # print(target[0])
        print(type((out[0].T).cpu().detach().numpy()))
        print(target[0].cpu().detach().numpy())
        show_embeddings((out[0].T).cpu().detach().numpy(),target[0].cpu().detach().numpy())
        self.training_loss.append(np.mean(train_losses))
        # self.training_acc.append(np.mean(train_acc))
        self.learning_rate.append(self.optimizer.param_groups[0]['lr'])
        # batch_iter.close()

    def _validate(self):


        self.model.eval()  # evaluation mode
        valid_losses = []  # accumulate the losses here
        valid_acc    = []
        batch_iter = self.validation_DataLoader

        for (x, y) in batch_iter:
            input, target = x.to(self.device), y.to(self.device)  # send to device (GPU or CPU)

            with torch.no_grad():
                out = self.model(input)
                loss = self.criterion(out, target)
                loss_value = loss.item()
                acc = (torch.argmax(out, dim=1) == target).float().mean()
                valid_losses.append(loss_value)
                # valid_acc.append(acc.item())
                # print(f'Validation: (loss {loss_value:.4f})')

        self.validation_loss.append(np.mean(valid_losses))
        # self.validation_acc.append(np.mean(valid_acc))

        # batch_iter.close()



@hydra.main("config/config.yaml")
def main(cfg):
    hypers = hydra_params_to_dotdict(cfg)
    print(cfg)
    # print(model)

    print("=====>",hypers["optimizer.lr"])

    data_set_train = Indoor3DSemSeg(num_points=4096,train=True,test_area=[2,3,4,5,6])
    # data_set_test  = Indoor3DSemSeg(num_points=4096,train=False,test_area=[5])

    data_set_eval  = Indoor3DSemSeg(num_points=4096,train=False,test_area=[6])




    # data_set_train = fakeIndoor3DSemSeg()
    # data_set_eval  = fakeIndoor3DSemSeg()



    data_loader_train =  DataLoader(data_set_train, batch_size=24, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=2)
    # data_loader_test  = DataLoader(data_set_test, batch_size=24, shuffle=False, sampler=None,
    #        batch_sampler=None, num_workers=2)
    data_loader_eval  = DataLoader(data_set_eval, batch_size=24, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=2)

   
    if torch.cuda.is_available():
         device = torch.device('cuda')
        
    else:
         torch.device('cpu')

    
    model = hydra.utils.instantiate(cfg.task_model,hypers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hypers["optimizer.lr"]) #weight_decay= hypers["optimizer.lr_decay"]

    criterion =  Contrast_loss_point_cloud()


    # optimizer
    

    # trainer
 
    trainer = Trainer(model=model,
                    device=device,
                    criterion=criterion,
                    optimizer=optimizer,
                    training_DataLoader=data_loader_train,
                    validation_DataLoader=data_loader_eval,
                    lr_scheduler=None,
                    epochs=hypers["epochs"],
                    epoch=0,
                    notebook=True)


    # start training

    training_losses, validation_losses, lr_rates = trainer.run_trainer()
    #test







if __name__ == "__main__":
    main()
