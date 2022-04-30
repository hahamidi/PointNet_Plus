import os
import sys

pointnet2_dir = os.path.split(os.path.abspath(__file__))[0]
main_dir = "/".join(pointnet2_dir.split("/")[0:-1])
pointnet2_ops_lib_dir = main_dir+"/pointnet2_ops_lib/" 

sys.path.insert(0,main_dir)
sys.path.insert(0,pointnet2_ops_lib_dir)

import hydra
import omegaconf
# import pytorch_lightning as pl
import torch
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


@hydra.main("config/config.yaml")
def main(cfg):
    model = hydra.utils.instantiate(cfg.task_model, hydra_params_to_dotdict(cfg))
    print(cfg)
    print(model)
    
    # early_stop_callback = pl.callbacks.EarlyStopping(patience=5)
    # checkpoint_callback = pl.callbacks.ModelCheckpoint(
    #     monitor="val_acc",
    #     mode="max",
    #     save_top_k=2,
    #     filepath=os.path.join(
    #         cfg.task_model.name, "{epoch}-{val_loss:.2f}-{val_acc:.3f}"
    #     ),
    #     verbose=True,
    # )
    # trainer = pl.Trainer(
    #     gpus=list(cfg.gpus),
    #     max_epochs=cfg.epochs,
    #     early_stop_callback=early_stop_callback,
    #     checkpoint_callback=checkpoint_callback,
    #     distributed_backend=cfg.distrib_backend
    # )

    # print(get_layers(model))
    # # trainer.fit(model)
    # # trainer.test(model)
    


if __name__ == "__main__":
    main()
