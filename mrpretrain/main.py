import copy
import os
import resource
import torch

import pytorch_lightning as pl

from umd.config import ex
from umd.datamodules.multitask_datamodule import MTDataModule
from umd.modules import UMDTransformerSS
from pytorch_lightning import Callback
from copy import deepcopy
import torch
import torch.nn as nn

from pytorch_lightning.utilities import rank_zero_only
from timm.utils.model import get_state_dict, unwrap_model
# from timm.utils.model_ema import ModelEmaV2
from umd.modules.modelemav2 import ModelEmaV2

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))


# export
class EMACallback(Callback):
    """
    Model Exponential Moving Average. Empirically it has been found that using the moving average
    of the trained parameters of a deep network is better than using its trained parameters directly.

    If `use_ema_weights`, then the ema parameters of the network is set after training end.
    use self.ema.module to call the ema module.(self.module = deepcopy(model))
    """

    def __init__(self, configs, decay=0.995, use_ema_weights: bool = True):
        self.decay = decay
        self.configs = configs

    def on_fit_start(self, trainer, pl_module):
        "Initialize `ModelEmaV2` from timm to keep a copy of the moving average of the weights"
        pl_module.ema = ModelEmaV2(configs=self.configs, device=pl_module.device, decay=self.decay)
        for param in pl_module.ema.parameters():
            param.detach_()

    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        "Update the stored parameters using a moving average"
        # Update currently maintained parameters.
        pl_module.ema.update(pl_module)

    '''
    def on_validation_epoch_start(self, trainer, pl_module):
        "do validation using the stored parameters"
        # save original parameters before replacing with EMA version
        self.store(pl_module.parameters())

        # update the LightningModule with the EMA weights
        # ~ Copy EMA parameters to LightningModule
        self.copy_to(self.ema.module.parameters(), pl_module.parameters())

    def on_validation_end(self, trainer, pl_module):
        "Restore original parameters to resume training later"
        self.restore(pl_module.parameters())

    def on_train_end(self, trainer, pl_module):
        # update the LightningModule with the EMA weights
        if self.use_ema_weights:
            self.copy_to(self.ema.module.parameters(), pl_module.parameters())
            # msg = "Model weights replaced with the EMA version."
            # log_main_process(_logger, logging.INFO, msg)
            print('msg: Model weights replaced with the EMA version.')

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.ema is not None:
            return {"state_dict_ema": get_state_dict(self.ema, unwrap_model)}

    def on_load_checkpoint(self, callback_state):
        if self.ema is not None:
            self.ema.module.load_state_dict(callback_state["state_dict_ema"])

    def store(self, parameters):
        "Save the current parameters for restoring later."
        self.collected_params = [param.clone() for param in parameters]

    def restore(self, parameters):
        """
        Restore the parameters stored with the `store` method.
        Useful to validate the model with EMA parameters without affecting the
        original optimization process.
        """
        for c_param, param in zip(self.collected_params, parameters):
            param.data.copy_(c_param.data)

    def copy_to(self, shadow_parameters, parameters):
        "Copy current parameters into given collection of parameters."
        for s_param, param in zip(shadow_parameters, parameters):
            if param.requires_grad:
                param.data.copy_(s_param.data)
    '''



@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    pl.seed_everything(_config["seed"])

    # Data modules
    dm = MTDataModule(_config, dist=True)

    # Module
    model = UMDTransformerSS(_config)

    # Loggers
    os.makedirs(_config["log_dir"], exist_ok=True)
    exp_name = f'{_config["exp_name"]}'
    run_name = f'{exp_name}-seed{_config["seed"]}-from_{_config["load_path"].replace("/", "_")}'
    tb_logger = pl.loggers.TensorBoardLogger(_config["log_dir"], name=run_name)
    wb_logger = pl.loggers.WandbLogger(project="UMD", name=run_name, offline = True)
    loggers = [tb_logger, wb_logger]
    # loggers = [tb_logger]

    # Callback
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        verbose=True,
        monitor="val/the_metric",
        mode="max",
        save_last=False,
        save_weights_only=True if "finetune" in exp_name else False
    )
    # ema   
    # ema_update_callback = EmaUpdate()
    ema_callback = EMACallback(_config)
    # learning rate
    lr_callback = pl.callbacks.LearningRateMonitor(logging_interval="step")

    # callbacks_noema = [checkpoint_callback, lr_callback]
    callbacks_addema = [checkpoint_callback, lr_callback, ema_callback]
    # callbacks = callbacks_addema if "pretrain" in exp_name else callbacks_noema
    callbacks = callbacks_addema

    # Training Hyper-Parameters
    num_gpus = (_config["num_gpus"] if isinstance(_config["num_gpus"], int) else len(_config["num_gpus"]))
    grad_steps = max(_config["batch_size"] // (_config["per_gpu_batchsize"] * num_gpus * _config["num_nodes"]), 1)
    max_steps = _config["max_steps"] if _config["max_steps"] is not None else None
    max_epochs = _config["max_epoch"] if max_steps is None else 1000

    # Trainer
    trainer = pl.Trainer(
        gpus=num_gpus,
        num_nodes=_config["num_nodes"],
        precision=_config["precision"],
        accelerator="ddp",
        benchmark=True,
        deterministic=True,
        max_epochs=max_epochs,
        max_steps=max_steps,
        callbacks=callbacks,
        logger=loggers,
        prepare_data_per_node=False,
        replace_sampler_ddp=False,
        accumulate_grad_batches=grad_steps,
        log_every_n_steps=10,
        flush_logs_every_n_steps=10,
        resume_from_checkpoint=_config["resume_from"],
        weights_summary="top",
        fast_dev_run=_config["fast_dev_run"],
        val_check_interval=_config["val_check_interval"],
        default_root_dir=_config["default_root_dir"]
    )

    if not _config["test_only"]:
        trainer.fit(model, datamodule=dm)
        # print('prepared model down')
        if "finetune" in exp_name:
            trainer.test(ckpt_path="best", datamodule=dm)
            # print('prepared model down2')
    else:
        trainer.test(model, datamodule=dm)
