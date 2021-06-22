from argparse import ArgumentParser
import os
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from HIT_convttlstm import convttlstm
from HIT_dataset3d import hit_dm

def main(hparams):
    checkpoint_callback = ModelCheckpoint(
    dirpath=os.getcwd(),
    save_top_k=True,
    verbose=True,
    monitor='val_loss',
    mode='min',
    save_last=True)
    model = convttlstm(hparams)
    dm = hit_dm(hparams)
    trainer = Trainer.from_argparse_args(hparams,callbacks=checkpoint_callback,
                         auto_select_gpus = True,
                         progress_bar_refresh_rate=1)   
    trainer.fit(model, datamodule = dm)

if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)
    parser = convttlstm.add_model_specific_args(parser)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)