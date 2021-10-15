import os
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from argparse import ArgumentParser
from collections import OrderedDict
import functools
import operator
import random
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

import torchaudio
import torchaudio.transforms


import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule

from torchmetrics import Accuracy

from pooling import *
from kaldiio import WriteHelper


EPSILON = torch.tensor(torch.finfo(torch.float).eps)



class SpeechClassificationModel(LightningModule):
    """
    Sample model to show how to define a template
    """
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        # For dumping x-vectors
        self.write_helper = None

        # if you specify an example input, the summary will show input/output for each layer
        #self.example_input_array = torch.rand((self.batch_size, self.feat_dim, 300))

        # build model
        self.__build_model()

    # ---------------------
    # MODEL SETUP
    # ---------------------
    def __build_model(self):
        self.espnet2_model = None
        if self.hparams.espnet2_model != "":
            from espnet2.tasks.asr import ASRTask
            
            model_json_file = f"{os.path.dirname(self.hparams.espnet2_model)}/config.yaml"
            logging.info(f"Loading Espnet2 model from {self.hparams.espnet2_model}")
            self.espnet2_model, asr_train_args = \
                ASRTask.build_model_from_file(model_json_file, self.hparams.espnet2_model)
                
            #self.espnet2_model.eval()
            #for param in self.espnet2_model.parameters():
            #    param.requires_grad = False                
            self.encoder_output_dim = self.espnet2_model.encoder.output_size()
            
        elif self.hparams.wav2vec2_model != "":
            from huggingface_wav2vec import HuggingFaceWav2Vec2
            self.wav2vec2 = HuggingFaceWav2Vec2(
                source=self.hparams.wav2vec2_model,
                output_norm=True,
                freeze=False,
                freeze_feature_extractor=True,
                save_path=".huggingface")

            wavs = torch.randn((1,16000), dtype=torch.float)
            reps = self.wav2vec2(wavs)            
            self.encoder_output_dim = reps.shape[2]

        else:
            raise Exception("not implemented")    

        if "pooling" not in self.hparams:
            self.hparams.pooling = "stats" # backward compability
            
        pooling_map = {"stats": StatisticsPooling, 
                       "attentive-stats": AttentiveStatisticsPooling,
                       "lde" : LDEPooling, 
                       "mha": MultiHeadAttentionPooling, 
                       "global-mha": GlobalMultiHeadAttentionPooling,
                       "multires-mha": MultiResolutionMultiHeadAttentionPooling}
                      
        self.pooling = pooling_map[self.hparams.pooling](input_dim=self.encoder_output_dim)
        
        post_pooling_layers = []

        post_pooling_layers.append(nn.Linear(self.pooling.get_output_dim(), self.hparams.hidden_dim))
        post_pooling_layers.append(nn.BatchNorm1d(self.hparams.hidden_dim))
        post_pooling_layers.append(nn.ReLU(inplace=True))

        post_pooling_layers.append(nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim))
        post_pooling_layers.append(nn.BatchNorm1d(self.hparams.hidden_dim))
        post_pooling_layers.append(nn.ReLU(inplace=True))

        linear = nn.Linear(self.hparams.hidden_dim, self.hparams.num_outputs)
        linear.bias.data.fill_(0.0)
        linear.weight.data.fill_(0.0)
        post_pooling_layers.append(linear)
        post_pooling_layers.append(nn.LogSoftmax(dim=1))

        self.post_pooling_layers = nn.Sequential(*post_pooling_layers)

        #print(self.model)
        #from torchsummary import summary
        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()
        


    # ---------------------
    # TRAINING
    # ---------------------
    def forward(self, wavs, wav_lens):
        pooling_output = self._forward_until_pooling(wavs, wav_lens)
        
        return self.post_pooling_layers(pooling_output)

    def _forward_until_before_pooling(self, wavs, wav_lens):
        if self.espnet2_model is not None:
            batch = {"speech": wavs, "speech_lengths": wav_lens}
            e, e2 = self.espnet2_model.encode(**batch)
            return e.permute(0, 2, 1)
        elif self.wav2vec2 is not None:            
            reps = self.wav2vec2(wavs)
            #breakpoint()
            return reps.permute(0, 2, 1)
            
        else:
            raise Exception("not implemented")

    def _forward_until_pooling(self, wavs, wav_lens):
        pre_pooling_output = self._forward_until_before_pooling(wavs, wav_lens)
        return self.pooling(pre_pooling_output).squeeze(2)

    def loss(self, logits, labels, smooth_eps=0.0):
        if smooth_eps > EPSILON:
            result = nll_loss_with_label_smoothing(logits, labels, smooth_eps)
        else:
            result = F.nll_loss(logits, labels)
        return result

    def extract_xvectors(self, wavs, wav_lens):
        pooling_output = self._forward_until_pooling(wavs, wav_lens)
        return self.post_pooling_layers[0](pooling_output)

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer: Optimizer,
        optimizer_idx: int,
        second_order_closure: Optional[Callable] = None,
        on_tpu: bool = False,
        using_native_amp: bool = False,
        using_lbfgs: bool = False,
    ) -> None:
        
        if self.trainer.global_step < self.hparams.lr_warmup_batches:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.hparams.lr_warmup_batches)
            #breakpoint()
            if lr_scale != 1.0:
                for i, pg in enumerate(optimizer.param_groups):
                    pg['lr'] = lr_scale * self.hparams.learning_rate

        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_idx, second_order_closure, on_tpu, using_native_amp, using_lbfgs)


    def training_step(self, batch, batch_idx):
        """
        Lightning calls this inside the training loop
        :param batch:
        :return:
        """
        # forward pass
        wav = batch["wav"]
        dur = batch["dur"]
        y = batch["label_id"]
        y_hat = self.forward(wav, dur)
        loss_val = self.loss(y_hat, y, smooth_eps=self.hparams.label_smoothing)


        lr  = torch.tensor(self.trainer.optimizers[0].param_groups[-1]['lr'], device=loss_val.device)
        self.log('train_loss', loss_val, prog_bar=True)
        self.log('lr', lr, prog_bar=True)
        self.log("train_acc", self.train_acc(y_hat, y.int()), prog_bar=True)
        
        return loss_val

    def validation_step(self, batch, batch_idx):
        """
        Lightning calls this inside the validation loop
        :param batch:
        :return:
        """
        # Normal validation
        wav = batch["wav"]
        dur = batch["dur"]
        y = batch["label_id"]
        y_hat = self.forward(wav, dur)
        loss_val = self.loss(y_hat, y, smooth_eps=0)
        # acc
        self.log("val_acc", self.valid_acc(y_hat, y.int()), prog_bar=True)
        self.log('val_loss', loss_val, prog_bar=True)


    def test_step(self, batch, batch_idx):
        wav = batch["wav"]
        dur = batch["dur"]
        utt_index = batch["index"]
        
        #breakpoint()
        if self.dump_xvectors_dir is not None:
            xvectors = self.extract_xvectors(wav, dur)
        
            if self.write_helper is None and self.trainer.global_rank == 0:            
                with open(f"{self.dump_xvectors_dir}/architecture.txt", "w") as f:
                    print(self, file=f)

                self.write_helper = WriteHelper(f'ark,scp:{self.dump_xvectors_dir}/xvector.ark,{self.dump_xvectors_dir}/xvector.scp')

            
            for i in range(len(xvectors)):
                utt = self.test_dataloader.dataloader.dataset.utts[utt_index[i]]
                self.write_helper(utt, xvectors[i].cpu().numpy())
            output = OrderedDict({})
        elif self.dump_predictions is not None:
            if self.write_helper is None and self.trainer.global_rank == 0:            
                self.write_helper = WriteHelper(f'ark,scp:{self.dump_predictions}.ark,{self.dump_predictions}.scp')

            y_hat = self.forward(wav, dur)
            #breakpoint()
            for i in range(len(y_hat)):
                utt = self.test_dataloader.dataloader.dataset.utts[utt_index[i]]
                self.write_helper(utt, y_hat[i].cpu().numpy())
            output = OrderedDict({})

        

        # can also return just a scalar instead of a dict (return loss_val)
        return output

    def test_epoch_end(self, outputs):
        logging.info("Closing write helper...")
        if self.write_helper:
            self.write_helper.close()
        return {}


    # ---------------------
    # TRAINING SETUP
    # ---------------------
    def configure_optimizers(self):

        params = list(self.named_parameters())

        def is_backbone(n): return (('wav2vec2' in n) or ('espnet2' in n))

        grouped_parameters = [
            {"params": [p for n, p in params if is_backbone(n)], 'lr': self.hparams.learning_rate * 0.01},
            {"params": [p for n, p in params if not is_backbone(n)], 'lr': self.hparams.learning_rate},
        ]

        if self.hparams.optimizer_name == "adam":
            optimizer = optim.Adam(grouped_parameters, lr=self.hparams.learning_rate)
        elif self.hparams.optimizer_name == "adamw":
            optimizer = optim.AdamW(grouped_parameters, lr=self.hparams.learning_rate)
        elif self.hparams.optimizer_name == "sgd":
            optimizer = optim.SGD(grouped_parameters, lr=self.hparams.learning_rate, momentum=0.5, weight_decay=1e-5)
        elif self.hparams.optimizer_name == "fusedlamb":
            optimizer =  apex.optimizers.FusedLAMB(grouped_parameters, lr=self.hparams.learning_rate)
        else:
            raise NotImplementedError()


        #scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.97)
        return [optimizer], [scheduler]


    @staticmethod
    def add_model_specific_args(parent_parser, root_dir):  # pragma: no cover
        """
        Parameters you define here will be available to your model through self
        :param parent_parser:
        :param root_dir:
        :return:
        """
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # param overwrites
        # parser.set_defaults(gradient_clip_val=5.0)

        parser.add_argument('--batch-size', default=64, type=int)
        parser.add_argument('--test-batch-size', default=32, type=int)
        parser.add_argument('--hidden-dim', default=500, type=int)
        parser.add_argument('--espnet2-model', default="", type=str)
        parser.add_argument('--wav2vec2-model', default="", type=str)
        parser.add_argument('--pooling', default="stats", choices=['stats', 'attentive-stats', 'lde', "mha", "global-mha", "multires-mha"],)
        # training params (opt)

        parser.add_argument('--optimizer-name', default='adamw', type=str)
        parser.add_argument('--learning-rate', default=0.0005, type=float)
        parser.add_argument('--lr-warmup-batches', default=0, type=int)

        parser.add_argument('--label-smoothing', default=0.0, type=float)

        parser.add_argument('--sample-rate', default=16000,  type=int)

        parser.add_argument('--dump-xvectors-dir', required=False, type=str)        
        parser.add_argument('--dump-predictions', required=False, type=str)        

        return parser
