"""
Runs a model on a single node across N-gpus.
"""
import os
import sys
from argparse import ArgumentParser
import multiprocessing as mp

import logging
import numpy as np
from pytorch_lightning import callbacks
import torch
import torch.utils.data 

from models import SpeechClassificationModel
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging

from data import WavDataset, SortedSampler, DynamicBatchSampler

seed_everything(234)

def main(args):
    """
    Main training routine specific for this project
    :param hparams:
    """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------

    if args.train_datadir is not None and args.dev_datadir is not None:
        train_dataset = WavDataset(args.train_datadir, extract_chunks=True, label_file=args.utt2class, 
                **vars(args))
        dev_dataset = WavDataset(args.dev_datadir, extract_chunks=False, label_file=args.utt2class,  
                label2id=train_dataset.label2id,
                **vars(args), no_augment=True)

        batch_size = args.batch_size
        test_batch_size = args.test_batch_size

        if args.use_balanced_sampler:
            from torchsampler import ImbalancedDatasetSampler
            train_sampler = ImbalancedDatasetSampler(dataset=train_dataset, labels=list(train_dataset.utt2label.values()))
        else:
            train_sampler = None

        train_loader = torch.utils.data.DataLoader(
                dataset=train_dataset,
                batch_size=batch_size,
                shuffle=False if train_sampler else True,
                collate_fn=train_dataset.collater,
                num_workers=8,
                sampler=train_sampler,
                drop_last=True,
                pin_memory=True)
        if test_batch_size == 1:
            sampler = None
        else:
            sampler=SortedSampler(dev_dataset, sort_key="dur")
        dev_loader = torch.utils.data.DataLoader(            
                dataset=dev_dataset,
                batch_size=test_batch_size,                
                shuffle=False,
                collate_fn=dev_dataset.collater,
                sampler=sampler,
                num_workers=4)

        if (args.load_checkpoint):
            breakpoint()
            model = SpeechClassificationModel.load_from_checkpoint(args.load_checkpoint)
        else:
            model = SpeechClassificationModel(num_outputs=train_dataset.num_labels, **vars(args))

        checkpoint_callback = ModelCheckpoint(
                save_top_k=4,
                save_last=True,
                verbose=True,
                monitor='val_loss',
                mode='min'                
        )    

        callbacks=[checkpoint_callback]

        if args.swa_epoch_start < 1.0:
            swa_callback = StochasticWeightAveraging(swa_epoch_start=args.swa_epoch_start)
            callbacks.append(swa_callback)


        trainer = Trainer.from_argparse_args(args, callbacks=callbacks)    
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=dev_loader)
    elif args.test_datadir is not None and args.load_checkpoint is not None:
        model = SpeechClassificationModel.load_from_checkpoint(args.load_checkpoint)
        print("Loaded model")
        model.dump_xvectors_dir = args.dump_xvectors_dir
        model.dump_predictions = args.dump_predictions
        model.xvector_layer_index = args.xvector_layer_index

        test_dataset = WavDataset(args.test_datadir, extract_chunks=False, label_file=args.utt2class,  
                label2id=None,
                **vars(args), no_augment=True)

        batch_size = args.test_batch_size
        test_loader = torch.utils.data.DataLoader(
                dataset=test_dataset,
                collate_fn=test_dataset.collater,
                #batch_size=batch_size,
                batch_sampler=DynamicBatchSampler("dur", SortedSampler(test_dataset, sort_key="dur"), args.max_test_batch_dur, test_dataset),
                num_workers=8)

        model.eval()
        args.logger = False
        trainer = Trainer.from_argparse_args(args)
        trainer.test(model, dataloaders=test_loader)        
    else:
        raise Exception("Either --train-datadir and --dev-datadir or --test-datadir and --load-checkpoint should be specified")

        


if __name__ == '__main__':
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    numba_logger = logging.getLogger('numba')
    numba_logger.setLevel(logging.WARNING)

    #mp.set_start_method('fork')

    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)
    

    # each LightningModule defines arguments relevant to it
    parser = SpeechClassificationModel.add_model_specific_args(parent_parser, root_dir)
    parser = WavDataset.add_data_specific_args(parser, root_dir)
    parser = Trainer.add_argparse_args(parser)

    # data
    parser.add_argument('--train-datadir', required=False, type=str)       
    parser.add_argument('--dev-datadir', required=False, type=str)
    parser.add_argument('--max-test-batch-dur', required=False, type=int, default=500)
    
    parser.add_argument('--load-checkpoint', required=False, type=str)        
    parser.add_argument('--test-datadir', required=False, type=str)        
    parser.add_argument('--utt2class', default="utt2lang", type=str)

    parser.add_argument('--swa-epoch-start', default=1.0, type=float)

    parser.add_argument('--use-balanced-sampler', default=False, action='store_true')

    args = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(args)