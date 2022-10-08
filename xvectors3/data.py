import random
import torch
import kaldiio
import logging
import itertools
import numpy as np
import torchaudio
import logging
import sys
import struct
from argparse import ArgumentParser

from collections import OrderedDict
from functools import lru_cache
from tqdm import tqdm
import torchaudio.transforms
import torchaudio.functional

import audiomentations

class DynamicBatchSampler():

    def __init__(self, seq_len_key, sampler, max_total_len, dataset):
        self.seq_len_key = seq_len_key
        self.sampler = sampler
        self.max_total_len = max_total_len
        self.batches = []
        current_len = 0
        batch = []
        for idx in self.sampler:
            this_len = dataset[idx][seq_len_key]
            if current_len + this_len > max_total_len:
                if len(batch) > 0:
                    self.batches.append(batch)
                batch = [idx]
                current_len = this_len
            else:
                current_len += this_len
                batch.append(idx)            
        if len(batch) > 0:
           self.batches.append(batch)        
        

    def __iter__(self):
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)



class SortedSampler(torch.utils.data.Sampler):

    def __init__(self, dataset, sort_key):
        self.dataset = dataset
        keys = torch.tensor([i[sort_key] for i in dataset])
        self.sort_order = torch.argsort(keys, descending=True)

    def __iter__(self):
        return iter(self.sort_order)

    def __len__(self) -> int:
        return len(self.dataset)


class WavDataset(torch.utils.data.Dataset):
    def __init__(self, datadir, extract_chunks=True, min_chunk_length=2.0, max_chunk_length=4.0,
                 label_file="utt2lang", sample_rate=16000, label2id=None,
                 num_fbanks=40, noise_dir="", rir_dir="", short_noise_dir="", no_augment=False,
                 speed_perturbation_probability=0.0, spec_augment=False, num_augmix_copies=0,
                 **kwargs):
        self.extract_chunks = extract_chunks
        self.min_length = min_chunk_length
        self.max_length = max_chunk_length
        self.sample_rate = sample_rate
        self.num_fbanks = num_fbanks
        self.utt2label = {}
        self.speed_perturbation_probability = speed_perturbation_probability
        self.no_augment = no_augment
        self.spec_augment = spec_augment
        self.num_augmix_copies = num_augmix_copies

        for l in open(f"{datadir}/{label_file}"):
            ss = l.split()
            self.utt2label[ss[0]] = ss[1]

        self.labels = list(sorted(set(self.utt2label.values())))
        if label2id is None:
            self.label2id = {label: i for i, label in enumerate(self.labels)}
        else:
            self.label2id = label2id
            for label in self.labels:
                assert label in label2id

        logging.info(f"Reading wav locations from {datadir}/wav.scp")
        self.utt2file = {}
        self.utts = []
        self.utt2index = {}
        for line in open(f"{datadir}/wav.scp"):
            wav_id, location = line.split()
            if wav_id in self.utt2label:
                self.utt2file[wav_id] = location 
                self.utt2index[wav_id] = len(self.utts)
                self.utts.append(wav_id)

        self.num_labels = len(self.labels)
        self.utt2dur = {}
        for l in open(f"{datadir}/utt2dur"):
            ss = l.split()
            if ss[0] in self.utt2label:
                self.utt2dur[ss[0]] = float(ss[1])
        self.total_dur = sum(self.utt2dur.values())

        self.augment = None
        self.transforms = []
        if not no_augment:
            augmentations = []
            if rir_dir != "":
                augmentations.append(audiomentations.ApplyImpulseResponse(ir_path=rir_dir, p=0.9, lru_cache_size=1024, leave_length_unchanged=True))
            if noise_dir != "":
                augmentations.append(audiomentations.AddBackgroundNoise(sounds_path=noise_dir, p=0.5, lru_cache_size=1024))
            augmentations.append(audiomentations.AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.3))
            if short_noise_dir != "":
                augmentations.append(audiomentations.AddShortNoises(sounds_path=short_noise_dir, p=0.5, lru_cache_size=1024))
            augmentations.append(audiomentations.Resample(min_sample_rate=sample_rate * 0.9, max_sample_rate=sample_rate * 1.1, p=speed_perturbation_probability))

            self.augment = audiomentations.Compose(augmentations)


    def get_wav_audio(self, wav):
        wav_tensor, utt_sample_rate = torchaudio.load(wav)
        if utt_sample_rate != self.sample_rate:
            wav_tensor = torchaudio.functional.resample(wav_tensor, utt_sample_rate, self.sample_rate) 
        if wav_tensor.shape[0] != 1:
            wav_tensor = wav_tensor.mean(dim=0, keepdim=True)
        wav_tensor = wav_tensor[0]
        return wav_tensor    

    def __getitem__(self, index):
        utt = self.utts[index]
        return {"index": index,
                "utt": utt, 
                "wav": self.utt2file[utt],
                "label": self.utt2label[utt],
                "label_id": self.label2id[self.utt2label[utt]],
                "dur": self.utt2dur[utt]}

    def __len__(self):
        return len(self.utts)

    def index2utt(self, index):
        return self.utts[index]

    def utt2index(self, utt):
        return self.utt2index[utt]



    def collater(self, samples):
        """Merge a list of wavs to form a mini-batch.

        Args:
            samples (List[dict]): wavs to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        if len(samples) == 0:
            return {}

        indexes = [s["index"] for s in samples]
        wavs = [s["wav"] for s in samples]
        utts = [s["utt"] for s in samples]

        audios = []
        durs = []

        if self.extract_chunks:
            chunk_length = random.uniform(self.min_length, self.max_length)
            chunk_length_in_samples = int(chunk_length * self.sample_rate)
            collated_audio = torch.zeros(len(wavs), chunk_length_in_samples)

            start_positions = []
            for i, wav in enumerate(wavs):
                audio_tensor = self.get_wav_audio(wav)
                if self.augment is not None:
                    audio_tensor = torch.from_numpy(self.augment(samples=audio_tensor.numpy(), sample_rate=self.sample_rate))

                # TODO: speed perturbation
                start_pos = random.randint(0, max(0, len(audio_tensor) - chunk_length_in_samples))
                start_positions.append(int(start_pos * 100))
                current_chunk_length = min(chunk_length_in_samples, len(audio_tensor))
                chunk = audio_tensor[start_pos:start_pos+current_chunk_length]

                audios.append(chunk)
                durs.append(len(chunk))
        else:
            mel_spec = []
            mel_spec_lengths = torch.zeros(len(wavs),  dtype=torch.int32)
            start_positions = []
            for i, wav in enumerate(wavs):
                audio_tensor = self.get_wav_audio(wav)
                start_positions.append(0)
                if self.augment is not None:
                    audio_tensor = torch.from_numpy(self.augment(samples=audio_tensor.numpy(), sample_rate=self.sample_rate))
                audios.append(audio_tensor)
                durs.append(len(audio_tensor))

        batch = {
            "index": torch.tensor(indexes),
            "wav": torch.nn.utils.rnn.pad_sequence(audios, batch_first=True),
            "dur": torch.tensor(durs),
            "label_id": torch.tensor([s["label_id"] for s in samples])
        }

        return batch

    @staticmethod
    def add_data_specific_args(parent_parser, root_dir):  # pragma: no cover
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--rir-dir', default="", type=str)
        parser.add_argument('--noise-dir', default="", type=str)
        parser.add_argument('--short-noise-dir', default="", type=str)
        parser.add_argument('--speed-perturbation-probability', default=0.0, type=float)        
        parser.add_argument('--min-chunk-length', default=2.0, type=float)
        parser.add_argument('--max-chunk-length', default=4.0, type=float)
        return parser



if __name__ == "__main__":
    logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

    dataset = WavDataset("../youtube-lid/data/lre07/eval_sad_filtered_wav", extract_chunks=False)
    #breakpoint()
    dl = torch.utils.data.DataLoader(
        dataset=dataset, batch_size=64, num_workers=0, collate_fn=dataset.collater, shuffle=True)
    it = iter(dl)
    

    it = iter(dl)
    for i in tqdm(range(50)):
        batch = next(it)
        breakpoint()


    #i = dev_dataset[0]
