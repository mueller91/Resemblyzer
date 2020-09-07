import os
from collections import defaultdict
from pathlib import Path

import numpy
import sklearn
from tqdm import tqdm
from Resemblyzer.resemblyzer.voice_encoder import VoiceEncoder
from Resemblyzer.resemblyzer.audio import preprocess_wav as res_pp

from load_datasets import get_preprocessor_by_name
from tacotron import TacotronHparams

res_enc = VoiceEncoder()

def get_emb_path(file):
    return Path(file.replace('/opt/mueller/audio-datasets/en', TacotronHparams.path_precomputed_embeddings))

if __name__ == "__main__":
    """
    Using the data in Hparams, create the corresponding embeddings
    """
    done_datasets = []
    for dataset in TacotronHparams.datasets:
        if dataset in done_datasets:
            print(f"Already computed {dataset} during earlier run (train / test duplicate?)")
            continue
        path = dataset['path']
        name = dataset['name']
        usage = dataset['usage']
        if usage is not 'train':
            continue
        print(f"Processing {name}, writing to {TacotronHparams.path_precomputed_embeddings}")

        # [text, file, speaker_id, start, end]
        preprocessor = get_preprocessor_by_name(name)
        meta_data = preprocessor(path)
        # for m in tqdm(meta_data, desc="Processing wav -> embedding"):
        #     text, file, spid, start, end = m
        #     file_embds = file_to_embd(file)
        #     os.makedirs(str(file_embds.parent), exist_ok=True)
        #     wav = res_pp(file)
        #     embds = res_enc.embed_utterance(wav)
        #     numpy.save(file_embds, embds)
        # done_datasets.append(dataset)

        d = defaultdict(lambda: [])
        for m in tqdm(meta_data, desc="Creating speaker dict"):
            text, file, spid, start, end = m
            d[spid].append(file)

        for spid in tqdm(d.keys(), desc="Averaging embeddings"):
            all_embds = [str(get_emb_path(f)) + ".npy" for f in d[spid]]
            all_embds = [numpy.load(x) for x in all_embds]
            avg_emb = numpy.stack(all_embds).mean(axis=0)
            cosine_sims = sklearn.metrics.pairwise.cosine_similarity(numpy.expand_dims(avg_emb, 0), numpy.stack(all_embds))

            # save to all folders (easier!)
            saveto_paths = [get_emb_path(f).parent / 'averaged_embedding.npy' for f in d[spid]]
            for s in saveto_paths:
                numpy.save(s, avg_emb)
            # print(f"Cosine sim for speaker {spid} to averaged embeddings is (avg'd): {cosine_sims.mean():.3f}. Saving to {s} and others")


