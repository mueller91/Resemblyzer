import os
from pathlib import Path

import numpy
from tqdm import tqdm
from Resemblyzer.resemblyzer.voice_encoder import VoiceEncoder
from Resemblyzer.resemblyzer.audio import preprocess_wav as res_pp

from load_datasets import get_preprocessor_by_name
from tacotron import TacotronHparams

res_enc = VoiceEncoder()

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
        print(f"Processing {name}, writing to {TacotronHparams.path_precomputed_embeddings}")

        # [text, file, speaker_id, start, end]
        preprocessor = get_preprocessor_by_name(name)
        meta_data = preprocessor(path)
        for m in tqdm(meta_data):
            text, file, spid, start, end = m
            file_embds = Path(file.replace('/opt/mueller/audio-datasets/en', TacotronHparams.path_precomputed_embeddings))
            os.makedirs(str(file_embds.parent), exist_ok=True)
            wav = res_pp(file)
            embds = res_enc.embed_utterance(wav)
            numpy.save(file_embds, embds)
        done_datasets.append(dataset)



