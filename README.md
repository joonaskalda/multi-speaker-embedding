# Diarization-Guided Multi-Speaker Embeddings
Authors: Joonas Kalda, Clément Pagés, Tanel Alumäe, Hervé Bredin  
Interspeech 2025  
[Paper](https://www.isca-archive.org/interspeech_2025/kalda25_interspeech.html)

> Reliable speaker embeddings are critical for multi-speaker speech processing tasks. Traditionally models are trained on single-speaker utterances and suffer from domain mismatch when applied in multi-speaker contexts. Recently proposed guided speaker embeddings (GSE) were shown to improve this by training on synthetic multi-speaker mixtures guided by oracle speaker activity labels. Additionally modeling all speakers present in a chunk is desirable but the performance of such methods has been sub-par up to now. We build on GSE by modeling multiple speakers together and using diarization features for guiding. We also propose a new validation metric for embeddings in multi-speaker context and demonstrate its effectiveness. Results on multiple speaker diarization datasets demonstrate that we improve on speed and performance while reducing the embedding model size.

## Training

To set up the diarization datasets using pyannote-database, follow the instructions [here](https://github.com/FrenchKrab/datasets-pyannote). To setup VoxCeleb, follow the instructions [here](https://github.com/pyannote/pyannote-db-voxceleb). For the exact configurations of the paper, you will also need the room background noise from [MUSAN](https://www.openslr.org/17/) and the simulated room impulse responses from [RIR](https://www.openslr.org/17/).

### Training the segmentation model

Training the multi-speaker embeddings requires a local segmentation model. We opt for a WavLM+LSTM based architecture for which a training script is given in train_segmentation.py. You will need to set the `DB_YAML_PATH` and `PROTOCOL_NAME` for your (compound) diarization dataset and the name of the protocol respectively. Note that our approach is compatible with any pre-trained local segmentation model (e.g. [DiariZen](https://github.com/BUTSpeechFIT/DiariZen)) with slight adjustments.

### Training the multi-speaker embeddings

Training script is given in train_embeddings.py. You will need to set the `DB_YAML_PATH` and `PROTOCOL_NAME` for your (compound) diarization dataset and the name of the protocol respectively. Also the `PATH_TO_NOISE` and `PATH_TO_RIR` should be indicated and the `PRETRAINED_SEGMENTATION_MODEL_PATH` should be the path to the pre-trained segmentation model. Since we use diarization datasets for validation, the compound yaml should look something like this:

```yaml
Requirements:
/path/to/voxceleb/database.yaml
/path/to/ami-sdm/database.yaml
# other diarization datasets

Protocols:
  X:
    SpeakerDiarization:
      MyProtocol:
          train:
            VoxCeleb.SpeakerVerification.VoxCeleb: [train, ]
            AMI-SDM.SpeakerDiarization.only_words: [train, ]
                # other diarization datasets
          development:
            AMI-SDM.SpeakerDiarization.only_words: [development, ]
                # other diarization datasets
          test:
            AMI-SDM.SpeakerDiarization.only_words: [test, ]
                # other diarization datasets
```

## Inference

```python
from pyannote.audio.pipelines import SpeakerDiarizationV2

MODEL_PATH = "path/to/model.ckpt"
WAV_FILE = "path/to/audio/file.wav"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model.from_pretrained(MODEL_PATH).to(device).eval()

pipeline = SpeakerDiarizationV2(
    model=model,
    batch_size=128,
).to(device)

# hyperparameters should be optimized on validation set
pipeline.instantiate({
    "clustering": {
        "method": "centroid",
        "min_cluster_size": 30,
        "threshold": 0.58,
    }
})
pipeline(WAV_FILE)
```