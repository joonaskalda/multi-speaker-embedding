import os
from pyannote.audio import Model
from argparse import ArgumentParser, Namespace
from pyannote.database import get_protocol, FileFinder, registry
from pyannote.audio.tasks import SpeakerDiarization
import pytorch_lightning as pl
from pyannote.audio.models.segmentation import SSeRiouSS
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)
import torch
from types import MethodType

pl.seed_everything(42)
torch.set_float32_matmul_precision("medium")

DB_YAML_PATH = "path/to/database.yaml"
PROTOCOL_NAME = "MyDatabase.Protocol.MyProtocol"

def main():
    preprocessors = {"audio": FileFinder()}
    registry.load_database(DB_YAML_PATH)
    dataset = get_protocol(PROTOCOL_NAME, preprocessors=preprocessors)

    task = SpeakerDiarization(
        dataset,
        duration=10.0,
        max_speakers_per_chunk=3,
        max_speakers_per_frame=2,
        batch_size=32,
        num_workers=8,
    )

    segmentation_model = SSeRiouSS(
        wav2vec="WAVLM_BASE_PLUS",
        lstm={
            "hidden_size": 128,
            "num_layers": 4,
            "bidirectional": True,
            "monolithic": True,
            "dropout": 0.5,
        },
        linear={
            "hidden_size": 128,
            "num_layers": 2,
        },
        freeze_wav2vec=False,
        automatic_optimization=False,
        wav2vec_layer=-1,
        task=task,
    )

    checkpoint = ModelCheckpoint(
        dirpath=None,
        monitor="loss/val",
        mode="min",
        save_top_k=10,
        every_n_epochs=1,
        save_last=True,
        save_weights_only=False,
        verbose=True,
    )

    callbacks = [checkpoint]
    callbacks.append(
        EarlyStopping(monitor="loss/val", mode="min", patience=20, verbose=True)
    )

    def configure_optimizers(self):
        optimizer_wavlm = torch.optim.Adam(self.wav2vec.parameters(), lr=1e-5)
        other_params = list(
            filter(lambda kv: "wav2vec." not in kv[0], self.named_parameters())
        )
        optimizer_rest = torch.optim.Adam(dict(other_params).values(), lr=3e-4)
        return [optimizer_wavlm, optimizer_rest]

    segmentation_model.configure_optimizers = MethodType(
        configure_optimizers, segmentation_model
    )

    task.prepare_data()
    task.setup()

    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=callbacks,
        accelerator="gpu",
        devices=1,
        num_nodes=1,
    )
    trainer.fit(segmentation_model)


if __name__ == "__main__":
    main()