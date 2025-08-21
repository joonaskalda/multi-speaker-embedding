import math
import torch
from types import MethodType
from torch.optim.lr_scheduler import _LRScheduler
from torch_audiomentations import AddBackgroundNoise, ApplyImpulseResponse
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import TensorBoardLogger
from pyannote.database import FileFinder, registry
from pyannote.audio.core.io import get_torchaudio_info
from pyannote.audio.tasks.joint_task.speaker_diarization_and_embedding import (
    JointSpeakerDiarizationAndEmbedding,
)
from pyannote.audio.core.model import Model
from pyannote.audio.models.joint.multi_embedding import MultiEmbedding

pl.seed_everything(42)
torch.set_float32_matmul_precision("medium")

DB_YAML_PATH = "path/to/database.yaml"
PROTOCOL_NAME = "MyDatabase.Protocol.MyProtocol"
PRETRAINED_SEGMENTATION_MODEL_PATH = "path/to/pretrained_segmentation_model.ckpt"
PATH_TO_NOISE = "path/to/noise"
PATH_TO_RIR = "path/to/rir"

class AbsBatchStepScheduler:
    def step(self, epoch=None):
        pass

    def state_dict(self):
        pass

    def load_state_dict(self, state):
        pass


class CosineAnnealingWarmupRestarts(_LRScheduler, AbsBatchStepScheduler):
    def __init__(
        self,
        optimizer,
        first_cycle_steps,
        cycle_mult=1.0,
        max_lr=0.1,
        min_lr=0.001,
        warmup_steps=0,
        gamma=1.0,
        last_epoch=-1,
    ):
        assert warmup_steps < first_cycle_steps
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        super().__init__(optimizer, last_epoch)
        self.init_lr()

    def init_lr(self):
        self.base_lrs = [self.min_lr for _ in self.optimizer.param_groups]

    def get_lr(self):
        if self.step_in_cycle == -1:
            return self.base_lrs
        elif self.step_in_cycle < self.warmup_steps:
            return [
                base_lr
                + (self.max_lr - base_lr) * self.step_in_cycle / self.warmup_steps
                for base_lr in self.base_lrs
            ]
        else:
            return [
                base_lr
                + (self.max_lr - base_lr)
                * (
                    1
                    + math.cos(
                        math.pi
                        * (self.step_in_cycle - self.warmup_steps)
                        / (self.cur_cycle_steps - self.warmup_steps)
                    )
                )
                / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle += 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle -= self.cur_cycle_steps
                self.cur_cycle_steps = (
                    int((self.cur_cycle_steps - self.warmup_steps) * self.cycle_mult)
                    + self.warmup_steps
                )
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(
                        math.log(
                            (
                                epoch / self.first_cycle_steps * (self.cycle_mult - 1)
                                + 1
                            ),
                            self.cycle_mult,
                        )
                    )
                    self.cycle = n
                    self.step_in_cycle = epoch - int(
                        self.first_cycle_steps
                        * (self.cycle_mult**n - 1)
                        / (self.cycle_mult - 1)
                    )
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult**n
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch

        self.max_lr = self.base_max_lr * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


def main():
    
    # Setup database and preprocessors
    preprocessors = {"audio": FileFinder(), "torchaudio.info": get_torchaudio_info}
    registry.load_database(DB_YAML_PATH)

    # Setup augmentations
    noise_augmentation = AddBackgroundNoise(
        background_paths=PATH_TO_NOISE,
        min_snr_in_db=0.0,
        max_snr_in_db=10.0,
        mode="per_example",
        p=1.0,
        output_type="dict",
    )
    rir_augmentation = ApplyImpulseResponse(
        ir_paths=PATH_TO_RIR,
        convolve_mode="full",
        compensate_for_propagation_delay=True,
        p=0.5,
        output_type="dict",
    )

    # Setup protocol and task
    protocol = registry.get_protocol(PROTOCOL_NAME, preprocessors=preprocessors)
    task = JointSpeakerDiarizationAndEmbedding(
        protocol,
        duration=10,
        dia_task_rate=0.0,
        alpha=0.0,
        margin=11.4,
        scale=30.0,
        batch_size=128,
        cache=f"cache/{PROTOCOL_NAME}", # data pre-processing takes a few hours for VoxCeleb
        num_workers=8,
        diar_pooling=True,
        noise_augmentation=noise_augmentation,
        rir_augmentation=rir_augmentation,
        mean_var_norm=True,
        ami_aam_weight=0.0,
        vc_dia_weight=0.0,
        max_speakers_per_chunk=3,
        lambda_param=0.2,
        normalize_utterances=True,
    )

    # Initialize model
    model = MultiEmbedding(
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
        freeze_wav2vec=True,
        automatic_optimization=False,
        gradient_clip_val=5,
        task=task,
        num_classes=7205,
        margin=11.4,
        scale=30.0,
        num_classes_diar=152,
    )

    # Load pretrained model components
    model_diar = Model.from_pretrained(PRETRAINED_SEGMENTATION_MODEL_PATH)
    model.wav2vec = model_diar.wav2vec
    model.classifier = model_diar.classifier
    model.lstm = model_diar.lstm
    model.linear = model_diar.linear
    model.dia_wav2vec_weights = model_diar.dia_wav2vec_weights
    model.lstm.dropout = 0.0

    # Configure optimizers
    def configure_optimizers(self):
        pretrained_param_names = [
            "classifier",
            "lstm",
            "linear",
            "dia_wav2vec_weights",
        ]
        pretrained_params = []
        other_params = []
        for kv in self.named_parameters():
            if any(param in kv[0][: len(param)] for param in pretrained_param_names):
                pretrained_params.append(kv)
            elif "wav2vec." in kv[0][: len("wav2vec.")]:
                continue
            else:
                other_params.append(kv)
        optimizer_rest = torch.optim.Adam(dict(other_params).values(), lr=1e-3)
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer_rest,
            first_cycle_steps=50000,
            cycle_mult=1,
            max_lr=1e-3,
            min_lr=0.000005,
            warmup_steps=1000,
            gamma=0.75,
        )
        return {
            "optimizer": optimizer_rest,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    model.configure_optimizers = MethodType(configure_optimizers, model)

    # Setup callbacks
    callbacks = [LearningRateMonitor(logging_interval="step")]
    
    checkpoint = ModelCheckpoint(
        monitor="DiarizationErrorRate",
        dirpath=None,
        mode="min",
        save_top_k=-1,
        every_n_epochs=1,
        save_last=True,
        save_weights_only=False,
        verbose=False,
    )
    callbacks.append(checkpoint)

    # Setup trainer
    trainer = pl.Trainer(
        callbacks=callbacks,
        accelerator="gpu",
        devices=1,
        num_nodes=1,
        benchmark=True,
        accumulate_grad_batches=1,
        deterministic=False,
        check_val_every_n_epoch=1,
        log_every_n_steps=50,
        max_epochs=150,
        max_steps=-1,
        max_time=None,
        min_epochs=1,
        min_steps=None,
        num_sanity_val_steps=2,
        limit_val_batches=1.0,
        val_check_interval=0.2,
    )

    model = model.to("cuda:0")
    trainer.fit(model)
    
    checkpoint.to_yaml()


if __name__ == "__main__":
    main()
