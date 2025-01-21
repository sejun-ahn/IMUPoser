import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything

from imuposer.config import Config, amass_combos
from imuposer.models.utils import get_model
from imuposer.datasets.utils import get_datamodule
from imuposer.utils import get_parser, get_checkpoints
from imuposer.models.LSTMs.IMUPoser_Model import IMUPoserModel
seed_everything(20241078, workers=True)

parser = get_parser()
args = parser.parse_args()
combo_id = args.combo_id
fast_dev_run = False
_experiment = args.experiment
ckpt_path = args.ckpt_path

config = Config(experiment=f"{_experiment}_{combo_id}", model="GlobalModelIMUPoserFineTuneDIP",
                project_root_dir="../../", joints_set=amass_combos[combo_id], normalize="no_translation",
                r6d=True, loss_type="mse", use_joint_loss=True, device="0")

model_ = IMUPoserModel.load_from_checkpoint(ckpt_path, strict=False)
model = get_model(config=config, pretrained=model_)

datamodule = get_datamodule(config=config)
checkpoint_path = config.checkpoint_path

wandb_logger = WandbLogger(project=config.experiment, save_dir=checkpoint_path)

early_stopping_callback = EarlyStopping(monitor="validation_step_loss", mode="min", verbose=False,
                                        min_delta=0.00001, patience=5)

checkpoint_callback = ModelCheckpoint(monitor="validation_step_loss", mode="min", verbose=False, 
                                      save_top_k=5, dirpath=checkpoint_path, save_weights_only=False, 
                                      filename='epoch={epoch}-val_loss={validation_step_loss:.5f}')

trainer = pl.Trainer(fast_dev_run=fast_dev_run, logger=wandb_logger, max_epochs=100, accelerator="gpu", devices=[0, 1],
                     strategy="ddp", callbacks=[early_stopping_callback, checkpoint_callback], deterministic=True)

trainer.fit(model_, datamodule=datamodule)

with open(checkpoint_path / "best_model.txt", "w") as f:
    f.write(f"{checkpoint_callback.best_model_path}\n\n{checkpoint_callback.best_k_models}")
