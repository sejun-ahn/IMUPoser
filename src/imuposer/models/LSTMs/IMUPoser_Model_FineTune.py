r"""
IMUPoser Model
"""

import torch.nn as nn
import torch
import pytorch_lightning as pl
from imuposer.models.loss_functions import *
from imuposer.smpl.parametricModel import ParametricModel
from imuposer.math.angular import r6d_to_rotation_matrix

class IMUPoserModelFineTune(pl.LightningModule):
    r"""
    Inputs - N IMUs, Outputs - SMPL Pose params (in Rot Matrix)
    """
    def __init__(self, config, pretrained_model):
        super().__init__()
        # load a pretrained model
        self.pretrained_model = pretrained_model
        self.n_pose_output = pretrained_model.n_pose_output

        self.batch_size = config.batch_size
        self.config = config

        if config.use_joint_loss:
            self.bodymodel = ParametricModel(config.og_smpl_model_path, device=config.device)

        if config.loss_type == "mse":
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.L1Loss()

        self.lr = 3e-4
        self.save_hyperparameters(ignore=['pretrained_model'])
        
        #modified
        self.train_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, imu_inputs, imu_lens):
        pred_pose = self.pretrained_model(imu_inputs, imu_lens)
        return pred_pose

    def training_step(self, batch, batch_idx):
        imu_inputs, target_pose, input_lengths, _ = batch

        _pred = self(imu_inputs, input_lengths)

        pred_pose = _pred[:, :, :self.n_pose_output]
        _target = target_pose
        target_pose = _target[:, :, :self.n_pose_output]
        loss = self.loss(pred_pose, target_pose)
        if self.config.use_joint_loss:
            pred_joint = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(pred_pose).view(-1, 216))[1]
            target_joint = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(target_pose).view(-1, 216))[1] ## If training is slow, get this from the dataloader
            joint_pos_loss = self.loss(pred_joint, target_joint)
            loss += joint_pos_loss

        self.log(f"training_step_loss", loss.item(), batch_size=self.batch_size)
        
        #modified
        self.train_step_outputs.append(loss)

        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        imu_inputs, target_pose, input_lengths, _ = batch

        _pred = self(imu_inputs, input_lengths)

        pred_pose = _pred[:, :, :self.n_pose_output]
        _target = target_pose
        target_pose = _target[:, :, :self.n_pose_output]
        loss = self.loss(pred_pose, target_pose)
        if self.config.use_joint_loss:
            pred_joint = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(pred_pose).view(-1, 216))[1]
            target_joint = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(target_pose).view(-1, 216))[1] ## If training is slow, get this from the dataloader
            joint_pos_loss = self.loss(pred_joint, target_joint)
            loss += joint_pos_loss

        self.log(f"validation_step_loss", loss.item(), batch_size=self.batch_size)
        
        #modified
        self.validation_step_outputs.append(loss)

        return {"loss": loss}

    def predict_step(self, batch, batch_idx):
        imu_inputs, target_pose, input_lengths, _ = batch

        _pred = self(imu_inputs, input_lengths)

        pred_pose = _pred[:, :, :self.n_pose_output]
        _target = target_pose
        target_pose = _target[:, :, :self.n_pose_output]
        loss = self.loss(pred_pose, target_pose)
        if self.config.use_joint_loss:
            pred_joint = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(pred_pose).view(-1, 216))[1]
            target_joint = self.bodymodel.forward_kinematics(pose=r6d_to_rotation_matrix(target_pose).view(-1, 216))[1] ## If training is slow, get this from the dataloader
            joint_pos_loss = self.loss(pred_joint, target_joint)
            loss += joint_pos_loss
            
        #modified
        self.test_step_outputs.append(loss)

        return {"loss": loss.item(), "pred": pred_pose, "true": target_pose}

    #modified
    """
    def training_epoch_end(self, outputs):
        self.epoch_end_callback(outputs, loop_type="train")

    def validation_epoch_end(self, outputs):
        self.epoch_end_callback(outputs, loop_type="val")

    def test_epoch_end(self, outputs):
        self.epoch_end_callback(outputs, loop_type="test")

    def epoch_end_callback(self, outputs, loop_type="train"):
        loss = []
        for output in outputs:
            loss.append(output["loss"])

        # agg the losses
        avg_loss = torch.mean(torch.Tensor(loss))
        self.log(f"{loop_type}_loss", avg_loss, prog_bar=True, batch_size=self.batch_size)
    """
    def on_train_epoch_end(self):
        avg_loss = torch.stack(self.train_step_outputs).mean()
        self.log(f"train_loss", avg_loss, prog_bar=True, batch_size=self.batch_size)
        self.train_step_outputs.clear()
        
    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.validation_step_outputs).mean()
        self.log(f"val_loss", avg_loss, prog_bar=True, batch_size=self.batch_size)
        self.validation_step_outputs.clear()
        
    def test_epoch_end(self):
        avg_loss = torch.stack(self.test_step_outputs).mean()
        self.log(f"test_loss", avg_loss, prog_bar=True, batch_size=self.batch_size)
        self.test_step_outputs.clear()
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
