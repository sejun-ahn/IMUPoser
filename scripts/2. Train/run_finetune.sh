#!/bin/bash

combos='global'

for combo in $combos
do
    echo Running combo $combo
    python 2.\ FineTune\ Trained\ Model.py --combo_id $combo --experiment 'IMUPoserGlobalModelFineTuneDIP' --ckpt_path '../../checkpoints/IMUPoserGlobalModel_global-01192025-184309/epoch=epoch=839-val_loss=validation_step_loss=0.00565.ckpt'
done