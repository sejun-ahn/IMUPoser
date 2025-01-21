#!/bin/bash

combos='global'

for combo in $combos
do
  echo Running combo $combo resume
  python 1.\ Train\ Global\ Model.py --combo_id $combo --experiment 'IMUPoserGlobalModel' --resume --ckpt_path '../../checkpoints/IMUPoserGlobalModel_global-01182025-182217/epoch=epoch=524-val_loss=validation_step_loss=0.00683.ckpt' 
done