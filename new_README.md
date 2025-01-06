```bash
conda create -n "imuposer" python=3.8
conda activate imuposer
conda install cudatoolkit=11.8
conda install pytorch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
python -m pip install -r new_requirements.txt
python -m pip install -e src/
```

```bash
cd scripts/1.\ Preprocessing/
python 1.\ preprocess_all.py
python 2.\ preprocess_all_to_imuposer_at_25fps.py
```

```bash
cd scripts/2.\ Train/
bash run_combos.sh
```