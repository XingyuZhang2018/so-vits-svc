# SoftVC VITS Singing Voice Conversion

A package copy from [here](https://github.com/svc-develop-team/so-vits-svc) and modified for single speaker singing voice conversion.
# requirements
```
pip install --upgrade pip setuptools numpy numba
pip install pyworld praat-parselmouth fairseq
curl -L https://ibm.ent.box.com/shared/static/z1wgl1stco8ffooyatzdwsqn2psd9lrr -o /content/so-vits-svc/hubert/checkpoint_best_legacy_500.pt
```

# steps
## define singer
```
speaker=Dehua_Liu
```
## Resample to 44100Hz and mono
```
python resample.py --in_dir "./dataset_raw/$speaker" --out_dir2 "./dataset/44k/$speaker"
```

## Divide filelists and generate config.json
```
mkdir ./filelists/$speaker/ && mkdir ./configs/$speaker/ && python preprocess_flist_config.py --train_list "./filelists/$speaker/train.txt" --test_list "./filelists/$speaker/test.txt" --val_list "./filelists/$speaker/val.txt" --source_dir "./dataset/44k/$speaker"
```

## Generate hubert and f0
```
python preprocess_hubert_f0.py --in_dir "./dataset/44k/$speaker" --s "$speaker"
```

## copy D_0 and G_0
```
mkdir logs/44k/$speaker && cp logs/44k/D_0.pth logs/44k/$speaker/D_0.pth && cp logs/44k/G_0.pth logs/44k/$speaker/G_0.pth
```

## Start training
```
python train.py -c configs/$speaker/config.json -m 44k/$speaker
```

## train_cluster
```
python cluster/train_cluster.py --dataset "./dataset/44k/$speaker" --output "logs/44k/$speaker"
```

## interence
```
python inference_main.py -m "logs/44k/$speaker/G_69600.pth" -c "configs/$speaker/config.json" -n "富士山下_原唱" -t 0 -s "$speaker"
```