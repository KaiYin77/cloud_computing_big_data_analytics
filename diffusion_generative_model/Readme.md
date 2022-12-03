# Diffusion_generative_model

## Dataset

- Unzip data.zip to `../data`

    ```sh
    unzip data.zip -d ../data
    ```

- Folder structure

    ```txt
    .
    ├── data/
    │   └── hw3/
	│ 		├── mnist/
    │   	└── mnist.npz
    │ 
	└── diffusion_generative_model 
    	├── grid/
    	├── samples/
    	├── src
		│ 	├── __init__.py
		│ 	├── diffusion.py
		│ 	├── models.py
    	│   └── utils.ckpt
    	├── weights
    	│   └── epoch01.ckpt
    	├── config.yaml
    	├── dataloader.py
    	├── get_fid_score.py
    	├── Readme.md
    	├── requirements.txt
    	└── trainer.py
    ```
## Environment
Boosted by pytorch-lightning

### Create Environment

```sh
conda create --name ccbda_hw3 python=3.7
conda activate ccbda_hw3

pip3 install -r requirements.txt
```


## Training
```sh
python train.py --train [--ckpt xxx.ckpt]
```
- `--train` : train mode
- `--ckpt CKPT` : resume from the checkpoint 

## Sampling

```sh
python train.py --test --ckpt xxxxx.ckpt 
```

## Calculate FID score
```sh
python get_fid_score.py
```

