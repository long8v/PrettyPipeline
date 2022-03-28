## PrettyPipeline
### goals :golf:
- hydra를 이해한다
- config 관리를 fancy하게 한다
- 나만의 아름다운 파이프라인을 만든다
### steps :sunrise_over_mountains:
- [x] clone repo & environment setting
- [x] 그냥 실행.
`python train.py`
- [x] config 하나를 바꿔서 실행
`python train.py +train.minist.input_size=512`
- [x] mnist 모델말고 다운받은 데이터셋을 사용하여 학습을 돌려본다([code](https://github.com/long8v/PrettyPipeline/commit/e55df910dba6996a0d52326780a3eb9cff6d1463))
1) `MNISTDataModule`와 매우 유사하게 `src.datamodules.cifar_datamodule.CIFARDataModule`과 `configs/datamodule/cifar.yaml`을 만든다.
```
_target_: src.datamodules.cifar_datamodule.CIFARDataModule
data_dir: ${data_dir} # configs/train.yaml에 있다.
batch_size: 64
train_val_test_split: [55_000, 5_000, 10_000]
num_workers: 0
pin_memory: False
```
2) CIFARDataModule내에서 `cifar.yaml`은 `self.hparams`로 접근할 수 있다.
```
CIFAR10(root=self.hparams.data_dir, train=True, download=True)
```
3) 위의 코드들은 training_pipeline.py 내에서 instantiate된다.
```
datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)
```
4) mnist와 input 차원이 다르기 때문에 `configs/model/cifar.yaml`도 만들어준다. 모델은 그냥 똑같은거 쓴다.
```
_target_: src.models.mnist_module.MNISTLitModule
lr: 0.001
weight_decay: 0.0005
net:
  _target_: src.models.components.simple_dense_net.SimpleDenseNet
  input_size: 3072
  lin1_size: 256
  lin2_size: 256
  lin3_size: 256
  output_size: 10
```
5) `configs/train.yaml`을 아래와 같이 바꿔준다.
```
defaults:
  - _self_
  - datamodule: cifar.yaml
  - model: cifar.yaml
  - callbacks: default.yaml
```
6) `python train.py`!
- [ ] mnist 말고 간단한 모델을 만들어 본다
### materials :card_file_box:
- hydra 번역 doc : https://pjt3591oo.github.io/hydra_translate/build/html/index.html
- omegaconf + yaml은 variable 기능이 있음 : https://omegaconf.readthedocs.io/en/latest/usage.html#variable-interpolation
- path 설정은 이렇게 하면 될까? : `dotenv` or [omegaconf-env](https://omegaconf.readthedocs.io/en/latest/custom_resolvers.html#oc-env) ?
