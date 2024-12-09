# Dacon_restore

### [주제]

이미지의 색상화와 손실 부분을 복원하는 AI 알고리즘 개발

### [설명]

손실된 이미지의 결손 부분을 복구하고, 흑백 이미지에 자연스러운 색을 입히는 AI 알고리즘 개발

## 1. FlowCart
![image](https://github.com/user-attachments/assets/ebc25581-809c-402f-a888-755ca40e2edf)

## 2. 데이터 

![image](https://github.com/user-attachments/assets/fba409e2-d1f8-48d6-9d89-18890124619b)

차례대로 train_input, train_gt, test_intput입니다. train_input과 test_input이 매우 흡사하여 충분한 epoch로 훈련한다면 좋은 성과가 나올 것이라고 생각했습니다.

- Augmentation 진행 xx : 가로-세로 Flip, 회전, 부분 earasing 등 다양한 기법들을 사용해봤지만 결과는 좋지 않았습니다. 해당 과제에서는 Augmentation보다 loss를 이용하는 것이 중요하다고 생각합니다. 
- train / val 분할 (8:2)
- data limit : train = 10000개 / val = 3000개

## 3. 모델
G_model = Inpainting Generator
D_model = LSGANDiscriminator

생성자와 판별자의 모델을 다르게 해 inpainting에 적합하게 모델을 적용시켰습니다. 모델에 Residule block 및 gateway block을 활용하여 중요한 정보는 전달되면서 모델의 깊이를 깊게 하고자 했지만 예상 외로 성능이 낮아지는 것을 확인해 기본 모델 그대로 두었습니다. (건들면 건들수록 더 안좋아지는 느낌 ..)

- shceduler : StepLr
- Optimizer : AdamW
- G_loss : g_loss_adv + 50 * g_loss_pixel + 50 * g_loss_ssim
- D_loss : loss_real + loss_fake + lambda_gp * gp

## 4. 결과


## 5. 시도한 실험


## 7. 결론
이번 대회에서 아쉬웠던 점은 마지막까지 

## 8. 개발 환경
conda == 24.5
autocommand==2.2.2
backports.tarfile==1.2.0
Brotli==1.0.9
certifi==2024.8.30
charset-normalizer==3.3.2
click==8.1.7
contourpy==1.1.1
cycler==0.12.1
docker-pycreds==0.4.0
filelock==3.16.1
fonttools==4.55.0
fsspec==2024.10.0
gitdb==4.0.11
GitPython==3.1.43
graphviz==0.20.3
hdbscan==0.8.40
huggingface-hub==0.26.3
idna==3.7
imageio==2.35.1
importlib-metadata==8.5.0
importlib-resources==6.4.5
inflect==7.3.1
jaraco.collections==5.1.0
jaraco.context==5.3.0
jaraco.functools==4.0.1
jaraco.text==3.12.1
joblib==1.4.2
kiwisolver==1.4.7
lazy-loader==0.4
llvmlite==0.41.1
matplotlib==3.7.5
mkl-fft==1.3.8
mkl-random==1.2.4
mkl-service==2.4.0
more-itertools==10.3.0
natsort==8.4.0
networkx==3.1
numba==0.58.1
numpy==1.24.3
opencv-python==4.5.5.64
packaging==24.2
pandas==2.0.3
pillow==10.4.0
pip==24.2
platformdirs==4.3.6
protobuf==5.29.0
psutil==6.1.0
pynndescent==0.5.13
pyparsing==3.1.4
PySocks==1.7.1
python-dateutil==2.9.0.post0
pytz==2024.2
PyWavelets==1.4.1
PyYAML==5.4
regex==2024.11.6
requests==2.32.3
safetensors==0.4.5
scikit-image==0.21.0
scikit-learn==1.3.2
scipy==1.10.1
segment-anything==1.0
sentry-sdk==2.19.0
setproctitle==1.3.4
setuptools==75.1.0
six==1.16.0
smmap==5.0.1
tensorboardX==2.6.2.2
threadpoolctl==3.5.0
tifffile==2023.7.10
timm==1.0.11
tokenizers==0.20.3
tomli==2.0.1
torch==1.13.0
torchaudio==0.13.0
torchsummary==1.5.1
torchvision==0.14.0
torchviz==0.0.3
tqdm==4.67.1
transformers==4.46.3
typeguard==4.3.0
typing-extensions==4.11.0
tzdata==2024.2
umap-learn==0.5.7
urllib3==2.2.3
wandb==0.18.7
wheel==0.44.0
zipp==3.20.2
