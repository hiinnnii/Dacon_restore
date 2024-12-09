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
  
  ![image](https://github.com/user-attachments/assets/58216450-0a30-4cc4-b1a6-00adad11fe38)

- train / val 분할 (8:2)
- data limit : train = 10000개 / val = 3000개

## 3. 모델
#### G_model = Inpainting Generator
#### D_model = LSGANDiscriminator

생성자와 판별자의 모델을 다르게 해 inpainting에 적합하게 모델을 적용시켰습니다. 모델에 Residule block 및 gateway block을 활용하여 중요한 정보는 전달되면서 모델의 깊이를 깊게 하고자 했지만 예상 외로 성능이 낮아지는 것을 확인해 기본 모델 그대로 두었습니다. (건들면 건들수록 더 안좋아지는 느낌 ..)

- shceduler : StepLr-G는 10 step에 한번씩 0.5 줄이고, D는 15에 step에 한번씩 0.5 줄임
- Optimizer : AdamW
- G_loss 계산 방법 : g_loss_adv + 50 * g_loss_pixel + 50 * g_loss_ssim
- D_loss 계산 방법 : loss_real + loss_fake + lambda_gp * gp
- D_loss 계산 시 Gradient Panalty 적용 : Discriminator의 그라디언트 크기를 1에 가깝게 유지

## 4. 결과
epoch = 43
Dacon Public Score = 0.52
Dacon Private Score = 0.54 (26위)

![image](https://github.com/user-attachments/assets/5b7928a3-fb51-40dc-a9f1-1f164d72572c)

다양한 방법들을 시도해보면서 D_loss와 G_loss의 학습 속도 간격이 벌어져 성능이 안나온다는 것이 큰 문제였습니다. Gan에서 이 문제를 해결하기 위해 다양한 시도들이 있었고, D_loss와 G_loss의 차이를 줄이기 위해 가장 효과적이었던 방법은 Gradient Panalty를 적용하는 것이었습니다. 

ssim loss는 두 이미지의 유사도를 비교하는 방식으로 validation을 평가할 때도 사용했지만 train 중에서도 사용해 원본과 더 유사한 이미지를 만들고자 하였습니다. ssim loss의 가중치를 더 높이면 더 좋은 결과가 나올 것이라 예상했지만 생각과는 다르게 좋지 않은 결과가 나와 G_loss의 가중치는 이후 건들지 않고, g_loss_adv + 50 * g_loss_pixel + 50 * g_loss_ssim를 사용했습니다. 아래 사진은 ssim loss 가중치를 70으로 늘렸을 때 사진입니다.

![image](https://github.com/user-attachments/assets/7b7a35c6-b85b-43d7-a932-a7b7670b13f6)


baseline에서 성능을 올릴 수 있었던 가장 큰 이유는 Gradient Panalty, ssim loss의 적인 것 같습니다.

물체 복원은 형태가 보이니 이정도에서 만족합니다... ㅎㅎ all data를 사용하고, epoch를 더 늘린다면 좋은 성능을 보일 것이라 예상했습니다. 아래 이미지를 보면 환경 이미지에서는 어느정도 복원을 잘 해낸 것으로 보이지만, 색상 복원을 잘 해내지 못한다는 점에서 점수가 많이 낮은 것이라 예상됩니다. 

![image](https://github.com/user-attachments/assets/61a06d73-d4c5-4844-bf02-35a654f4ba34)

## 5. 추가 실험 아이디어
- ssim을 활용하여 train_mask 생성 후 학습 진행
  
  ![image](https://github.com/user-attachments/assets/4af9dbfe-ff34-4df6-a5b8-d1876a648147)

  대부분 mask가 잘 형성되었지만 일부 데이터셋에서는 잘 만들어지지 않아 학습이 잘 안되어 결과물이 좋지 않았습니다. 
- Test 결과에 다양한 필터 입혀보기
  
  ![그림1](https://github.com/user-attachments/assets/58a9699c-28ee-4a85-8cb7-a9a208984a57)

- 2단계 UNet 생성하기
  1단계 UNet에서 흑백 - 흑백 으로 복원할 부분만 복원시키고자 했지만 색상이랑 같이 변화했을 때와 비슷한 결과물이 출력되었고, 이 방법은 시간이 너무 오래 걸려 포기하였습니다.
  2단계 UNet에서 흑백 - 컬러로 변환시켰을 때 복원할 부분이 제대로 복원되지 않은 상태로 컬러를 입히려니 흑백 -> 컬러로 변화했을 때보다 기괴한 결과물을 내보였습니다.

## 7. 결론
이번 대회에서 아쉬웠던 점은 baseline + ssim을 적용한 코드가 최종 코드라고 생각했고, 이 코드를 돌리면 30등 안에는 들 수 있을 것이라 생각했었는데 BIGGGGGG 과적합이 나버려 .. epoch 30 결과물 score가  0.41이 되었습니다. D_loss와 G_loss의 차이가 벌어져서 그렇다고 판단하였고, Gradient Panalty를 적용한 후에 성능이 좋아진 것을 확인했을 땐 남은 시간이 없어 전체 데이터로 진행하지 못해 아쉬웠습니다. 마지막에 data 10000개로 돌린 것이 그래도 최종 private에서는 0.54로 기특하게 커주어 뿌듯했고 시도한 것에 비해 좋은 성과를 내지 못해 아쉬움이 있었습니다. 

UNet으로 class 분류는 해봤지만 inpainting 기술은 처음 사용해봐 많은 흥미를 느낄 수 있었습니다. 또한 재밌었던 점은 UNet이 크게 훼손된 부분은 그냥 막무가내로 복원할 줄 알았는데 생각보다 논리적으로 복원을 한 다는 점이었습니다.

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
