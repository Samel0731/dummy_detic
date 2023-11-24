# funny_Dummy_Device

### 功能：

    輸入：讀入一張照片，使用者指定要從照片中找什麼物件（只能輸入英文）
    輸出：找到的物件個數，並讓 iottalk 上的燈泡閃爍次數等於找到物件個數

### 如何使用：

  ```sh
  # create virtaul venv
  $ python -m venv .venv
  
  # activate the virtaul env
  $ source .venv/bin/activate # linux
  $ .venv/Script/activate.bat # windows
  $ .venv/Script/activate.ps1 # vscode (PowerShell)

  # install packages
  $ pip install -r requirements.txt

  # run the DAI.py
  $ python ./DAI.py
  ```

### demo video:
demo_funny_dummy_device.mp4

## 使用到的模型 Detic

facebookresearch, Computer Vision and Pattern Recognition 2022, arxiv

github [repo](https://github.com/facebookresearch/Detic)

paper [link](https://arxiv.org/abs/2201.02605)

## iottalk

github [repo](https://github.com/IoTtalk/Dummy_Device_IoTtalk_v1_py)