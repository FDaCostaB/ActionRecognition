
# ML for Human Action Recognition

A machine learning project based on an academic project to recognize human action based on still pictures or optical flow precomputed with opencv.

## Installation

Create a python virtual environment (venv)

```bash
python -m venv env
OR
python3 -m venv env
```

Activate the environment (Linux and macOS):
```bash
source env/bin/activate
```

Activate the environment (Windows):
```bash
env\Scripts\activate.bat
```

Install required package:

```bash
pip install -r /requirements.txt
```

Create a `plots` folder in ./output/

## Models description

### cnn
![cnn](https://github.com/user-attachments/assets/7e4d9ea9-f0b8-4240-a115-8ef9f7771569)

### cnn_2
![cnn_2](https://github.com/user-attachments/assets/24117d6f-e464-4ca7-b35a-acbd6c2266e2)

### deep_cnn_2
![deep_cnn_2](https://github.com/user-attachments/assets/67e052d5-b8d9-4d88-8b3f-e8ee77ca93e0)

### alightnet
![alightnet](https://github.com/user-attachments/assets/e2f317ee-6aad-4d8c-ab51-d906a99c38de)

### opt_flow
![opt_flow_cnn](https://github.com/user-attachments/assets/e88edc61-8cfd-490b-ab3f-32e4ddeabb3f)

### two_stream
![two_stream_cnn](https://github.com/user-attachments/assets/9312359f-b134-405e-8b24-ad03a560fccc)

## User interface (Qt/PySide)

### Settings windows

![SettingsWindows-HAR](https://github.com/user-attachments/assets/d6b36e64-4d67-44d6-9f0a-47e6860cd000)

### Prediction windows

![ResultsWindows-HAR](https://github.com/user-attachments/assets/925bf647-dfd5-4212-807e-b35e36d66ff9)

## Performance


### Accuracy and Loss
| Dataset         | Model         | Top-1 Validation Accuracy      | Top-1 Validation Loss      |
| :--------       | :-------      | :---------                     |:-----                      |
| `Stanford40`    | `cnn`         | 0.421                          | 1.759                      |
| `Stanford40`    | `cnn_2`       | 0.473                          | 1.649                      |
| `Stanford40`    | `deep_cnn_2`  | 0.476                          | 1.720                      |
| `Stanford40`    | `alightnet`   | 0.348                          | 1.942                      |
| `HMDB51`        | `opt_flow`    | 0.244                          | 2.188                      |
| `HMDB51`        | `two_stream`  | 0.411                          | 1.892                      |

### cnn accuracy
![cnn_acc](https://github.com/user-attachments/assets/c83f7afb-95e4-46ae-8896-fdc2e21ff910)

### cnn loss
![cnn_loss](https://github.com/user-attachments/assets/bd5efe81-5b8b-4a1b-a060-d6425ec6a9a8)

### cnn_2 accuracy
![cnn_2_acc](https://github.com/user-attachments/assets/fb709de8-87b3-4920-82ba-e052770a8d6b)

### cnn_2 loss
![cnn_2_loss](https://github.com/user-attachments/assets/42c5ab43-7c60-4762-b71d-7dbd7cf7ef26)

### deep_cnn_2 accuracy
![deep_cnn_2_acc](https://github.com/user-attachments/assets/63e35402-ebe9-4714-bff7-b9533060a83d)

### deep_cnn_2 loss
![deep_cnn_2_loss](https://github.com/user-attachments/assets/c782500a-b7b5-4046-b811-adc9e3dbff1c)

### alightnet accuracy
![alightnet_acc](https://github.com/user-attachments/assets/613d6ca8-e21b-486a-ad6b-ec6832234f18)

### alightnet loss
![alightnet_loss](https://github.com/user-attachments/assets/f8c4b20d-a47d-46dc-8f9a-a875df716fbd)

### opt_flow accuracy
![opt_flow_cnn_acc](https://github.com/user-attachments/assets/d9092a2d-f8ab-4b03-9121-e296a2ab9ab4)

### opt_flow loss
![opt_flow_cnn_loss](https://github.com/user-attachments/assets/514be6be-f390-4f0b-ad6a-850c0821fda4)

### two_stream accuracy
![two_stream_cnn_acc](https://github.com/user-attachments/assets/1d15ff66-49cc-4b73-9f93-a8f4de90957b)

### two_stream loss
![two_stream_cnn_loss](https://github.com/user-attachments/assets/c140e88d-c7c7-4d35-b0d8-02004c6c6115)

## Reference
HMD51 Dataset:
H. Kuehne, H. Jhuang, E. Garrote, T. Poggio, and T. Serre. HMDB: A Large Video Database for Human Motion Recognition. ICCV, 2011.

Stanford40 Dataset:
B. Yao, X. Jiang, A. Khosla, A.L. Lin, L.J. Guibas, and L. Fei-Fei. Human Action Recognition by Learning Bases of Action Attributes and Parts. Internation Conference on Computer Vision (ICCV), Barcelona, Spain. November 6-13, 2011
