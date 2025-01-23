
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

Run the manager.py script and experiment with the different models
```bash
python manager.py
```

## Performance
The following result present the performance obtained with the different models. Accuracy and loss can still be improved as below 50% is a low result but decent on 12 categories (clap, climb, drink, jump, pour, ride_bike, ride_horse, run, shoot_bow, smoke, throw, wave). The biggest limitation to further improvements is the target hardware (consummer-grade hardware) so the limited size of the different dataset and the come that come from such a limited sample is the biggest limitations. Overall this is an exploration of what ML is able to achieve in more concrete setup than the usual MNIST hand-written digit with a decent laptop.

### Original result w/ different model architectures
| Dataset         | Format        | Top-1 Validation Accuracy      | Top-1 Validation Loss      |
| :--------       | :-------      | :---------                     |:-----                      |
| `Stanford40`    | `Frames`      | 0.434                          | 0.571                      |
| `HMDB51`        | `Frames`      | 0.258                          | 1.903                      |
| `HMDB51`        | `Opt. Flow`   | 0.181                          | 2.334                      |
| `HMDB51`        | `Both`        | 0.327                          | (Missing)                  |

### Accuracy and Loss
| Dataset         | Model         | Format        | Top-1 Validation Accuracy      | Top-1 Validation Loss      |
| :--------       | :-------      | :-------      | :---------                     |:-----                      |
| `Stanford40`    | `cnn`         | `Frames`      | 0.421                          | 1.759                      |
| `Stanford40`    | `cnn_2`       | `Frames`      | 0.503                          | 1.617                      |
| `HMDB51`        | `cnn_2`       | `Frames`      | 0.316                          | 1.972                      |
| `Stanford40`    | `deep_cnn_2`  | `Frames`      | 0.476                          | 1.720                      |
| `Stanford40`    | `alightnet`   | `Frames`      | 0.348                          | 1.942                      |
| `HMDB51`        | `opt_flow`    | `Opt. Flow`   | 0.244                          | 2.188                      |
| `HMDB51`        | `opt_flow_2`  | `Opt. Flow`   | 0.308                          | 2.186                      |
| `HMDB51`        | `two_stream`  | `Both`        | 0.419                          | 1.820                      |

## User interface (Qt/PySide)

### Settings windows

![SettingsWindows-HAR](https://github.com/user-attachments/assets/d6b36e64-4d67-44d6-9f0a-47e6860cd000)

### Prediction windows

![ResultsWindows-HAR](https://github.com/user-attachments/assets/925bf647-dfd5-4212-807e-b35e36d66ff9)

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

## Reference
HMD51 Dataset:
H. Kuehne, H. Jhuang, E. Garrote, T. Poggio, and T. Serre. HMDB: A Large Video Database for Human Motion Recognition. ICCV, 2011.

Stanford40 Dataset:
B. Yao, X. Jiang, A. Khosla, A.L. Lin, L.J. Guibas, and L. Fei-Fei. Human Action Recognition by Learning Bases of Action Attributes and Parts. Internation Conference on Computer Vision (ICCV), Barcelona, Spain. November 6-13, 2011
