# Polish Sign Language  recognition

## What is it?
It is algorithm to detect 27 Polish Sign Language (PSL) static words from images.
In static words only hand shape is important - we do not care about movement.

## Results
95.70% accuracy, 95.62% recall and 95.86% precision - test on "unknown" hand.

## Used technologies:
- Keras
- Mediapipe
- Numpy
- Pandas
- sklearn

## How does it work?
Firstly, MediaPipe Holistic Solution detects characteristic points of hands and
pose. In next steps only hands, elbows and shoulders points will be used. Then 
Euclidean distances between every point are calculated which are input for Neural
Network.

## Which do signs be detected?
1. Digits (0-9)
2. Letters (a, b, c, e, i, l, m, n, o, s, t, v, w, x, y) 
3. Words (ja (I in Polish), ty (you in Polish), on/ona (he/she in Polish)

### Disclaimer
"0" and "o" is the same sign. "ty" means also "to jest" towards people 
(he is/she is). "on/ona" means also "to jest" towards things (it is).

## Modules

augmentation.py - functions to make augmentation with flip, mirroring and resize.

custom_exceptions.py - exceptions customized for project.

custom_types.py - types customized for project.

img_management.py - functions to make csv files with saved landmarks coordinates and distances.

name_files.py - functions to rename photos and move to database.

preprocessing.py - functions to make vectors and compute distances.

utils.py - different functions to operate functions.

## How to use?
Easy way to start using - **soon!**

## Demo

![](demo.avi)

Article about the algorithm: https://drive.google.com/file/d/1BiQ1X0OU98suErqC_kA631PraXG-Kh9m/view