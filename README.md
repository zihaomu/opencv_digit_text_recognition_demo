# Digit&Text detection and recognition
[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

### My environment
1. hardware
   `cpu: i5-8300H 8-core + 16G RAM + 1050 GPU`
2. software
   `Ubuntu 18 + Opencv 4.3`

### Digit detection and recognition

 Very easy demo, based on opencv dnn c++. Digital detector uses connected components. Since the pre-processing process is simple, the model is sensitive to light changes. It can achieve about 13FPS on my laptop.

 There are many things can be done to improve its performance, like using good image preprocessing or deep-learning-based digit detector.

 hardware：
 `cpu: i5-8300H 8-core + 16G RAM`
 
model:

lenet.caffemodel

lenet.prototxt

DEMO:
![image](./demo/demo.gif)

### Text detection and recognition
