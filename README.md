# instru-vision

## Musical instrument classifier - project for Soft Computing course, SIIT, FTN, UNS

![Research project poster](/img/poster.png "Research project poster")

## Demo

Demo app made with React and Tensorflow.js can be found [here](https://ikkjo.github.io/instru-vision-app/).

## Description

This project contains two different [`TensorFlow`](https://www.tensorflow.org/) machine learning models for image classification that were trained on a dataset of ~1200 images of different types of instruments. A demo web app with both models made with node.js and React that predicts an image provided from an upload or an image url is available.

## Requirements
    python: 3.10
    node: 16.17.1

### **Python**

    tensorflow==2.10
    tensorflow-datasets==4.6.0
    tensorflow-addons==0.18.0
    keras==2.10
    opencv-python==4.6.0
    matplotlib==3.6.0
    numpy==1.23.0
    keras-tuner==1.1.3
    scikit-learn==1.1.2
    jupyter==1.0.0
    selenium==4.4.3
    Pillow==9.2.0

### **Node.js**

Node dependencies can be found in `js/instrument-classification-app/package.json`

## Installation

### **Python**

Install packages from dependency list with `pip install <package-name>==<version>` or copy/paste the dependencies in a text file and use `pip install -r <depencency-file-name>.txt`

### **Node.js**

    cd js/instrument-classification-app
    npm install
    npm run build

### Conversion to TensorFlow.js

Conversion to `TensorFlow.js` was done using `tensorflowjs_converter` that comes built in with the `tensorflowjs` Python library. A batch script for using `tensorflowjs_converter` was made for ease of use.

## How to start the app

To use the web app, you must first run the Node.js server using the included script: `./scripts/run_webapp.sh`, and then connect to http://localhost:3000 with any browser.
