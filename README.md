# Mask Detection

This project uses Transfer Learning to train a Model which classifies with_mask, without_mask and mask_weared_incorrecly. The [Mask Detection Dataset](https://www.kaggle.com/vijaykumar1799/face-mask-detection) from kaggle is used to train the model. The Models used in this projects is MobilenetV2. Accuracy of Model is 95.27%.

## How to run?

You need python installed on your system. You can download and install it from [here](https://www.python.org/). After installing python, follow these steps:

### On Windows:

- Open the `cmd` in the working dictionary.
- Type `python -m venv venv` to create a vitual environment.
- Type `call venv\Scripts\activate.bat` to activate the virtual environment.
- Type `pip install -r requirements.txt` to install all the required dependencies.
- Type `jupyter-lab` to run the jupyter-server.

### On Mac OS/Linux:

- Open the `terminal` in the working dictionary.
- Type `python3 -m venv venv` to create a vitual environment.
- Type `source ./venv/bin/activate` to activate the virtual environment.
- Type `pip3 install -r requirements.txt` to install all the required dependencies.
- Type `jupyter-lab` to run the jupyter-server.


## How to Predict?

Follow the First 4 steps from `How to run?` section. After that, use `python Predict.py [.../path/to/img]` to predict the images.

## How to Predict from camera?

Follow the First 4 steps from `How to run?` section. After that, use `python Predict_live.py` to predict the images.