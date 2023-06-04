# Mushrooms Clasification

## Introduction
This is a built-from-scratch deep learning project classifying mushrooms classification. \
There are 9 classes considered: Suillus, Cortinarius, Russula, Entoloma, Amanita, Hygrocybe, Lactarius, Agaricus, Boletus.

Link data: https://www.kaggle.com/datasets/lizhecheng/mushroom-classification

## Technical Overview
The model is an implementation of ResNet50 using PyTorch. 
For detailed architecture, please have a look on: https://arxiv.org/abs/1512.03385 \
I have also built Vision Transformer and it can work well. However, I decided not to use it due to scripting problem.
There might be some modification in the future. 

About the repo structure, there are 4 main folders:
- model contains the architecture and scripted model 
- streamlit contains the web interface 
- tools contains code to train, inference and script model
- utils contains supported functions and classes 

I tried to use DVC to save weight, scripted model and other data but due to drive authentication, it is not usable at the moment, 
so you need to download the scripted model by yourself.

The docker is not completely built, but there will be some updates in the future.

## Download
To run it, you need to download the weight (and some sample images incase you need it):\
Scripted model: https://drive.google.com/drive/folders/1PRJcGUHMfbKYekd1os8C5ORzFMMUMj38?usp=sharing \
Data: https://drive.google.com/drive/folders/1Ijk4i3yFf3YsG8q-4gqCO1gg3q8uZu3r?usp=sharing

To use the scripted model, modify and paste it into `model/` folder, it will be structured like this:
```
model
├── scripted_model
│   ├── resnet50
│   │   ├── class_names.txt
│   │   └── scripted_model.pt
│   ├── resnet.py
│   └── vit.py
│
```

In case the Drive requires you to have permission to access, please contact me.

## Environment
I personally recommend using conda to create environment because PyTorch can work differently on CPU only and on GPU version. \
To install environment using CPU only, run:
```
conda env create -f environment_cpu.yaml
```

To install environment using GPU, run:
```
conda env create -f environment_gpu.yaml
```

To create environment by pip, run:
```
pip install -r requirements.txt
```
Please notice that you should you Python>=3.7. To make sure, my python version is 3.8.16.


## How to run
To inference the model, make sure that you change the path `origin_path` to your image/folder path, then run:
``` 
python tools/infer.py
```

Mlchain is a library helps user to quickly deploy Machine Learning model to hosting server easily and efficiently.
To run MLChain server, run:
``` 
mlchain run -c mlconfig.yaml 
```
An API server will be hosted at http://0.0.0.0:8001 \
You can click on Swagger on the top right corner, then move to Mlchain Format APIs: 
- To test model using path, click on function /call/predict, then choose "Try it out": 
    + Type in the path (and the batch_size in case your path belongs to an image folder)
    + Click Execute
- To test model with uploaded image, click on function /call/predict_from_image and choose "Try it out": 
    + Browse the image you want to try, then TYPE "True" in "use_cv2" part.
    + Click Execute
- After try the function, you will get the Curl or Request URL to import to your application by third-party tool like Postman. 

## How to train
To train the model, make sure to change the path `data_path` to your dataset path. \
You can also change the parameter in `train_task.train(n_epochs=10,save_name="test", eval_interval=1)` by your self:
- n_epochs is the number of epoch training
- save_name is the name of the folder which saves tensorboard log, weight and class names file. 
- eval_interval: the model evaluate after that number of epochs
To train, please run:
```
python tools/train.py
```
The training result then will be in the folder `runs`.


## Streamlit
Streamlit is a free, open-source, all-python framework that enables data scientists to quickly build interactive dashboards and machine learning web apps.
If you want test with streamlit, let's create a new terminal, and run:
```
streamlit run streamlit/streamlit.py
```
Hosted at http://0.0.0.0:8001

---

If you have any question or encouter any problem regarding this repository. Please open an issue and cc me. Thank you.







