# Final-ML-Code 

### Download Dataset:
- Download RGB dataset: https://drive.google.com/drive/folders/14p5pHFEjjhjkb2wI5H63rr2i4KYNQ_Xb
- Download RGB+IR dataset: https://drive.google.com/file/d/1Tz92S2YnJ0JN7cmTR9HBbWfP5ouDjY-g/view
- Name them RGB_Dataset.zip and RGB+IR_Dataset.zip respectively.


### SHAP:
- Download `shap2.zip` folder 

Run these `.ipynb` files to see results:
+ 4 Bands: https://colab.research.google.com/drive/1vv7qxwJaRFeKLp3y_5RXFcaz6x2YYenu?usp=sharing
+ 3 Bands: https://colab.research.google.com/drive/1NZs7B77lykQMJNr5kfe30cpR3LfcY55c?usp=sharing


### LIME implementation:
    LIME modifications were done in `lime/lime_image.py`, while `lime/old_lime_image.py` is the original.

#### Setup Steps
+ Verify that the dataset is in the current directory. The choice of dataset is specified above, `'./RGB Only Dataset' or './RGB+IR Dataset'`, and will be automatically selected based on the `RGB_or_IR` variable.

+ Unzip `lime.zip` and put it in the same directory. This is the lime source code with the undated `lime_image.py`

+ This needs to use pretrained models for 3 or 4 channels, so in order to run this, you need to download the models from the colab and save them here, named: `'model_3.pth' or 'model_4.pth'`.

+ The output will be a list of River and Forest images with the respective explainations for each channel, you can see these outputs in images `River_Final.png` and `Forest_Final.png`.

+ To run the lime implementation you will use the `main_lime.py`. In that file there is a variable on line 56, toggle this variable if you want to use the 4channel NIR implmentation or the 3channel RGB implementation.
    ```
    RGB_or_IR = 'IR' # 'RGB'
    ```
