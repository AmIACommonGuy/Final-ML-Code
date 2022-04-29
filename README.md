# Final-ML-Code
Download RGB dataset: https://drive.google.com/drive/folders/14p5pHFEjjhjkb2wI5H63rr2i4KYNQ_Xb
Download RGB + nir dataset: https://drive.google.com/file/d/1Tz92S2YnJ0JN7cmTR9HBbWfP5ouDjY-g/view

Name them RGB_Dataset.zip and RGB+IR_Dataset.zip respectively.

run ipynb files to see results:
4 Bands: https://colab.research.google.com/drive/1vv7qxwJaRFeKLp3y_5RXFcaz6x2YYenu?usp=sharing
3 Bands: https://colab.research.google.com/drive/1NZs7B77lykQMJNr5kfe30cpR3LfcY55c?usp=sharing


### LIME implementation
To run the lime implementation you will use the `main_lime.py`. In that file there is a variable on line 56 
```
RGB_or_IR = 'IR' # 'RGB'
```
Here you toggle if you want to use the 4channel NIR implmentation or the 3channel RGB implementation. The output will be a list of River and Forest images with the respective explainations for each channel. Verify that the dataset is in the current directory. The choice of dataset is specified above, `'./RGB Only Dataset' or './RGB+IR Dataset'`, and will be automatically selected based on the RGB_or_IR variable. As will the pretrained model for 3 or for 4 channels, `'model_3.pth' or 'model_4.pth'`

> LIME modifications were done in `lime/lime_image.py`, while `lime/old_lime_image.py` is the original. 
