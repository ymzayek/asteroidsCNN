# Asteroids CNN
CNN for detecting asteroid images from the Kilo-Degree Survey (KiDS) 

Based on https://github.com/MikkoPontinen/EuclidCNN

---

## Training 
File trainSimpleCNN.py can be used to train a model from scratch. From the command line, the following arguments need to be set:
* --image_path \<path-to-image-files>
* 

```
trainSimpleCNN.py --image_path /data/pg-ds_cit/Projects/Astronomy/AstronomyProject/Images
```

Training can also be done inside the notebook trainSimpleCNN.ipynb

The training script also includes testing model performance with a test set.



---
## Inference

