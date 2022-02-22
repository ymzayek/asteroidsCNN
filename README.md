# Asteroids CNN
CNN for detecting asteroid images from the Kilo-Degree Survey (KiDS) 

Based on https://github.com/MikkoPontinen/EuclidCNN

---

## Training 
File trainSimpleCNN.py can be used to train a model from scratch. From the command line, the following arguments need to be set:
* --image_path \<path-to-image-files>
* --model_out  \<path-to-save-model>
* --plots_out  \<path-to-save-plots> [optional]

### Example:
```
python3 trainSimpleCNN.py --image_path /data/pg-ds_cit/Projects/Astronomy/AstronomyProject/Images --model_out /data/p301081/astronomy/test --plots_out /data/p301081/astronomy/test
```

A training example can be found in the notebook trainSimpleCNN.ipynb

The training script also includes testing model performance with a test set.


