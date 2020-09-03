import numpy as np

from keras.preprocessing import image
from keras.applications.densenet import DenseNet121
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

import util

labels = ['Cardiomegaly', 
          'Emphysema', 
          'Effusion', 
          'Hernia', 
          'Infiltration', 
          'Mass', 
          'Nodule', 
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening', 
          'Pneumonia', 
          'Fibrosis', 
          'Edema', 
          'Consolidation']

# create the base pre-trained model
base_model = DenseNet121(weights='densenet.hdf5', include_top=False)

x = base_model.output

# add a global spatial average pooling layer
x = GlobalAveragePooling2D()(x)

# and a logistic layer
predictions = Dense(len(labels), activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# load pretrained weights
model.load_weights("pretrained_model.h5")

def load_image(img_path, preprocess=True, H=320, W=320):
    """Load and preprocess image."""
    mean = np.float64(134.5683984375)
    std = np.float64(64.79917887884459)
    x = image.load_img(img_path, target_size=(H, W))
    if preprocess:
        x -= mean
        x /= std
        x = np.expand_dims(x, axis=0)
    return x

# get sample image
img = '000001-3.png'
IMAGE_DIR = ''
img_path = IMAGE_DIR + img

# preprocess input image
preprocessed_input = load_image(img_path)

# get predictions probabilites
probs = model.predict(preprocessed_input)

# match probs with labels and sort them
readable_predictions = sorted(zip(labels, map(float, probs[0])), key=lambda p: p[1], reverse=True)
print(readable_predictions)