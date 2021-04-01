import numpy as np
import keras 
from PIL import Image
import argparse
import parser
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf



def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def ImagePreProcessing(image_path):
    img = Image.open(image_path)
    img = img.resize((48,48))
    img_array = np.asarray(img)
    baw_img = rgb2gray(img_array).astype(int)
    final_img = baw_img.reshape((48,48,1))
    return final_img

def pred_to_decision(pred):
    if pred>=0.5:
        M_F = 'Female'
    else:
        M_F = 'Male'
    return M_F

def decision_to_prob(prediction,decision):
    if decision=='Male':
        p = (1-prediction)*100
    else:
        p=prediction*100
    return p

def main():
    
    args = parser.parse_args()
    im_path = args.image_path
    model_path = args.model_path
    image = ImagePreProcessing(im_path)
    model = keras.models.load_model(model_path)
    prediction = model.predict(np.array([image]))[0][0]
    #print(prediction)
    decision = pred_to_decision(prediction)
    print('The prediction for the input image is: ' + str(decision)
          + ' with probability %.2f' %(decision_to_prob(prediction=prediction,
                                                  decision=decision
                                                  )) + '%')
    return decision
if __name__=='__main__':
    
    parser = argparse.ArgumentParser(formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--image_path',type=str,help = 'Path of the png image'),
    parser.add_argument('--model_path',type=str,help = 'Path of the model'),
    
    main()
    
