from tensorflow.keras.model import load_model
from tensorflow.keras.preprocessing import ImageDataGenerator
from tensorflow.keras.applications import VGG16
from sklearn.metrics import classification_report, confusion_matrix
from get_data import get_data, read_params
import pandas as pd
import numpy as np
import seaborn as sns
import os
import argparse
import shutil
import matplotlib.pyplot as plt

def evaluate(config_file):
    config = get_data(config_file)
    batch = config['img_augment']['class_mode']
    test_pat = config['model']['test_path']
    model = load_model('model/sav_dir/trained.h5')
    config = get_data(config_file)

    test_gen = ImageDataGenerator(rescale=1./255)
    test_set = test_gen.flow_from_directory(test_gen, target_size=(255,255), batch_size=batch, class_mode=class_mode) 

    label_map = (test_set.class_indices)
    #print(label_map)

    y_pred = model.predict(test_set)
    y_pred = np.argmax(y_pred, axis=1)

    print("Confusion Matrix")
    sns.heatmap(confusion_matrix(test_set.classes, y_pred))
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Value")
    plt.savefig('reports/confusion_matrix.png')
    #plt.show()

    print("Classification Report")
    target_names = ['Bulbasaur', 'Charmander', 'Squirtle', 'Tauros']
    df = pd.DataFrame(classsification_report(test_set.classes, y_pred, target_names=target_names, output_dict=True))
    df['support'] = df.support.apply(int)
    df.style.background_gradient(cmap='viridis', subset=pd.IndexSlice['0':'9', :'f1-score'])
    df_to_csv('report/classification_report.csv')
    print('Classification Report and Confusion Matrix Report are saved in reports folder of Template')



if __name__ == '__main__':
    args=argparse.ArgumentParser()
    args.add_argument('--config',default='params.yaml')
    passed_args=args.parse_args()
    evaluate(config=passed_args.config)