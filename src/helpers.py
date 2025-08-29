import yaml
import os
import matplotlib.pyplot as plt


def extract_yml():
    with open("src/conf.yml") as stream:
        try:
            file = yaml.safe_load(stream)
            return file
        except yaml.YAMLError as exc:
            return print(exc)

def plot(history,placement:str):
    plt.plot(history.history['accuracy'][1:], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    # Loss
    plt.plot(history.history['loss'][1:], label='train loss')
    plt.plot(history.history['val_loss'], label='val')
    plt.xlabel('number of Epoch')
    plt.ylabel('losses during training')
    plt.legend(loc=placement)
    return plt.show()

if __name__=="__main__":
    print('Helpers')

