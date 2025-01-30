import pickle

# Path to your .pkl file
file_path = '/research/iprobe-farmanif/openiris-sam-final/checkpoints/focal_casia_multi_cls/gamma_2.0/epoch_loss_history.pkl'

# Load the data
with open(file_path, 'rb') as file:
    loss_data = pickle.load(file)

import matplotlib.pyplot as plt

# Assuming loss_data is a list of loss values
plt.figure(figsize=(10, 5))
plt.plot(loss_data, label='Training Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss.png')
