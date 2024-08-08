import os, sys
import shutil
from tools import stochastic_profile_steering

# Define tha data folder
data_folder = 'EC_with_3households_data_15min'

# Define the time_data and agents_data files
time_data_file = f'input_data/{data_folder}/time_data.json'
agents_data_file = f'input_data/{data_folder}/agents_data.json'
parameters_file = f'input_data/{data_folder}/parameters.json'

# Create Estudos/ if not exists
if not os.path.exists('results'):
    os.makedirs('results')

# Remove .txt and .png files in Estudos/
dir_name = "results/"
test = os.listdir(dir_name)
for item in test:
    if item.endswith(".png"):
        os.remove(os.path.join(dir_name, item))

if shutil.which("ipopt") or os.path.exists('ipopt'):
    stochastic_profile_steering.stochastic_profile_steering(time_data_file, agents_data_file, parameters_file)
else:
    print('Warning: the solver ipopt has not been found or cannot be called')
