import os, sys
import shutil
from tools import stochastic_profile_steering

# Define the time_data and agents_data files
time_data_file = 'input_data/EC_with_3households_data_15min/time_data_15min.json'
agents_data_file = 'input_data/EC_with_3households_data_15min/agents_data_15min.json'
parameters_file = 'input_data/EC_with_3households_data_15min/parameters_15min.json'
# time_data_file = 'input_data/EC_with_3households_data_1hour/time_data_1hour.json'
# agents_data_file = 'input_data/EC_with_3households_data_1hour/agents_data_1hour.json'
# parameters_file = 'input_data/EC_with_3households_data_1hour/parameters_1hour.json'

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
