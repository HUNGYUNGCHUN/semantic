from datasets import load_dataset

# Load the Coupang Product Set 1M dataset
dataset = load_dataset('clw8998/Coupang-Product-Set-1M')

# Display the first few rows of the dataset
for i in range(5):
    print(dataset['train'][i])
