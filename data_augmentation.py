import os
import numpy as np
import pandas as pd 

# Define the source directory
data_dir = r"C:\Users\Owner\Desktop\schoolstuff\research\optimal_decision_tree-main_13\optimal_decision_tree-main\data"
output_dir = data_dir  # Save augmented files in the same directory

# Debugging: Confirm directory exists
if not os.path.exists(data_dir):
    print(f"Error: The directory {data_dir} does not exist!")
    exit()

# Define parameters
num_iterations = 1000  # Number of variations per file
noise_std_dev = 0.05   # Standard deviation for Gaussian noise
removal_fraction = 0.02  # Fraction of rows to remove per iteration

# Get list of all files (any type) in the data directory
all_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]

# Debugging: Print found files
if not all_files:
    print("No files found in the directory!")
else:
    print(f"Found {len(all_files)} file(s): {all_files}")

# Process each file in the data directory
for filename in all_files:
    file_path = os.path.join(data_dir, filename)
    
    try:
        # Try reading the file as a CSV/TXT
        df = pd.read_csv(file_path, header=None, delimiter=None, engine='python')

        # Debugging: Print the first few rows
        print(f"\nProcessing file: {filename}")
        print(df.head())

        # Identify the number of columns (assuming last one is the label)
        num_features = df.shape[1] - 1

        for i in range(1, num_iterations + 1):
            # Copy the dataframe
            df_aug = df.copy()

            # Apply Gaussian noise to feature columns
            noise = np.random.normal(0, noise_std_dev, df_aug.iloc[:, :-1].shape)
            df_aug.iloc[:, :-1] += noise

            # Randomly remove a fraction of rows
            num_rows_to_remove = int(len(df_aug) * removal_fraction)
            if num_rows_to_remove > 0:  # Ensure we don't try to remove more rows than exist
                rows_to_remove = np.random.choice(df_aug.index, num_rows_to_remove, replace=False)
                df_aug = df_aug.drop(rows_to_remove)

            # Save the modified file (same extension as original)
            file_ext = filename.split('.')[-1]  # Get original file extension
            new_filename = f"{filename.split('.')[0]}_{i}.{file_ext}"
            output_path = os.path.join(output_dir, new_filename)
            df_aug.to_csv(output_path, index=False, header=False)

            # Debugging: Confirm file saved
            if i % 100 == 0:
                print(f"Saved: {output_path}")

    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("\nData augmentation complete!")
