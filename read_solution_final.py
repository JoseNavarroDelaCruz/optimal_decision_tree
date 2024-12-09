# --- TRAIN 

import re
import numpy as np
import pickle
import os

# Extract matrices and vectors from the log file
def extract_matrices_and_vectors(problem_path):

    with open(problem_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    log_text = ''.join(lines)

    dimension_pattern = r'dimension: (\d+), class: (\d+)'
    dimension_match = re.search(dimension_pattern, log_text)
    if dimension_match:
        num_rows = int(dimension_match.group(1))
        num_cols = int(dimension_match.group(2))
    else:
        raise ValueError("Dimensions not found in the log file.")

    matrix_pattern = r'(\d+)×(\d+) Matrix{Float64}:([\s\S]+?)(?=\n\d|\ncomparison|\n\d+-element)'
    vector_pattern = r'(\d+)-element Vector{Float64}:([\s\S]+?)(?=\n\d|\ncomparison|\n7×|\n3×)'

    matrices = re.findall(matrix_pattern, log_text)
    vectors = re.findall(vector_pattern, log_text)

    def parse_matrix(matrix_data):
        rows, cols, values = matrix_data
        rows, cols = int(rows), int(cols)
        values = np.fromstring(values.replace('\n', ' ').replace('  ', ' '), sep=' ')
        return values.reshape((rows, cols))

    def parse_vector(vector_data):
        size, values = vector_data
        values = np.fromstring(values.replace('\n', ' '), sep=' ')
        return values

    matrices_extracted = [parse_matrix(matrix) for matrix in matrices]
    vectors_extracted = [parse_vector(vector) for vector in vectors]

    variable_names = ['variable_a', 'variable_b', 'variable_c', 'variable_d']
    
    warm_start = {}
    final_solution = {}
    
    if len(matrices_extracted) > 0 and len(vectors_extracted) > 0:
        warm_start[variable_names[0]] = matrices_extracted[0] if len(matrices_extracted) > 0 else None
        warm_start[variable_names[1]] = vectors_extracted[0] if len(vectors_extracted) > 0 else None
        warm_start[variable_names[2]] = matrices_extracted[1] if len(matrices_extracted) > 1 else None
        warm_start[variable_names[3]] = vectors_extracted[1] if len(vectors_extracted) > 1 else None
    
    if len(matrices_extracted) > 2 and len(vectors_extracted) > 2:
        final_solution[variable_names[0]] = matrices_extracted[2] if len(matrices_extracted) > 2 else None
        final_solution[variable_names[1]] = vectors_extracted[2] if len(vectors_extracted) > 2 else None
        final_solution[variable_names[2]] = matrices_extracted[3] if len(matrices_extracted) > 3 else None
        final_solution[variable_names[3]] = vectors_extracted[3] if len(vectors_extracted) > 3 else None

    return warm_start, final_solution

# Extract time, ub, lb, and gap from the log file
def extract_values_from_segment(segment):
    lines = segment.strip().split("\n")
    values_line = lines[1]  
    values = re.split(r'\s+', values_line)
    time = float(values[1])  
    ub = float(values[2])    
    lb = float(values[3])    
    gap = float(values[4])   
    return {'time': time, 'ub': ub, 'lb': lb, 'gap': gap}

# Process the file and extract solution data
def process_solution_file(problem_path):
    with open(problem_path, 'r') as file:
        file_content = file.read()
    
    lines = file_content.strip().split("\n")
    extracted_info = []
    
    for i in range(len(lines)):
        if "Dataname" in lines[i]:
            segment = "\n".join(lines[i:i+3])
            extracted_values = extract_values_from_segment(segment)
            extracted_info.append(extracted_values)
    
    return extracted_info

# Normalize the training data
def normalize_data(training_data):
    data_min = np.min(training_data[:, :-1], axis=0)
    data_max = np.max(training_data[:, :-1], axis=0)
    normalized_data = (training_data[:, :-1] - data_min) / (data_max - data_min)
    return normalized_data


# Predict leaf nodes for a sample
def predict(sample, feature_indices, variable_b):
    current_node = 0  
    feature_idx = feature_indices[current_node]
    split_value = variable_b[current_node]  

    if sample[feature_idx] >= split_value:
        current_node = 2
    else:
        current_node = 1

    if current_node == 1:  
        if sample[feature_indices[1]] >= variable_b[1]:
            final_leaf_node = 1  
        else:
            final_leaf_node = 0  
    else:  
        if sample[feature_indices[2]] >= variable_b[2]:
            final_leaf_node = 3  
        else:
            final_leaf_node = 2  

    return final_leaf_node + 1


# One-hot encoding of the predicted leaf nodes
def leaf_node_encoding(leaf_node, num_leaves=4):
    encoding = np.zeros(num_leaves)
    encoding[leaf_node - 1] = 1  
    return encoding


# Predict all samples
def predict_all_samples(normalized_data, feature_indices, variable_b):
    predicted_leaf_nodes = []
    for sample in normalized_data:
        leaf_node = predict(sample, feature_indices, variable_b)
        one_hot_leaf_node = leaf_node_encoding(leaf_node)
        predicted_leaf_nodes.append(one_hot_leaf_node)
    return np.array(predicted_leaf_nodes)


# Convert real leaf values to one-hot encoding
def convert_real_leaf_values_to_one_hot(real_leaf_values):
    # Determine the number of leaves by finding the max value in real_leaf_values
    min_label = int(np.min(real_leaf_values))
    max_label = int(np.max(real_leaf_values))
    num_leaves = max_label - min_label + 1
    
    # Initialize the one-hot encoded matrix
    one_hot_real_leafs = np.zeros((real_leaf_values.shape[0], num_leaves))
    
    # Populate the one-hot encoding based on real_leaf_values
    for i, leaf in enumerate(real_leaf_values):
        one_hot_real_leafs[i, int(leaf) - min_label] = 1  # Adjust for potential non-zero-based labels
    
    return one_hot_real_leafs



# Compute the total cost L
# Compute the prediction cost per sample (1 if prediction is incorrect, 0 if correct)
def compute_prediction_cost(predicted_leaf_nodes, real_leaf_values_one_hot, variable_c):
    # Extract the leaf node labels from the last four columns of variable_c
    leaf_labels = np.argmax(variable_c[:, -4:], axis=0) + 1  # Add 1 to get actual labels (1-based indexing)

    # Determine the predicted label for each sample based on predicted_leaf_nodes
    # Each row in predicted_leaf_nodes corresponds to an observation and each column corresponds to a leaf node
    predicted_labels = np.argmax(predicted_leaf_nodes, axis=1)  # Get the leaf node index (0-based)

    # Map the leaf node index to the actual label using the extracted leaf_labels
    predicted_classifications = leaf_labels[predicted_labels]

    # Convert the real one-hot labels back to class labels
    real_classifications = np.argmax(real_leaf_values_one_hot, axis=1) + 1  # Add 1 to get 1-based labels

    # Compare predicted_classifications with real_classifications to determine cost (0 if correct, 1 if wrong)
    sample_costs = (predicted_classifications != real_classifications).astype(int)

    return sample_costs



def extract_time_from_file(file_path):
    """
    Extracts the `time` value from the provided text file.

    Args:
        file_path (str): Path to the text file.

    Returns:
        float: Extracted time value.
    """
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("Dataname"):
                # The next line contains the desired data
                next_line = next(file).strip()
                # Split and extract the time value (2nd column)
                time_value = float(next_line.split("\t")[1])
                return time_value
    raise ValueError("Time value not found in the file.")



# Main function that calls all the others and organizes the results
def generate_solution_data(problem_path, test_data_path):
    warm_start, final_solution = extract_matrices_and_vectors(problem_path)
    
    solution_data = {}
    solution_data['warm_start'] = warm_start
    solution_data['final_solution'] = final_solution
    
    # Extract additional solution data (time, ub, lb, gap)
    solution_data['extracted_info'] = process_solution_file(problem_path)
    
    # Read the test data file
    test_data = process_file_to_numpy(test_data_path)
    
    if test_data is None:
        raise ValueError(f"Failed to process test data from {test_data_path}")

    # Normalize data for the second-stage prediction
    normalized_data = normalize_data(test_data)
    
    # Input the normalized data as variable_x
    variable_x = normalized_data
    solution_data['variable_x'] = variable_x
    
    # Use the final_solution matrices to predict leaf nodes
    feature_indices = np.argmax(final_solution['variable_a'], axis=0)
    predicted_leaf_nodes = predict_all_samples(normalized_data, feature_indices, final_solution['variable_b'])
    
    # Store the second-stage variable z (predicted leaf nodes)
    solution_data['variable_z'] = predicted_leaf_nodes
    
    # Convert real leaf values to one-hot encoding for cost computation
    real_leaf_values = test_data[:, -1]
    real_leaf_values_one_hot = convert_real_leaf_values_to_one_hot(real_leaf_values)
    
    # Store the one-hot encoded real labels as 'variable_y'
    solution_data['variable_y'] = real_leaf_values_one_hot
    
    # Compute the total cost L
    total_cost = compute_prediction_cost(predicted_leaf_nodes, real_leaf_values_one_hot, final_solution['variable_c'])
    solution_data['variable_L'] = total_cost 

       # Extract the time value from the problem file and add to solution_data
    time_value = extract_time_from_file(problem_path)
    solution_data['time'] = time_value
    
    return solution_data


# Function to read file and process data
def process_file_to_numpy(file_path):
    try:
        # Read file content
        with open(file_path, 'r') as file:
            data = file.read()

        # Split the data into rows and convert to list of lists
        data_rows = [list(map(float, line.split(','))) for line in data.strip().split('\n')]

        # Convert list of lists into a NumPy array
        test_data = np.array(data_rows)

        return test_data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None







# ---------------- TEST CODE -------------------------- #
problem_file_path = "seeds-sd1-2-CMS-.out"
problem_path = "seeds-sd1-2-CMS-.out"
#problem_file_path = "small_toy-sd1-2-CMS-.out"

test_file_path = "data/seeds"
#test_file_path = "data/small_toy"

solution_data = generate_solution_data(problem_file_path, test_file_path)
print(solution_data)


# Path to save the solution data
save_directory = "/Users/navarrodelacruz/OneDrive - University of South Florida/USF/PROJECTS/neural_diving_pytorch"
save_filename = "solution_data.pkl"
# Create the directory if it doesn't exist
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Full path to save the solution data
save_path = os.path.join(save_directory, save_filename)

# Save the solution_data object to the specified directory
with open(save_path, 'wb') as file:
    pickle.dump(solution_data, file)