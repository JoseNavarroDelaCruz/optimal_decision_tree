################################################################
###################### EXTRACT SOLVER DATA #####################
################################################################



import re
import numpy as np

# Read data path
problem_path = "seeds-sd1-2-CMS-.out"



def extract_matrices_and_vectors(problem_path):

    # Open the file for reading with the correct UTF-8 encoding
    with open(problem_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()


    # Combine all lines into a single string
    log_text = ''.join(lines)


    # Extract the dimensions from the file (e.g., 'dimension: 7, class: 3')
    dimension_pattern = r'dimension: (\d+), class: (\d+)'
    dimension_match = re.search(dimension_pattern, log_text)
    if dimension_match:
        num_rows = int(dimension_match.group(1))  # number of rows (7 in the example)
        num_cols = int(dimension_match.group(2))  # number of columns (3 in the example)
    else:
        raise ValueError("Dimensions not found in the log file.")



    # Regex patterns to identify matrices and vectors based on their dimensions
    matrix_pattern = r'(\d+)×(\d+) Matrix{Float64}:([\s\S]+?)(?=\n\d|\ncomparison|\n\d+-element)'
    vector_pattern = r'(\d+)-element Vector{Float64}:([\s\S]+?)(?=\n\d|\ncomparison|\n7×|\n3×)'

    # Find all matrices and vectors
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

    # Extract matrices and vectors
    matrices_extracted = [parse_matrix(matrix) for matrix in matrices]
    vectors_extracted = [parse_vector(vector) for vector in vectors]

    # Assuming two groups of warm start and final solution
    # Assign variable names in order for both warm start and final solution
    variable_names = ['variable_a', 'variable_b', 'variable_c', 'variable_d']
    
    warm_start = {}
    final_solution = {}
    
    # Split extracted data into two groups: warm start and final solution
    mid_point = len(matrices_extracted) // 2
    
    # Assign matrices and vectors to corresponding variables for warm start
    if len(matrices_extracted) > 0 and len(vectors_extracted) > 0:
        warm_start[variable_names[0]] = matrices_extracted[0] if len(matrices_extracted) > 0 else None
        warm_start[variable_names[1]] = vectors_extracted[0] if len(vectors_extracted) > 0 else None
        warm_start[variable_names[2]] = matrices_extracted[1] if len(matrices_extracted) > 1 else None
        warm_start[variable_names[3]] = vectors_extracted[1] if len(vectors_extracted) > 1 else None
    
    # Assign matrices and vectors to corresponding variables for final solution
    if len(matrices_extracted) > 2 and len(vectors_extracted) > 2:
        final_solution[variable_names[0]] = matrices_extracted[2] if len(matrices_extracted) > 2 else None
        final_solution[variable_names[1]] = vectors_extracted[2] if len(vectors_extracted) > 2 else None
        final_solution[variable_names[2]] = matrices_extracted[3] if len(matrices_extracted) > 3 else None
        final_solution[variable_names[3]] = vectors_extracted[3] if len(vectors_extracted) > 3 else None

    return warm_start, final_solution

results=extract_matrices_and_vectors('seeds-sd1-2-CMS-.out')
warm_start_variables=results[0]
final_solution_variables=results[1]

warm_start_variables
final_solution_variables


# ---------------------- Extract final solution data

# Function to extract time, ub (objv), lb, and gap from a segment
def extract_values_from_segment(segment):

    lines = segment.strip().split("\n")
    
    # Extract the relevant values line
    values_line = lines[1]  # Assuming this is always the third line after the headers
    
    # Split the values line based on whitespace
    values = re.split(r'\s+', values_line)
    
    # Extract the specific values
    time = float(values[1])  # Second column is 'time'
    ub = float(values[2])    # Third column is 'objv' (renamed to 'ub')
    lb = float(values[3])    # Fourth column is 'lb'
    gap = float(values[4])   # Fifth column is 'gap'
    
    return {'time': time, 'ub': ub, 'lb': lb, 'gap': gap}

# Main function to process the larger file
def process_solution_file(problem_path):

    # Open the file for reading with the correct UTF-8 encoding
    with open(problem_path, 'r') as file:
        file_content = file.read()
        
    # Split the file content by lines
    lines = file_content.strip().split("\n")
    
    # Initialize a list to store extracted information
    extracted_info = []
    
    # Iterate through the lines and detect relevant segments
    for i in range(len(lines)):
        if "Dataname" in lines[i]:  # Detect the header line
            # Extract the next three lines (header + values) as a segment
            segment = "\n".join(lines[i:i+3])  # Get 3 lines (headers + values)
            
            # Extract the time, ub, lb, and gap from the segment
            extracted_values = extract_values_from_segment(segment)
            
            # Add to the list of extracted info
            extracted_info.append(extracted_values)
    
    return extracted_info


# Process the file content and get the extracted information
extracted_info = process_solution_file(problem_path)

# Print the extracted information for each segment
for entry in extracted_info:
    print(entry)








###########################################################

# ----------- CREATE SECOND STAGE VARIABLES


# Provided matrices and vectors
variable_a = np.array([[ 0.,  0., -0.],
       [ 1.,  0., -0.],
       [ 0.,  0., -0.],
       [ 0.,  0., -0.],
       [ 0.,  0., -0.],
       [ 0.,  1., -0.],
       [ 0., -0.,  1.]])  # 2x3 matrix for node branchings

variable_b = np.array([0.27066116, 0.25171306, 0.5]) # 3-element vector for split values

variable_c = np.array([[ 0.,  0.,  0.,  1.,  0.,  1.,  0.],
       [ 0.,  0.,  0.,  0.,  0.,  0.,  1.],
       [ 0.,  0.,  0.,  0.,  1., -0., -0.]])  # 3x7 matrix for leaf node labels

variable_d = np.array([1.0, 1.0, 1.0])  # 3-element vector for split status



sample = np.array( [15.26, 14.84, 0.871, 5.763, 3.312, 2.221, 5.22, 1])

# Example input data (replace with the training dataset)
training_data = np.array([
    [15.26, 14.84, 0.871, 5.763, 3.312, 2.221, 5.22, 1],
    [14.88, 14.57, 0.8811, 5.554, 3.333, 1.018, 4.956, 1],
    [14.29, 14.09, 0.905, 5.291, 3.337, 2.699, 4.825, 1],
    [13.84, 13.94, 0.8955, 5.324, 3.379, 2.259, 4.805, 1],
    [16.14, 14.99, 0.9034, 5.658, 3.562, 1.355, 5.175, 1],
    [14.38, 14.21, 0.8951, 5.386, 3.312, 2.462, 4.956, 1],
    [14.69, 14.49, 0.8799, 5.563, 3.259, 3.586, 5.219, 1],
    [14.11, 14.1, 0.8911, 5.42, 3.302, 2.7, 5, 1],
    [16.63, 15.46, 0.8747, 6.053, 3.465, 2.04, 5.877, 1],
    [16.44, 15.25, 0.888, 5.884, 3.505, 1.969, 5.533, 1],
    [15.26, 14.85, 0.8696, 5.714, 3.242, 4.543, 5.314, 1],
    [14.03, 14.16, 0.8796, 5.438, 3.201, 1.717, 5.001, 1],
    [13.89, 14.02, 0.888, 5.439, 3.199, 3.986, 4.738, 1],
    [13.78, 14.06, 0.8759, 5.479, 3.156, 3.136, 4.872, 1],
    [13.74, 14.05, 0.8744, 5.482, 3.114, 2.932, 4.825, 1],
    [14.59, 14.28, 0.8993, 5.351, 3.333, 4.185, 4.781, 1],
    [13.99, 13.83, 0.9183, 5.119, 3.383, 5.234, 4.781, 1],
    [15.69, 14.75, 0.9058, 5.527, 3.514, 1.599, 5.046, 1]
])

# Normalize the first 7 columns between 0 and 1
data_min = np.min(training_data[:, :-1], axis=0)
data_max = np.max(training_data[:, :-1], axis=0)

normalized_data = (training_data[:, :-1] - data_min) / (data_max - data_min)

# Combine with labels
normalized_data_with_labels = np.hstack((normalized_data, training_data[:, -1].reshape(-1, 1)))
normalized_data_with_labels




# Precompute the feature indices used at each node
feature_indices = np.argmax(variable_a, axis=0)

# Predict the leaf node for a single sample
def predict(sample):
    current_node = 0  # Start at the root node (node 0)
    
    # Traverse through the nodes
    feature_idx = feature_indices[current_node]  # Get the feature index for the current node
    split_value = variable_b[current_node]  # Get the corresponding split value

    # Compare the feature value of the current sample (observation)
    if sample[feature_idx] >= split_value:
        # Go to the lower right node
        current_node = 2
        print('go to node 2')
    else:
        # Go to the lower left node
        current_node = 1
        print('go to node 1')

    # Determine the final leaf node
    if current_node == 1:  # Lower left node (feature 6)
        if sample[feature_indices[1]] >= variable_b[1]:
            final_leaf_node = 1  # Right leaf of feature 6
            print('leaf node 2')
        else:
            final_leaf_node = 0  # Left leaf of feature 6
            print('leaf node 1')
    else:  # Lower right node (feature 7)
        if sample[feature_indices[2]] >= variable_b[2]:
            final_leaf_node = 3  # Right leaf of feature 7
            print('leaf node 4')
        else:
            final_leaf_node = 2  # Left leaf of feature 7
            print('leaf node 3')

    # Return the final leaf node index (1-based indexing)
    return final_leaf_node + 1

leaf_node = 3

# One-hot encoding for the leaf node (output will be a 4-element vector)
def leaf_node_encoding(leaf_node, num_leaves=4):
    encoding = np.zeros(num_leaves)
    encoding[leaf_node - 1] = 1  # Adjust 1-based indexing
    print('answer encoded')
    return encoding

# Predict for each sample in the normalized dataset using vectorization
def predict_all_samples(normalized_data):
    predicted_leaf_nodes = []
    for sample in normalized_data:
        leaf_node = predict(sample)
        one_hot_leaf_node = leaf_node_encoding(leaf_node)
        predicted_leaf_nodes.append(one_hot_leaf_node)
        print('answer appended')
    return np.array(predicted_leaf_nodes)

# Normalize the first 7 columns between 0 and 1
data_min = np.min(training_data[:, :-1], axis=0)
data_max = np.max(training_data[:, :-1], axis=0)

normalized_data = (training_data[:, :-1] - data_min) / (data_max - data_min)
type(training_data)
type(normalized_data)

sample = np.array([0.52595156, 0.6196319 , 0.02874743, 0.68950749, 0.44196429, 0.28534156, 0.42317823])


# Predict leaf nodes for all samples
predicted_leaf_nodes = predict_all_samples(normalized_data)
predicted_leaf_nodes # ----------- SECOND STAGE VARIABLE Z

# Combine original dataset with the one-hot encoded predicted labels
dataset_with_predictions = np.hstack((training_data, predicted_leaf_nodes))
print(dataset_with_predictions)




# Step 1: Convert real leaf values (last column) to one-hot encoding
def convert_real_leaf_values_to_one_hot(real_leaf_values, num_leaves=4):
    one_hot_real_leafs = np.zeros((real_leaf_values.shape[0], num_leaves))
    for i, leaf in enumerate(real_leaf_values):
        one_hot_real_leafs[i, int(leaf) - 1] = 1  # Convert real leaf value to one-hot (1-based indexing)
    return one_hot_real_leafs

# Step 2: Compute the absolute difference between the real and predicted leaf nodes
def compute_prediction_cost(predicted_leaf_nodes, real_leaf_values_one_hot):
    # Compute the element-wise absolute difference
    absolute_differences = np.abs(predicted_leaf_nodes - real_leaf_values_one_hot)
    
    # Sum the absolute differences to get the cost
    total_cost = np.sum(absolute_differences)
    
    return total_cost

# Extract real leaf values (assumed to be in the last column of the training_data)
real_leaf_values = training_data[:, -1]

# Convert real leaf values to one-hot encoded format
real_leaf_values_one_hot = convert_real_leaf_values_to_one_hot(real_leaf_values)

# Step 3: Compare real vs predicted and compute the cost
total_cost = compute_prediction_cost(predicted_leaf_nodes, real_leaf_values_one_hot)
total_cost # ---------------- SECOND STAGE VARIABLE L (COST)




