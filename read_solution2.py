################################################################
###################### EXTRACT SOLVER DATA #####################
################################################################

import re
import numpy as np

# Read data path
problem_path = "seeds-sd1-2-CMS-.out"

def extract_matrices_and_vectors(output, num_features, num_classes):

        # Open the file for reading
    with open(problem_path, 'r') as file:
        lines = file.readlines()

    # Combine all lines into a single string
    log_text = ''.join(lines)


    # Split the output into lines
    lines = log_text.split('\n')
    
    # Initialize containers for matrices and vectors
    matrices = []
    vectors = []
    
    # Regular expressions to identify matrices and vectors
    matrix_pattern = re.compile(r'^\d+×\d+ Matrix{Float64}:$')
    vector_pattern = re.compile(r'^\d+-element Vector{Float64}:$')
    
    # Temporary container for current matrix or vector
    current_matrix = []
    current_vector = []
    
    # Flags to indicate if we are reading a matrix or vector
    reading_matrix = False
    reading_vector = False
    
    for line in lines:
        line = line.strip()
        
        # Check if the line indicates the start of a matrix
        if matrix_pattern.match(line):
            reading_matrix = True
            reading_vector = False
            if current_matrix:
                matrices.append(np.array(current_matrix))
                current_matrix = []
            continue
        
        # Check if the line indicates the start of a vector
        if vector_pattern.match(line):
            reading_vector = True
            reading_matrix = False
            if current_vector:
                vectors.append(np.array(current_vector))
                current_vector = []
            continue
        
        # If we are reading a matrix, add the line to the current matrix
        if reading_matrix:
            if line and  matrix_pattern.match(line) and not vector_pattern.match(line):
                current_matrix.append([float(x) for x in line.split()])
        
        # If we are reading a vector, add the line to the current vector
        if reading_vector:
            try:
                current_vector.append(float(line))
            except ValueError:
                # If conversion fails, stop reading the vector
                reading_vector = False
    
    # Append the last read matrix or vector
    if current_matrix:
        matrices.append(np.array(current_matrix))
    if current_vector:
        vectors.append(np.array(current_vector))
    
    return matrices, vectors

def group_matrices_and_vectors(matrices, vectors):
    warm_start = {
        'matrices': matrices[:2],
        'vectors': vectors[:2]
    }
    optimal_solution = {
        'matrices': matrices[2:],
        'vectors': vectors[2:]
    }
    return warm_start, optimal_solution



# Example usage
output = """
7×3 Matrix{Float64}:
 0.0  1.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  0.0
 0.0  0.0  1.0
 1.0  0.0  0.0
3-element Vector{Float64}:
 0.5307730182176268
 0.2648725212464589
 0.18215033351103257
3×7 Matrix{Float64}:
 0.0  0.0  0.0  0.0  1.0  0.0  0.0
 0.0  0.0  0.0  0.0  0.0  1.0  1.0
 0.0  0.0  0.0  1.0  0.0  0.0  0.0
3-element Vector{Float64}:
 1.0
 1.0
 1.0
7×3 Matrix{Float64}:
 0.0   0.0  -0.0
 1.0   0.0  -0.0
 0.0   0.0  -0.0
 0.0   0.0  -0.0
 0.0   0.0  -0.0
 0.0   1.0  -0.0
 0.0  -0.0   1.0
3-element Vector{Float64}:
 0.2706611570247932
 0.25171306349061884
 0.5000000000000004
3×7 Matrix{Float64}:
 0.0  0.0  0.0  1.0  0.0   1.0   0.0
 0.0  0.0  0.0  0.0  0.0   0.0   1.0
 0.0  0.0  0.0  0.0  1.0  -0.0  -0.0
3-element Vector{Float64}:
 1.0
 1.0
 1.0
"""

matrices, vectors = extract_matrices_and_vectors(output)
warm_start, optimal_solution = group_matrices_and_vectors(matrices, vectors)

print("Warm Start Matrices and Vectors:")
print(warm_start)

print("\nOptimal Solution Matrices and Vectors:")
print(optimal_solution)
