import math
import pdb

def calculate_kernel_memory(c, k_x, k_y, f):
    return c * 2**(math.log2(k_x * k_y) + math.log2(f))

def calculate_neuron_memory(f, c_x, c_y, k_x, k_y, s_x, s_y, p_x, p_y):
    f_x = ((c_x - k_x + 2*p_x) // s_x) + 1
    f_y = ((c_y - k_y + 2*p_y) // s_y) + 1
    return f * f_x * f_y

# Layer definitions
layers = [
    ["Conv", 2, 1, 1, 0],
    ["Conv", 16, 5, 2, 1],
    ["ReLu"],
    ["Pool", 2],
    ["Conv", 64, 3, 1, 1],
    ["ReLu"],
    ["Pool", 2],
    ["Conv", 16, 3, 1, 1],
    ["ReLu"],
    ["Pool", 2],
    ["Conv", 16, 3, 1, 1],
    ["ReLu"],
    ["Conv", 8, 3, 1, 1],
    ["ReLu"],
    ["Conv", 8, 3, 1, 1],
    ["ReLu"],
    ["Flat"],
    ["Linear", 128],
    ["ReLu"],
    ["Linear", 275]
]

# Memory configuration for different cores
core_memory = {
    0: (16, 64),
    1: (16, 64),
    2: (16, 64),
    3: (32, 32),
    4: (32, 32),
    5: (64, 16),
    6: (64, 16),
    7: (16, 16),
    8: (16, 16)
}

# Calculate and store results
results = []
c_x, c_y, c = 64, 64, 2
for layer in layers:
    if layer[0] == "Conv":
        _, f, k, s, p = layer
        k_x = k_y = k
        s_x = s_y = s
        p_x = p_y = p 
        kernel_memory = calculate_kernel_memory(c, k_x, k_y, f)
        neuron_memory = calculate_neuron_memory(f, c_x, c_y, k_x, k_y, s_x, s_y, p_x, p_y)
        # Calculate output dimensions after convolution and padding
        c_x = ((c_x - k_x + 2 * p_x) // s_x) + 1
        c_y = ((c_y - k_y + 2 * p_y) // s_y) + 1

        c = f
    elif layer[0] == "Pool":
        _, k = layer
        c_x, c_y = c_x // k, c_y // k  # Update image size due to pooling
        kernel_memory = 0
        neuron_memory = 0
    elif layer[0] == "Linear":
        _, f = layer 
        kernel_memory = calculate_kernel_memory(c, 1, 1, f)
        neuron_memory = calculate_neuron_memory(f, 1, 1, 1, 1, 1, 1, 1, 1)
        c = f

    elif layer[0] == "Flat":  
        c = c_x * c_y * f
        kernel_memory = 0
        neuron_memory = 0

    else:
        kernel_memory = 0
        neuron_memory = 0
        
    results.append((layer[0], kernel_memory, neuron_memory))

# Print the results in a LaTeX table format
print("\\begin{table}")
print("\\centering")
print("\\begin{tabular}{|c|c|c|}")
print("\\hline")
print("Layer & Kernel Memory ($K_{MT}$ KiB) & Neuron Memory ($N_M$ KiB) \\\\")
print("\\hline")
for i, (name, kernel_memory, neuron_memory) in enumerate(results):
    if name not in ["Conv", "Linear"]:
        print(f"{i+1} & {name} &  &  & \\\\")
        continue
    core_id = i % 9
    core_k_memory, core_n_memory = core_memory[core_id]
    kernel_memory_str = f"{kernel_memory / 1024:.2f}"
    neuron_memory_str = f"{neuron_memory / 1024:.2f}" 
    compatible_cores = ''
    for key in core_memory.keys():
        if core_memory[key][0]>=float(kernel_memory_str) and core_memory[key][1]>=float(neuron_memory_str):
            compatible_cores = compatible_cores + f', {key}'

    print(f"{i+1} & {name} & {kernel_memory_str} KiB & {neuron_memory_str} KiB & {compatible_cores} \\\\")
print("\\end{tabular}")
print("\\caption{Memory Footprint for Each Layer}")
print("\\end{table}")
