"""
@file: latency.py
Created on 23.01.23
@project: CrazyAra
@author: TimM4ster

TODO
"""

import time
import torch
import csv
import sys

sys.path.insert(0, '../../../../../../')

from DeepCrazyhouse.src.domain.neural_net.nas.search_space.a0_space import AlphaZeroSearchSpace

def measure_average_module_latency(model, input_size=(1, 256, 8, 8), device='cuda', num_iterations=100):
    """
    Measure the average latency of a PyTorch model over a set number of iterations.
    :param model: PyTorch model to measure
    :param input_size: Tuple representing the size of the input to the model
    :param device: Device to run the model on
    :param num_iterations: Number of iterations to average over
    :return: Average latency in seconds
    """
    # Move the model to the specified device
    model = model.to(device)
    model.eval()

    # Create a random input tensor of the specified size
    input = torch.randn(input_size).to(device)

    # Warm up the model
    for _ in range(10):
        model(input)

    # Measure the time before the forward passes
    start_time = time.time()

    # Perform the forward passes
    for _ in range(num_iterations):
        model(input)

    # Measure the time after the forward passes
    end_time = time.time()

    # Compute the average latency
    average_latency = (end_time - start_time) / num_iterations

    return average_latency 

def measure_and_save_latencies(input_size=(1, 256, 8, 8), device='cuda', num_iterations=100, output_file='latencies.csv'):
    """
    Measure the latencies of all blocks and save them to a file.
    :param input_size: Tuple representing the size of the input to the blocks
    :param device: Device to run the blocks on
    :param num_iterations: Number of iterations to average over
    :param output_file: Path to the output file
    """
    # Create an instance of the search space
    search_space = AlphaZeroSearchSpace()

    # Open the output file in write mode
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write the header row
        writer.writerow(['Block', 'Average Latency'])

        # For each block in the search space
        for block_name, block in search_space.op_candidates.items():
            # Measure the average latency of the block
            average_latency = measure_average_module_latency(block, input_size, device, num_iterations)

            # Write the block name and average latency to the file
            writer.writerow([block_name, average_latency])

# Call the function to measure and save the latencies
measure_and_save_latencies()