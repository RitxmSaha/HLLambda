import matplotlib.pyplot as plt
import numpy as np
import os

def plot_value_curve(values, algorithm_name, save_path=None):
    """
    Plot the value function across states.
    """
    plt.figure()
    plt.plot(values, marker='o')
    plt.title(f"Value function: {algorithm_name}")
    plt.xlabel("State")
    plt.ylabel("Estimated Value")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"[INFO] Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_learning_curve(curves, labels, save_path=None, ylabel="RMS Error", xlabel="Step"):
    """
    Plot one or more learning curves.
    curves: list of np.arrays (one for each algorithm)
    labels: list of strings (matching curves)
    """
    plt.figure()
    for arr, label in zip(curves, labels):
        plt.plot(arr, label=label)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
        print(f"[INFO] Learning curve saved to {save_path}")
    else:
        plt.show()
    plt.close()
