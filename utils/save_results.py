import numpy as np
import os

def save_value_function(values, filename):
    """
    Save the value function (array) to disk.
    """
    np.save(filename, values)
    print(f"[INFO] Value function saved to {filename}")

def save_learning_curve(curve, filename):
    """
    Save the learning curve (e.g., RMS error over steps).
    """
    np.save(filename, curve)
    print(f"[INFO] Learning curve saved to {filename}")
