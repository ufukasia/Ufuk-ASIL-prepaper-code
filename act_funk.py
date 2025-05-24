import numpy as np
import matplotlib.pyplot as plt

# --- Activation Functions ------------------------------------------------

def relu(x):
    """Rectified Linear Unit (ReLU) activation function."""
    return np.maximum(0, x)

def step(x):
    """Step activation function (Heaviside step function)."""
    return np.where(x >= 0, 1, 0)

def casef(x, s):
    """
    Clipped Adaptive Saturation Exponential Function (CASEF).
    This function outputs 0 for negative inputs, scales exponentially
    within the [0, 1] range, and saturates at 1 for inputs greater than 1.
    The parameter 's' controls the steepness of the exponential curve,
    allowing for adaptive shaping of the activation.
    """
    x_clip = np.clip(x, 0.0, 1.0)
    return (np.exp(s * x_clip) - 1) / (np.exp(s) - 1)

def exponential_surge(x):
    """Exponential Surge function, rises from 0 to 1 with a specific curve."""
    return np.piecewise(
        x,
        [x < 0, (x >= 0) & (x <= 1), x > 1],
        [0, lambda x: 1 - (1 - x) ** 5, 1]
    )

def quartic_unit_step(x):
    """Quartic Unit Step function, provides a smooth transition from 0 to 1."""
    return np.minimum(x**4, 1)

# --- Parameters for Plotting ------------------------------------------------

# Top row: Four distinct activation functions for comparison.
top_functions = [
    (quartic_unit_step,     "Quartic Unit Step"),
    (relu,                  "ReLU"),
    (exponential_surge,     "Exponential Surge"),
    (step,                  "Step"),
]

# Bottom row: Four variants of the CASEF function with different 's' values
# to demonstrate its adaptability.
s_values = [4, 0.1, -2, -100]
bottom_functions = [
    (lambda x, s=s: casef(x, s), f"CASEF (s={s})")
    for s in s_values
]

# Colors for plotting (total 8, one for each function).
colors = [
    '#0000FF', '#FAAF00', '#FF00FF', '#228B22',
    '#8A2BE2', '#FF1493', '#FFA500', '#00CED1'
]

# --- Grafik çizimi ---------------------------------------------------------

# Initialize a 2x4 subplot grid for visualizing the activation functions.
fig, axes = plt.subplots(2, 4, figsize=(22, 8), constrained_layout=True)

for idx, (func, name) in enumerate(top_functions + bottom_functions):
    ax = axes[idx // 4, idx % 4]
    x = np.linspace(-0.1, 1, 1000)
    y = func(x)

    # Plot the function.
    ax.plot(x, y, label=name, color=colors[idx], linewidth=2)
    # Add horizontal and vertical lines at 0 for reference.
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    ax.axvline(0, color='k', linestyle='--', linewidth=0.5)
    # Set axis limits for consistent visualization.
    ax.set_xlim(-0.1, 1.01)
    ax.set_ylim(-0.1, 1.01)

    # Set plot title and labels.
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.set_xlabel('x', fontsize=9)
    ax.set_ylabel('f(x)', fontsize=9, labelpad=0)
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.grid(True, linestyle=':', alpha=0.6)

# Save the figure and display it.
plt.savefig('activation_functions_2rows.png', dpi=300, bbox_inches='tight')
plt.show()