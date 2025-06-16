import json
import matplotlib.pyplot as plt
import numpy as np
import os

# Set Chinese font for display
plt.rcParams['font.sans-serif'] = ['SimHei']  # For displaying Chinese characters
plt.rcParams['axes.unicode_minus'] = False  # For displaying negative signs correctly

# File paths
weighted_path = r"F:\github\Graph_Transformer_Networks\metapath2vec_including_edge_weight\original_metapath2vec\experiment_result\training_stats.json"
unweighted_path = r"F:\github\Graph_Transformer_Networks\experiment_result\training_stats.json"

def load_training_stats(file_path):
    """Load training statistics from JSON file
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        list: Array of epoch losses
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        return data['epoch_losses']  # Return epoch_losses array directly

# Load training data
weighted_losses = load_training_stats(weighted_path)
unweighted_losses = load_training_stats(unweighted_path)

# Create epoch list
epochs = list(range(1, len(weighted_losses) + 1))

# Create figure
plt.figure(figsize=(12, 7))

# Plot training curves
plt.plot(epochs, weighted_losses, 'b-', label='Weighted Training', linewidth=2)
plt.plot(epochs, unweighted_losses, 'r--', label='Unweighted Training', linewidth=2)

# Add chart elements
plt.title('Training Loss Comparison', fontsize=16, pad=15)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12, loc='upper right')

# Set y-axis range with margin
y_min = min(min(weighted_losses), min(unweighted_losses))
y_max = max(max(weighted_losses), max(unweighted_losses))
margin = (y_max - y_min) * 0.1
plt.ylim(y_min - margin, y_max + margin)

# Add grid lines
plt.grid(True, linestyle='--', alpha=0.7)

# Optimize x-axis ticks
plt.xticks(np.arange(0, len(epochs) + 1, 5))  # Show tick every 5 epochs

# Save figure
save_path = os.path.join(os.path.dirname(__file__), 'training_loss_comparison.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"Figure saved to: {save_path}")

# Print statistics
print("\nTraining Loss Statistics:")
print(f"Weighted Training - Final Loss: {weighted_losses[-1]:.4f}, Min Loss: {min(weighted_losses):.4f}")
print(f"Unweighted Training - Final Loss: {unweighted_losses[-1]:.4f}, Min Loss: {min(unweighted_losses):.4f}")

# Calculate improvement percentage
improvement = ((unweighted_losses[-1] - weighted_losses[-1]) / unweighted_losses[-1]) * 100
print(f"\nImprovement of weighted training over unweighted training: {improvement:.2f}%") 