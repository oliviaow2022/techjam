import torch
import torchvision.models as models
import matplotlib.pyplot as plt

# Load the ResNet-18 model
model = models.resnet18(pretrained=True)

# Function to extract and plot weight distributions
def plot_weight_distributions(model):
    layer_weights = []
    layer_names = []

    # Loop through model layers and collect weights
    for name, param in model.named_parameters():
        if 'weight' in name and ('conv1' in name or 'fc' in name):
            layer_weights.append(param.data.cpu().numpy().flatten())
            layer_names.append(name)
    
    print(layer_names)

    # Plot weight distributions
    fig, axs = plt.subplots(len(layer_weights), 1, figsize=(15, len(layer_weights) * 3))
    fig.suptitle('Weight Distributions in ResNet-18 Layers', fontsize=16)

    for i, weights in enumerate(layer_weights):
        axs[i].hist(weights, bins=50)
        axs[i].set_title(layer_names[i])
        axs[i].set_xlabel('Weight Value')
        axs[i].set_ylabel('Frequency')

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

# Call the function to plot weight distributions
plot_weight_distributions(model)

# Extract the first convolutional layer weights
conv1_weights = model.conv1.weight.data.cpu()

# Plot the filters
def plot_filters(filters, n_cols=8):
    n_filters = filters.shape[0]
    n_rows = (n_filters + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            if idx < n_filters:
                ax = axs[i, j] if n_rows > 1 else axs[j]
                ax.imshow(filters[idx].permute(1, 2, 0).squeeze(), cmap='gray')
                ax.axis('off')
    plt.show()

# Visualize filters of the first convolutional layer
plot_filters(conv1_weights)