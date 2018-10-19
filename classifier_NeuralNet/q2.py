import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# =============================================================================
class Perceptron(nn.Module):
    """
    """
    # -------------------------------------------------------------------------
    def __init__(self):
        """
        """
        super(Perceptron, self).__init__()
        self.layer = nn.Linear(2, 1) 
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def forward(self, x):
        """
        """
        y = torch.sigmoid(self.layer(x))
        return y
    # -------------------------------------------------------------------------
# =============================================================================
#------------------------------------------------------------------------------
def main():
    """
    """
    loss_fn = nn.MSELoss(reduction='sum')

    filenames = "Frogs-subsample.csv", "Frogs.csv"
    for filename in filenames:
        frogData = pd.read_csv("data/" + filename)
        label_mapping = {"HypsiboasCinerascens" : 1, "HylaMinuta": 0} 
        
        input_features = torch.tensor(\
                            np.array(frogData.values[:,:2], dtype=np.float32))
        labels = torch.tensor(frogData["Species"].map(label_mapping).values, 
                            dtype=torch.float32).view(len(input_features), 1)

        binary_classifier = Perceptron()
        optimizer = torch.optim.Adam(binary_classifier.parameters(), lr = 2)

        for epoch in range(1000):
            # Forward pass: compute predicted y by 
            # passing x to the binary_classifier.            
            predicted_labels = binary_classifier(input_features)
            
            # Compute loss.
            loss = loss_fn(predicted_labels, labels)
            
            # Zero all gradients to prevent gradient buffers from growing
            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss 
            # with respect to binary_classifier parameters
            loss.backward()

            # Update all weights and biases
            optimizer.step()

        # Save model
        torch.save(binary_classifier, "trained_models/frogMFCC_binary_classifier")
    
        # Plot the decision boundary
        # Decision Line is ax + by + c = 0
        a = binary_classifier.layer.weight.data[0][0].item()
        b = binary_classifier.layer.weight.data[0][1].item()
        c = binary_classifier.layer.bias.item()
        
        # Get uniform points in x direction
        x = np.linspace(min(frogData.values[:,0]), 
                            max(frogData.values[:,0]), 100)
        y = (-a*x -c)/b 



        frogData = pd.read_csv("data/" + filename).values 

        # Features
        x1 = frogData[np.where(frogData[:,2] == "HylaMinuta")][:,0]
        y1 = frogData[np.where(frogData[:,2] == "HylaMinuta")][:,1]
        
        x2 = frogData[np.where(frogData[:,2] != "HylaMinuta")][:,0]
        y2 = frogData[np.where(frogData[:,2] != "HylaMinuta")][:,1]

        plt.scatter(x1, y1, color="royalblue", label="HylaMinuta")
        plt.scatter(x2, y2, color="green", alpha=0.7, label="HypsiboasCinerascens")
        plt.ylim([-.5,.5])
        plt.xlim([-.6,.6])

        plt.xlabel("MFCC_10")
        plt.ylabel("MFCC_17")
        plt.legend(("HylaMinuta", "HypsiboasCinerascens"))
        plt.title("Decision boundary of Frequency-band-intensities")
        plt.plot(x, y, color="orange", label="boundary")

        plt.savefig('plots/decision_boundary_' + filename + '.png', 
                                                    bbox_inches='tight')
        plt.clf()

#------------------------------------------------------------------------------

if __name__ == "__main__":
    main()