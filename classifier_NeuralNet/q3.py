import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

ORDER = "23451"
# =============================================================================
class Net(nn.Module):
    """
    """
    # -------------------------------------------------------------------------
    def __init__(self):
        """
        """
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 3) 
        self.fc2 = nn.Linear(3, 4) 
        self.fc3 = nn.Linear(4, 5) 
        self.fc4 = nn.Linear(5, 1)
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    def forward(self, x):
        """
        """
        y1 = torch.sigmoid(self.fc1(x)) 
        y2 = torch.sigmoid(self.fc2(y1)) 
        y3 = torch.sigmoid(self.fc3(y2))
        y = torch.sigmoid(self.fc4(y3))
        return y
    # -------------------------------------------------------------------------
# =============================================================================
#------------------------------------------------------------------------------
def main():
    """
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataset = np.load("data/Hastie-data.npy")

    # Features
    x1 = dataset[np.where(dataset[:,2] == 0)][:,0]
    y1 = dataset[np.where(dataset[:,2] == 0)][:,1]
    
    x2 = dataset[np.where(dataset[:,2] != 0)][:,0]
    y2 = dataset[np.where(dataset[:,2] != 0)][:,1]

    # Visualize data before classifying
    plt.scatter(x1, y1)
    plt.scatter(x2, y2, color="green")
    
    plt.title("Hastie Data distribution")
    plt.xlabel("x-feature")
    plt.ylabel("y-feature")
    plt.ylim([-3, 4])
    plt.xlim([-3, 4])
    plt.figtext(0.18, 0.14, "Class Blue", \
                                backgroundcolor="royalblue", color="darkkhaki")
    plt.figtext(0.18, 0.205, "Class Green", \
                                backgroundcolor="green", color="white")


    plt.savefig('plots/default_hastie_.png', bbox_inches='tight')

    # Neural Net
    input_features = torch.Tensor(dataset[:,:2]).float()    
    labels = torch.Tensor(dataset[:,2]).view(len(input_features), 1)

    binary_classifier = Net()
    binary_classifier.to(device)

    optimizer = torch.optim.Adam(binary_classifier.parameters(), lr = 1e-2)
    loss_fn = nn.MSELoss(reduction='sum')

    #--------------------------------------------------------------------------
    # Train
    #--------------------------------------------------------------------------
    for epoch in range(1000):

        input_features, labels = input_features.to(device), labels.to(device)
        # Forward pass: compute predicted y by passing 
        # x to the binary_classifier.            
        predicted_labels = binary_classifier(input_features)
        
        # Compute and color =  loss.
        loss = loss_fn(predicted_labels, labels)
        # color = (convergence_step, loss.item())

        # Zero all gradients to prevent gradient buffers from growing
        optimizer.zero_grad()

        # Backward pass: compute gradient of the loss 
        # with respect to binary_classifier parameters
        loss.backward()

        # Update weights and biases
        optimizer.step()
    #--------------------------------------------------------------------------

    # Save model
    torch.save(binary_classifier, "trained_models/hastie_binary_classifier")

    #--------------------------------------------------------------------------
    # Draw decision regions
    #--------------------------------------------------------------------------
    with torch.no_grad():
        for x in torch.linspace(np.min(dataset[:,0]), 
                                        np.max(dataset[:,0]), 100):
            for y in torch.linspace(np.min(dataset[:,1]),
                                        np.max(dataset[:,1]), 100):
                
                input_xy = torch.stack((x,y))
                input_xy = input_xy.to(device)
                z = binary_classifier(input_xy)

                if z.item() <= 0.5:
                    color = "skyblue"
                else:
                    color = "lightgreen"
                
                plt.plot(x, y, color= color, \
                                    marker = 'o', markersize=20, alpha=0.01)
    
    #--------------------------------------------------------------------------

    error = loss.item()
    plt.title("Hastie Data decision boundary")
    plt.xlabel("x-feature")
    plt.ylabel("y-feature")
    plt.ylim([-3, 4])
    plt.xlim([-3, 4])
    plt.figtext(0.18, 0.14, "Class Blue", \
                                backgroundcolor="royalblue", color="darkkhaki")
    plt.figtext(0.18, 0.205, "Class Green", \
                                backgroundcolor="green", color="white")


    plt.savefig('plots/decision_boundary_hastie_' + ORDER + \
                    '_hidden_' + '%.2f' % error + '_.png', bbox_inches='tight')

#------------------------------------------------------------------------------

if __name__ == "__main__":
    main()