import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class TransitionNet(nn.Module):
    """
    Neural network f^theta(Y, Y*)
    Input: one-hot(Y) concatenated with one-hot(Y*)
    Output: scalar score in (0,1), interpreted as transition probability weight
    """
    def __init__(self, k, hidden_dim=16):
        super().__init__()
        self.k = k
        self.net = nn.Sequential(
            nn.Linear(2 * k, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid() # logistic
        )

    def forward(self, y_onehot, ystar_onehot):
        z = torch.cat([y_onehot, ystar_onehot], dim=1)
        return self.net(z).squeeze(1)


def build_transition_matrix(model, k, device="cpu"):
    """
    Evaluate f^theta(Y, Y*) on all category pairs (k,j)
    and normalize rows to get a valid transition matrix P.
    """
    rows = []

    for true_class in range(k): # loop over true class i
        row_scores = []
        for privatized_class in range(k): # loop over possible privastised class j
            y = F.one_hot(torch.tensor([true_class], device=device), num_classes=k).float() # create one-hot encoding for true label i
            y_star = F.one_hot(torch.tensor([privatized_class], device=device), num_classes=k).float() # creater one hot encoding for private label j

            #evaluate neural network f^theta(i,j)
            score = model(y, y_star)   #scalar
            row_scores.append(score)


        # stack scores into a row vector of shape (1,k)
        row_scores = torch.stack(row_scores, dim=1)#shape (1, k)
        row_probs = torch.softmax(row_scores, dim=1)#row sums to 1, apply softmax to onvert scores into probabilities
        rows.append(row_probs) # append row i to list of rows

    P = torch.cat(rows, dim=0)# shape (k, k) concatenate all rows to form full transition matrix P (kxk dimension)
    return P

def privacy_loss_fn(beta, P):
    """
    penalize large diagonal entries, since they reveal the true label more often.
    beta is included to match the paper structure.
    """
    beta_penalty = torch.sum(beta ** 2)
    diag_penalty = torch.sum(torch.diag(P))
    return beta_penalty + diag_penalty

def utility_loss_fn(P):
    """
    encourage transitions that preserve signal.
    Smaller off-diagonal mass means better utility.
    """
    off_diag = P - torch.diag(torch.diag(P))
    return torch.sum(off_diag ** 2)

def learn_transition_matrix(k, beta, gamma=0.5, epochs=500, lr=1e-2, device="cpu"):
    '''
    Learn optimal transition matrix P using a neural network.

    Objective:
        balance privacy and utility via:
            total_loss = -(1 - γ)*privacy + γ*utility
    '''
    model = TransitionNet(k).to(device) # initialise neural net that parameterises P
    optimizer = optim.Adam(model.parameters(), lr=lr) # adam optimization updates network parameters theta

    best_P = None
    best_loss = float("inf") # initialise loss so improvement is recorded

    beta_torch = torch.tensor(beta, dtype=torch.float32, device=device) # convert beta from MLR into a torch tensor for loss computations

    for epoch in range(epochs):
        P = build_transition_matrix(model, k, device=device) # build current transition matrix P(theta( from neural network))

        L_privacy = privacy_loss_fn(beta_torch, P) # penalize true label
        L_utility = utility_loss_fn(P) # penalize excessive distortion of labels

        total_loss = -(1 - gamma) * L_privacy + gamma * L_utility # OBJECTIVE FUNCTION

        optimizer.zero_grad() # reset gradients from previous iteration
        total_loss.backward() # back propogate gradients thru network
        optimizer.step() # update theta via adam

        current_loss = total_loss.item() # get scalar loss value & update best reansition matrix if current loss improves
        if current_loss < best_loss:
            best_loss = current_loss
            best_P = P.detach().cpu().numpy() # store as numpy not gradients

    return best_P # returns the best learned transition matrix