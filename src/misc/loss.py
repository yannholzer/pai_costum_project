
import tensorflow as tf
import torch
import torch

#translated given loss code to pytorch
def threeD_loss(inputs, outputs): #[batch_size x 100 x 3] -> [batch_size]$PLACEHOLDER$
    expand_inputs = torch.unsqueeze(inputs, 2) # add broadcasting dim [batch_size x 100 x 1 x 3]
    expand_outputs = torch.unsqueeze(outputs, 1) # add broadcasting dim [batch_size x 1 x 100 x 3]
    # => broadcasting [batch_size x 100 x 100 x 3] => reduce over last dimension (eta,phi,pt) => [batch_size x 100 x 100] where 100x100 is distance matrix D[i,j] for i all inputs and j all outputs
    distances = torch.sum(torch.square(expand_inputs - expand_outputs), -1)
    # get min for inputs (min of rows -> [batch_size x 100]) and min for outputs (min of columns)
    min_dist_to_inputs = torch.min(distances, 1).values
    min_dist_to_outputs = torch.min(distances, 2).values
    return torch.mean(torch.mean(min_dist_to_inputs, 1) + torch.mean(min_dist_to_outputs, 1)) # mean over batch_size