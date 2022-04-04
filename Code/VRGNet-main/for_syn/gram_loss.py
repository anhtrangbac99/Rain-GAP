import torch
import torch.nn.functional as F
import torch.nn as nn
def gram_matrix(input):
    a, b, c, d = input.size()  # a=batch size(=1)
    # b=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(a * b, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self):
        super(StyleLoss, self).__init__()

    def forward(self, input, target):
        G = gram_matrix(input)
        loss = F.mse_loss(G, gram_matrix(target).detach())
        return loss
