import torch

    
    
    
class Hiddenlayer(torch.nn.Module):
    def __init__(self, in_channel, out_channel, dropout=0.5, batchnorm=True):
        super().__init__()
        if batchnorm:
            if dropout > 0:
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(in_channel, out_channel),
                    torch.nn.BatchNorm1d(out_channel),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout)
                )
            else:
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(in_channel, out_channel),
                    torch.nn.BatchNorm1d(out_channel),
                    torch.nn.ReLU()
                )
        else:
            if dropout > 0:
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(in_channel, out_channel),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(dropout)
                )
            else:
                self.layers = torch.nn.Sequential(
                    torch.nn.Linear(in_channel, out_channel),
                    torch.nn.ReLU()
                )
    def forward(self, x):
        return self.layers(x)
    
    
    
    
class NeuralNetvGeneratorRegularized(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0, batchnorm=True, elementwise=False):
        super(NeuralNetvGeneratorRegularized, self).__init__()
        if not isinstance(hidden_dim, list):
            hidden_dim = [hidden_dim]
        self.hidden_dim = hidden_dim
        self.hidden_layers = []
        self.dropout = dropout
        self.batchnorm = batchnorm
        self.elementwise = elementwise
        if elementwise:
            self.input_layer = torch.nn.Sequential(
                ElementwiseLinear(input_dim),
                Hiddenlayer(input_dim, hidden_dim[0], dropout, batchnorm),
            )
        else:
            self.input_layer = torch.nn.Sequential(
                Hiddenlayer(input_dim, hidden_dim[0], dropout, batchnorm),
            )
        h_prev = hidden_dim[0]
        
        if len(self.hidden_dim) > 1:
            for h in self.hidden_dim[1:]:
                self.hidden_layers.append(Hiddenlayer(h_prev, h, dropout, batchnorm))
                h_prev = h
        
            self.hidden_layers = torch.nn.Sequential(*self.hidden_layers)    
        
        self.output_layer = torch.nn.Linear(hidden_dim[-1], output_dim)
       
    def forward(self, x):
        x = self.input_layer(x)
        if len(self.hidden_dim) > 1:
            x = self.hidden_layers(x)
        x = self.output_layer(x)

        return x
    
    
    def get_name(self):
        str_hidden_dim = "_".join([str(h) for h in self.hidden_dim])
        str_name = f"NeuralNetGeneratorRegularized_Nh{len(self.hidden_dim)}_hdim{str_hidden_dim}_dp{self.dropout:.0e}_BN{int(self.batchnorm)}"
        if self.elementwise:
            str_name += "_EW"
        return str_name
    
    
    
############################################################################################################


# model for feature importance visibility with first layer

# shamelessly stolen from stackoverflow
# https://stackoverflow.com/questions/66343862/how-to-create-a-1-to-1-feed-forward-layer
class ElementwiseLinear(torch.nn.Module):
    def __init__(self, input_size: int) -> None:
        super(ElementwiseLinear, self).__init__()

        # w is the learnable weight of this layer module
        self.w = torch.nn.Parameter(torch.rand(input_size), requires_grad=True)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # simple elementwise multiplication
        return self.w * x
    
    