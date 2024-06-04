import torch

class NeuralNetv0(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetv0, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_1 = torch.nn.Linear(input_dim, hidden_dim)
        self.layer_2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.layer_3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.layer_4 = torch.nn.Linear(hidden_dim, output_dim)
       
    def forward(self, x):
        x = torch.nn.functional.relu(self.layer_1(x))
        x = torch.nn.functional.relu(self.layer_2(x))
        x = torch.nn.functional.relu(self.layer_3(x))

        x = self.layer_4(x)

        return x
    
    
    def get_name(self):
        return f"NeuralNetv0_Nh2_hdim{self.hidden_dim}"
    
    
    
    
class NeuralNetvGenerator(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NeuralNetvGenerator, self).__init__()
        if not isinstance(hidden_dim, list):
            hidden_dim = [hidden_dim]
        self.hidden_dim = hidden_dim
        self.hidden_layers = []
        
        self.input_layer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim[0]),
            torch.nn.ReLU()
        )
        h_prev = hidden_dim[0]
        for h in self.hidden_dim[1:]:
            self.hidden_layers.append(torch.nn.Linear(h_prev, h))
            self.hidden_layers.append(torch.nn.ReLU())
            h_prev = h
        
        self.hidden_layers = torch.nn.Sequential(*self.hidden_layers)    
        
        self.output_layer = torch.nn.Linear(hidden_dim[-1], output_dim)
       
    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)

        return x
    
    
    def get_name(self):
        str_hidden_dim = "_".join([str(h) for h in self.hidden_dim])
        return f"NeuralNetGenerator_Nh{len(self.hidden_dim)}_hdim{str_hidden_dim}"