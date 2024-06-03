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