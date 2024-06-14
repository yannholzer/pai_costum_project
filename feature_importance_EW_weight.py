from src.data.utils import get_traintest_args
from src.data.dataset import DiskPlanetDataset
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml

args = get_traintest_args()

# load the data
processed_path = args.processed_path
batch_size = args.batch_size
test_dataset_init = DiskPlanetDataset(os.path.join(processed_path, "test.csv"))
input_dim = test_dataset_init.input_len()
output_dim = test_dataset_init.output_len()
hidden_dim = args.hidden_dim

DISK_COLUMNS = ["metallicity", "gas disk (Msun)", "solid disk (Mearth)", "life time (yr)", "luminosity"]


# load the model    
if args.model == "NeuralNetGenerator":
    from src.model.models import NeuralNetvGenerator as Model
    
    model = Model(input_dim, hidden_dim, output_dim)

    
elif args.model =="NeuralNetGeneratorRegularized":
    from src.model.models import NeuralNetvGeneratorRegularized as Model
    model = Model(input_dim, hidden_dim, output_dim, dropout=args.dropout, batchnorm=args.batchnorm, elementwise=args.elementwise)
else:
    raise ValueError(f"Model {args.model} not recognized")



model_name = model.get_name()

training_stats_name = f"batchsize{args.batch_size}_lr{args.learning_rate:.0e}_epochs{args.epochs}"

model_path = f"./models/{args.labels_type}/{model_name}_{args.optimizer}_{args.loss_fn}/{training_stats_name}/{model_name}_epoch{args.epochs}_best.pt"

feature_importance_path = f"./feature_importance/{args.labels_type}/{model_name}_{args.optimizer}_{args.loss_fn}/{training_stats_name}/"

try:
    os.makedirs(feature_importance_path)
except FileExistsError:
    print("folder already exists")
    pass
    # for f in os.listdir(feature_importance_path):
    #     os.remove(os.path.join(feature_importance_path, f))

    

state_dict = torch.load(model_path)
model.load_state_dict(state_dict["model_state_dict"])
model.eval()

performances_dict = {}


w = model.input_layer[0].w.detach().numpy()


############################################################################################################
# Get the weight for each feature


fig, ax = plt.subplots(1, 1, figsize=(12, 6))

for i_f, feature in enumerate(DISK_COLUMNS):
    ax.bar(i_f, w[i_f], label=feature)
    
    performances_dict[feature] = float(w[i_f])
    
ax.set(xlabel="feature", ylabel="weight", title="Input weight of each feature")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(feature_importance_path, "feature_importance_EW_weight.png"), dpi=90)

performances_dict_path = os.path.join(feature_importance_path, "performances_weight.yaml")

with open(performances_dict_path, "w") as f:
    yaml.safe_dump(performances_dict, f)
