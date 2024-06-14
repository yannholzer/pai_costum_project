from src.data.utils import get_traintest_args
from src.data.dataset import DiskPlanetDataset
from torch.utils.data.dataloader import DataLoader
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys
import copy

args = get_traintest_args()

# load the data
processed_path = args.processed_path
batch_size = args.batch_size
test_dataset_init = DiskPlanetDataset(os.path.join(processed_path, "test.csv"))
input_dim = test_dataset_init.input_len()
output_dim = test_dataset_init.output_len()
hidden_dim = args.hidden_dim




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

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)
############################################################################################################
# test the performances for each perumations

for feature in range(test_dataset_init.input_len()):
    test_dataset = DiskPlanetDataset(os.path.join(processed_path, "test.csv"), permute=feature)
    feature_name = test_dataset.permuted_feature
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    performances_dict[feature_name] = {}

    outputs_array = np.empty(test_dataset.labels_shape())
    labels_array = np.empty_like(outputs_array)






    with torch.no_grad():
        prev_index = 0
        for i_data, (inputs, labels) in enumerate(test_loader):
            
            outputs = model(inputs).numpy()
            labels = labels.numpy()
            
            outputs_array[prev_index:prev_index+outputs.shape[0]] = outputs
            labels_array[prev_index:prev_index+outputs.shape[0]] = labels
            prev_index += outputs.shape[0]


    names = test_dataset.label_names
    total_rmse = 0
    total_mse = 0
    total_mae = 0
    for dim in range(test_dataset.output_len()):        
        
        rmse = root_mean_squared_error(labels_array[:, dim], outputs_array[:, dim])
        mae = mean_absolute_error(labels_array[:, dim], outputs_array[:, dim])
        mse = mean_squared_error(labels_array[:, dim], outputs_array[:, dim])
        performances_dict[feature_name][f"RMSE_{names[dim]}"] = float(rmse)
        performances_dict[feature_name][f"MAE_{names[dim]}"] = float(mae)
        performances_dict[feature_name][f"MSE_{names[dim]}"] = float(mse)
        
        total_rmse += rmse
        total_mae += mae
        total_mse += mse
        
        print(names[dim])
        print("rmse:", rmse)
        print("mae:", mae)
        print("mse:", mse)    


    performances_dict[feature_name][f"total_RMSE"] = float(total_rmse)
    performances_dict[feature_name][f"total_MAE"] = float(total_mae)
    performances_dict[feature_name][f"total_MSE"] = float(total_mse)



fig, ax = plt.subplots(1, 1, figsize=(12, 6))

for i_f, feature in enumerate(performances_dict):
    ax.bar(i_f, performances_dict[feature]["total_RMSE"], label=feature)
ax.set(xlabel="feature", ylabel="total RMSE", title="Total RMSE for each permuted feature")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(feature_importance_path, "feature_importance_total_RMSE.png"), dpi=90)

performances_dict_path = os.path.join(feature_importance_path, "performances_permutation.yaml")

with open(performances_dict_path, "w") as f:
    yaml.safe_dump(performances_dict, f)
