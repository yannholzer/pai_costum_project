from src.data.utils import get_testing_args
from src.data.dataset import DiskPlanetDataset
from torch.utils.data.dataloader import DataLoader
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml


args = get_testing_args()

processed_path = args.processed_path


if args.model == "NeuralNetvGenerator":
    from src.model.models import NeuralNetvGenerator as Model
else:
    raise ValueError(f"Model {args.model} not recognized")


batch_size = args.batch_size
test_dataset = DiskPlanetDataset(os.path.join(processed_path, "test_data.npy"))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


input_dim = test_dataset.input_len()
output_dim = test_dataset.output_len()
hidden_dim = args.hidden_dim
model = Model(input_dim, hidden_dim, output_dim)

model_name = model.get_name()

training_stats_name = f"batchsize{args.batch_size}_lr{args.learning_rate:.0e}"

metric_path = f"./metrics/{model_name}_{args.optimizer}_{args.loss_fn}/{training_stats_name}/metrics.yaml"
plot_path = f"./plots/{model_name}_{args.optimizer}_{args.loss_fn}/{training_stats_name}/"
model_path = f"./models/{model_name}_{args.optimizer}_{args.loss_fn}/{training_stats_name}/{model_name}_epoch{args.epochs}.pt"

state_dict = torch.load(model_path)
model.load_state_dict(state_dict["model_state_dict"])
model.eval()



with open(metric_path, "r") as f:
    metrics = yaml.safe_load(f)


############################################################################################################
# test the performances

outputs_array = np.empty(test_dataset.labels_shape())
labels_array = np.empty_like(outputs_array)



def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)




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
    fig, ax = plt.subplots(1,1, figsize=(12, 6))
    ax.hist(labels_array[:, dim], alpha=0.3, label="true", bins=20)
    ax.hist(outputs_array[:, dim], alpha=0.3, label="predicted", bins=20)
    ax.set(title=f"{names[dim]}", xlabel="value", ylabel="count")
    fig.legend()
    plt.tight_layout()
    plot_file_hist = os.path.join(plot_path, "hist_"+names[dim]+".png")
    plt.savefig(plot_file_hist, dpi=90)
    plt.close()
    # plt.show()
    
    fig, ax = plt.subplots(1,1, figsize=(6, 6))
    ax.scatter(labels_array[:, dim], outputs_array[:, dim])
    ax.set(xlabel="true", ylabel="predicted", title=f"{names[dim]}")
    ax.plot([min(outputs_array[:, dim]), max(outputs_array[:, dim])], [min(outputs_array[:, dim]), max(outputs_array[:, dim])], 'r--')
    ax.set_aspect('equal', 'box')
    # fig.legend()
    plot_file_scat = os.path.join(plot_path, "scatter_"+names[dim]+".png")

    plt.savefig(plot_file_scat,  dpi=90)
    plt.close()
    
    # plt.show()
    
    rmse = root_mean_squared_error(labels_array[:, dim], outputs_array[:, dim])
    mae = mean_absolute_error(labels_array[:, dim], outputs_array[:, dim])
    mse = mean_squared_error(labels_array[:, dim], outputs_array[:, dim])
    metrics[f"RMSE_{names[dim]}"] = float(rmse)
    metrics[f"MAE_{names[dim]}"] = float(mae)
    metrics[f"MSE_{names[dim]}"] = float(mse)
    
    total_rmse += rmse
    total_mae += mae
    total_mse += mse
    
    print(names[dim])
    print("rmse:", rmse)
    print("mae:", mae)
    print("mse:", mse)    


metrics[f"total_RMSE"] = float(total_rmse)
metrics[f"total_MAE"] = float(total_mae)
metrics[f"total_MSE"] = float(total_mse)



with open(metric_path, "w") as f:
    yaml.safe_dump(metrics, f)
