from src.data.dataset import DiskPlanetDataset
from src.data.utils import get_training_args
from torch.utils.data.dataloader import DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import yaml
from importlib import import_module


args = get_training_args()
processed_path = args.processed_path

train_dataset = DiskPlanetDataset(os.path.join(processed_path, "val_data.npy"))
val_dataset = DiskPlanetDataset(os.path.join(processed_path, "val_data.npy"))
test_dataset = DiskPlanetDataset(os.path.join(processed_path, "test_data.npy"))


batch_size = args.batch_size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)



n_epochs = args.epochs
train_loss_values = np.empty(n_epochs)
val_loss_values = np.empty(n_epochs)


input_dim = train_dataset.input_len()
output_dim = train_dataset.output_len()

hidden_dim = args.hidden_dim

learning_rate = args.learning_rate


if args.model == "NeuralNetvGenerator":
    from src.model.models import NeuralNetvGenerator as Model
else:
    raise ValueError(f"Model {args.model} not recognized")

model = Model(input_dim, hidden_dim, output_dim)

if args.loss_fn == "MSELoss":
    loss_fn = torch.nn.MSELoss()
else:
    raise ValueError(f"Loss {args.loss_fn} not recognized")

if args.optimizer == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
else:
    raise ValueError(f"Optimizer {args.optimizer} not recognized")



metrics = {}
metrics["learning_rate"] = learning_rate
metrics["hidden_dim"] = hidden_dim
metrics["batch_size"] = batch_size
metrics["n_epochs"] = n_epochs
metrics["optimizer"] = args.optimizer
metrics["loss_fn"] = args.loss_fn


for i_epoch, epoch in enumerate(range(n_epochs)):
    running_loss = 0
    model.train(True)
    for i_data, (inputs, labels) in enumerate(train_loader):

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        running_loss += loss.item()#/batch_size pytorch return the mean by default
        loss.backward()
        optimizer.step()
        
    train_loss_values[i_epoch] = running_loss/len(train_loader)
    
    model.eval()
    running_loss = 0
    
    with torch.no_grad():
        for i_data, (inputs, labels) in enumerate(val_loader):
            
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()
        val_loss_values[i_epoch] = running_loss/len(val_loader)
            
        
    
    if n_epochs >= 10:
        if i_epoch % (n_epochs//10) == 0:
            print(f"Epoch {epoch} loss: {train_loss_values[i_epoch]}")
    else:
        print(f"Epoch {epoch} loss: {train_loss_values[i_epoch]}")


metrics["train_loss"] = train_loss_values.tolist()
metrics["val_loss"] = val_loss_values.tolist()

epoch_f = n_epochs//3
fig ,ax = plt.subplots(2, 1, figsize=(8, 8))
ax[0].plot(np.arange(n_epochs), train_loss_values, label="train loss")
ax[0].plot(np.arange(n_epochs), val_loss_values, label="test loss")
ax[1].plot(np.arange(epoch_f, n_epochs), train_loss_values[epoch_f:])
ax[1].plot(np.arange(epoch_f, n_epochs), val_loss_values[epoch_f:])
ax[0].set(
    xlabel="epoch",
    ylabel="loss"
)
ax[1].set(
    xlabel="epoch",
    ylabel="loss"
)
ax[0].legend()

plt.tight_layout()



name_loss_plot = f"loss.png"
model_name = model.get_name()

training_stats_name = f"batchsize{args.batch_size}_lr{args.learning_rate:.0e}"


plot_path = f"./plots/{model_name}_{args.optimizer}_{args.loss_fn}/{training_stats_name}/"
model_path = f"./models/{model_name}_{args.optimizer}_{args.loss_fn}/{training_stats_name}/"
metric_path = f"./metrics/{model_name}_{args.optimizer}_{args.loss_fn}/{training_stats_name}/"


model_save_name = f"{model_name}_epoch{n_epochs}.pt"
plot_file = os.path.join(plot_path, name_loss_plot)


def make_dir_and_clean(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        for f in os.listdir(path):
            os.remove(os.path.join(path, f))
        
make_dir_and_clean(plot_path)
make_dir_and_clean(model_path)
make_dir_and_clean(metric_path)

plt.savefig(plot_file, dpi=90)
plt.close()


model_file = os.path.join(model_path, model_save_name)
torch.save({
    "epoch":epoch,
    "model_state_dict":model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "loss":val_loss_values[-1]
}, model_file)



metrics_file = os.path.join(metric_path, "metrics.yaml")

with open(metrics_file, "w") as f:
    yaml.safe_dump(metrics, f)



# ############################################################################################################
# # test the performances

# outputs_array = np.empty(test_dataset.labels_shape())
# labels_array = np.empty_like(outputs_array)



# def root_mean_squared_error(y_true, y_pred):
#     return np.sqrt(np.mean((y_true - y_pred)**2))

# def mean_absolute_error(y_true, y_pred):
#     return np.mean(np.abs(y_true - y_pred))

# def mean_squared_error(y_true, y_pred):
#     return np.mean((y_true - y_pred)**2)




# with torch.no_grad():
#     prev_index = 0
#     for i_data, (inputs, labels) in enumerate(test_loader):
        
#         outputs = model(inputs).numpy()
#         labels = labels.numpy()
        
#         outputs_array[prev_index:prev_index+outputs.shape[0]] = outputs
#         labels_array[prev_index:prev_index+outputs.shape[0]] = labels
#         prev_index += outputs.shape[0]


# names = ["count", "masses"]
# total_rmse = 0
# total_mse = 0
# total_mae = 0
# for dim in range(test_dataset.output_len()):        
#     fig, ax = plt.subplots(1,1, figsize=(12, 6))
#     ax.hist(labels_array[:, dim], alpha=0.3, label="true", bins=20)
#     ax.hist(outputs_array[:, dim], alpha=0.3, label="predicted", bins=20)
#     ax.set(title=f"{names[dim]}", xlabel="value", ylabel="count")
#     fig.legend()
#     plt.tight_layout()
#     plot_file_hist = os.path.join(plot_path, "hist_"+names[dim]+".png")
#     plt.savefig(plot_file_hist, dpi=90)
#     plt.close()
#     # plt.show()
    
#     fig, ax = plt.subplots(1,1, figsize=(6, 6))
#     ax.scatter(labels_array[:, dim], outputs_array[:, dim])
#     ax.set(xlabel="true", ylabel="predicted", title=f"{names[dim]}")
#     ax.plot([min(outputs_array[:, dim]), max(outputs_array[:, dim])], [min(outputs_array[:, dim]), max(outputs_array[:, dim])], 'r--')
#     ax.set_aspect('equal', 'box')
#     # fig.legend()
#     plot_file_scat = os.path.join(plot_path, "scatter_"+names[dim]+".png")

#     plt.savefig(plot_file_scat,  dpi=90)
#     plt.close()
    
#     # plt.show()
    
#     rmse = root_mean_squared_error(labels_array[:, dim], outputs_array[:, dim])
#     mae = mean_absolute_error(labels_array[:, dim], outputs_array[:, dim])
#     mse = mean_squared_error(labels_array[:, dim], outputs_array[:, dim])
#     metrics[f"RMSE_{names[dim]}"] = float(rmse)
#     metrics[f"MAE_{names[dim]}"] = float(mae)
#     metrics[f"MSE_{names[dim]}"] = float(mse)
    
#     total_rmse += rmse
#     total_mae += mae
#     total_mse += mse
    
#     print(names[dim])
#     print("rmse:", rmse)
#     print("mae:", mae)
#     print("mse:", mse)    


# metrics[f"total_RMSE"] = float(total_rmse)
# metrics[f"total_MAE"] = float(total_mae)
# metrics[f"total_MSE"] = float(total_mse)


# metrics_file = os.path.join(metric_path, "metrics.yaml")

# with open(metrics_file, "w") as f:
#     yaml.dump(metrics, f)

