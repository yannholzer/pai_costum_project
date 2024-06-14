from src.data.utils import get_traintest_args
from torch.utils.data.dataloader import DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import yaml
import sys

args = get_traintest_args()
processed_path = args.processed_path


if "SET" in args.labels_type:
    from src.data.dataset_set import DiskPlanetDataset
else:
    from src.data.dataset import DiskPlanetDataset
train_dataset = DiskPlanetDataset(os.path.join(processed_path, "train.npy"))
val_dataset = DiskPlanetDataset(os.path.join(processed_path, "val.npy"))


batch_size = args.batch_size
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)



n_epochs = args.epochs
train_loss_values = np.ones(n_epochs)*1e10
val_loss_values = np.ones(n_epochs)*1e10


input_dim = train_dataset.input_len()
output_dim = train_dataset.output_len()

hidden_dim = args.hidden_dim

learning_rate = args.learning_rate


metrics = {}
metrics["learning_rate"] = learning_rate
metrics["hidden_dim"] = hidden_dim
metrics["batch_size"] = batch_size
metrics["n_epochs"] = n_epochs


if args.model == "NeuralNetGenerator":
    from src.model.models import NeuralNetvGenerator as Model
    
    model = Model(input_dim, hidden_dim, output_dim)

    
elif args.model =="NeuralNetGeneratorRegularized":
    from src.model.models import NeuralNetvGeneratorRegularized as Model
    model = Model(input_dim, hidden_dim, output_dim, dropout=args.dropout, batchnorm=args.batchnorm, elementwise=args.elementwise)
    metrics["dropout"] = args.dropout
    metrics["batchnorm"] = args.batchnorm

else:
    raise ValueError(f"Model {args.model} not recognized")


if args.loss_fn == "MSELoss":
    loss_fn = torch.nn.MSELoss()
    metrics["loss_fn"] = args.loss_fn
else:
    raise ValueError(f"Loss {args.loss_fn} not recognized")

if args.optimizer == "Adam":
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    metrics["optimizer"] = args.optimizer
else:
    raise ValueError(f"Optimizer {args.optimizer} not recognized")



name_loss_plot = f"loss.png"
model_name = model.get_name()

training_stats_name = f"batchsize{args.batch_size}_lr{args.learning_rate:.0e}_epochs{n_epochs}"



plot_path = f"./plots/{args.labels_type}/{model_name}_{args.optimizer}_{args.loss_fn}/{training_stats_name}/"
model_path = f"./models/{args.labels_type}/{model_name}_{args.optimizer}_{args.loss_fn}/{training_stats_name}/"
metric_path = f"./metrics/{args.labels_type}/{model_name}_{args.optimizer}_{args.loss_fn}/{training_stats_name}/"


model_save_name = f"{model_name}_epoch{n_epochs}.pt"
plot_file = os.path.join(plot_path, name_loss_plot)


def make_dir_and_clean(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        for f in os.listdir(path):
            if args.reset:
                os.remove(os.path.join(path, f))
            else:
                if ".pt" in f:
                    print("model already trained.")
                    sys.exit(0)
        # for f in os.listdir(path):
        #     os.remove(os.path.join(path, f))

        
make_dir_and_clean(model_path)
make_dir_and_clean(plot_path)
make_dir_and_clean(metric_path)

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
        
    if val_loss_values[i_epoch] == np.min(val_loss_values):
        torch.save({
            "epoch":epoch,
            "model_state_dict":model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss":val_loss_values[i_epoch]
        }, os.path.join(model_path, model_save_name.replace(f"epoch{n_epochs}.pt", f"epoch{n_epochs}_best.pt")))


metrics["train_loss"] = train_loss_values.tolist()
metrics["val_loss"] = val_loss_values.tolist()

epoch_f = n_epochs//3
fig ,ax = plt.subplots(2, 1, figsize=(8, 8))
ax[0].plot(np.arange(n_epochs), train_loss_values, label="train loss")
ax[0].plot(np.arange(n_epochs), val_loss_values, label="val loss")
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
