from src.data.utils import get_traintest_args
from src.data.dataset import DiskPlanetDataset
from torch.utils.data.dataloader import DataLoader
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
import sys

args = get_traintest_args()


# load the data
processed_path = args.processed_path
batch_size = args.batch_size

if "SET" in args.labels_type:
    from src.data.dataset_set import DiskPlanetDataset
else:
    from src.data.dataset import DiskPlanetDataset

test_dataset = DiskPlanetDataset(os.path.join(processed_path, "test.npy"))
    

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)


input_dim = test_dataset.input_len()
output_dim = test_dataset.output_len()
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

metric_path = f"./metrics/{args.labels_type}/{model_name}_{args.optimizer}_{args.loss_fn}/{training_stats_name}/metrics.yaml"
plot_path = f"./plots/{args.labels_type}/{model_name}_{args.optimizer}_{args.loss_fn}/{training_stats_name}/"
model_path = f"./models/{args.labels_type}/{model_name}_{args.optimizer}_{args.loss_fn}/{training_stats_name}/{model_name}_epoch{args.epochs}_best.pt"



for f in os.listdir(plot_path):
    if not "loss" in f:
        print("model already tested")
        sys.exit(0)
    

state_dict = torch.load(model_path)
model.load_state_dict(state_dict["model_state_dict"])
model.eval()



# get the og metrics file
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


names = test_dataset.labels_names
print(names)
total_rmse = 0
total_mse = 0
total_mae = 0
n_planets=20 #hardcoded for now


        
if "SET_FLAG" in args.labels_type:
    planet_count = np.zeros((outputs_array.shape[0], 2)) # [true, pred]
    total_mass = np.zeros((outputs_array.shape[0], 2)) # [true, pred]
    flag_pred = outputs_array[:, 0::len(names)]
    flag_true = labels_array[:, 0::len(names)]
    for i_l, lbl in enumerate(names):
        lbl = lbl.replace("(", "_").replace(")", "_").replace(" ", "_")
        pred = outputs_array[:, i_l::len(names)]
        true = labels_array[:, i_l::len(names)]
        if "Mass" in lbl:
            for system in range(outputs_array.shape[0]):
                continue
                # exist_true = true[system, :] >= mass_default + 0.1
                # exist_pred = pred[system, :] >= mass_default + 0.1
                
                # mass_true = true[system, :][exist_true]
                # mass_pred = pred[system, :][exist_pred]
                # mined, maxed = test_dataset.min_maxs[i_l]
                # denorm_mass_true = 0.5*(mass_true + 1)*(maxed-mined) + mined
                # denorm_mass_pred = 0.5*(mass_pred + 1)*(maxed-mined) + mined
                # delog_mass_true = 10**(denorm_mass_true)-1e-5
                # delog_mass_pred = 10**(denorm_mass_pred)-1e-5
                
                # planet_count[system, 0] = np.sum(exist_true)
                # planet_count[system, 1] = np.sum(exist_pred)
            
                # total_mass[system, 0] = np.sum(delog_mass_true)
                # total_mass[system, 1] = np.sum(delog_mass_pred)
                # total_mass[system, 0] = np.log10(total_mass[system, 0]+1e-5)
                # total_mass[system, 1] = np.log10(total_mass[system, 1]+1e-5)
                
        fig, ax = plt.subplots(1,1, figsize=(12, 6))
            
        ax.hist(true, alpha=0.3, label="true", bins=20, color=["b"]*20)
        ax.hist(pred, alpha=0.3, label="predicted", bins=20, color=["r"]*20)
        
                
        ax.set(title=f"{lbl}", xlabel="value", ylabel="count")
        fig.legend()
        plt.tight_layout()
        plot_file_hist = os.path.join(plot_path, "hist_"+lbl+".png")
        plt.savefig(plot_file_hist, dpi=90)
        plt.close()
    
        fig, ax = plt.subplots(1,1, figsize=(6, 6))
        ax.scatter(true, pred, color="blue", alpha=0.3)
        ax.set(xlabel="true", ylabel="predicted", title=f"{lbl}")
        ax.plot([np.min(pred), np.max(pred)], [np.min(pred), np.max(pred)], 'r--')
        ax.set_aspect('equal', 'box')
        # fig.legend()
        plot_file_scat = os.path.join(plot_path, "scatter_"+lbl+".png")

        plt.savefig(plot_file_scat,  dpi=90)
        plt.close()
    
        # plt.show()
    
        rmse = root_mean_squared_error(true, pred)
        mae = mean_absolute_error(true, pred)
        mse = mean_squared_error(true, pred)
        metrics[f"RMSE_{lbl}"] = float(rmse)
        metrics[f"MAE_{lbl}"] = float(mae)
        metrics[f"MSE_{lbl}"] = float(mse)
        
        total_rmse += rmse
        total_mae += mae
        total_mse += mse
        
        print("rmse:", rmse)
        print("mae:", mae)
        print("mse:", mse)    

        
    # calculated = [planet_count, total_mass]
    # for i_l, lbl in enumerate(["system_planet_count", "system_total_mass"]):
    #     fig, ax = plt.subplots(1,1, figsize=(12, 6))
    #     ax.hist(calculated[i_l][:, 0], alpha=0.3, label="true", bins=20)
    #     ax.hist(calculated[i_l][:, 1], alpha=0.3, label="predicted", bins=20)
    #     ax.set(title=f"{lbl}", xlabel="value", ylabel="count")
    #     fig.legend()
    #     plt.tight_layout()
    #     plot_file_hist = os.path.join(plot_path, "hist_"+lbl+".png")
    #     plt.savefig(plot_file_hist, dpi=90)
    #     plt.close()

    #     fig, ax = plt.subplots(1,1, figsize=(6, 6))
    #     ax.scatter(calculated[i_l][:, 0], calculated[i_l][:, 1])
    #     ax.set(xlabel="true", ylabel="predicted", title=f"{lbl}")
    #     ax.plot([np.min(calculated[i_l][:, 1]), np.max(calculated[i_l][:, 1])], [np.min(calculated[i_l][:, 1]), np.max(calculated[i_l][:, 1])], 'r--')
    #     ax.set_aspect('equal', 'box')
    #     # fig.legend()
    #     plot_file_scat = os.path.join(plot_path, "scatter_"+lbl+".png")

    #     plt.savefig(plot_file_scat,  dpi=90)
    #     plt.close()

    #     rmse = root_mean_squared_error(calculated[i_l][:, 0], calculated[i_l][:, 1])
    #     mae = mean_absolute_error(calculated[i_l][:, 0], calculated[i_l][:, 1])
    #     mse = mean_squared_error(calculated[i_l][:, 0], calculated[i_l][:, 1])
    #     metrics[f"RMSE_{lbl}"] = float(rmse)
    #     metrics[f"MAE_{lbl}"] = float(mae)
    #     metrics[f"MSE_{lbl}"] = float(mse)

    metrics[f"total_RMSE"] = float(total_rmse)
    metrics[f"total_MAE"] = float(total_mae)
    metrics[f"total_MSE"] = float(total_mse)

        
                

elif "SET" in args.labels_type:
    planet_count = np.zeros((outputs_array.shape[0], 2)) # [true, pred]
    total_mass = np.zeros((outputs_array.shape[0], 2)) # [true, pred]
    for i_l, lbl in enumerate(names):
        lbl = lbl.replace("(", "_").replace(")", "_").replace(" ", "_")
        pred = outputs_array[:, i_l::len(names)]
        true = labels_array[:, i_l::len(names)]
        if "Mass" in lbl:
            mass_default = -1
            for system in range(outputs_array.shape[0]):
                exist_true = true[system, :] >= mass_default + 0.1
                exist_pred = pred[system, :] >= mass_default + 0.1
                
                mass_true = true[system, :][exist_true]
                mass_pred = pred[system, :][exist_pred]
                mined, maxed = test_dataset.min_maxs[i_l]
                denorm_mass_true = 0.5*(mass_true + 1)*(maxed-mined) + mined
                denorm_mass_pred = 0.5*(mass_pred + 1)*(maxed-mined) + mined
                delog_mass_true = 10**(denorm_mass_true)-1e-5
                delog_mass_pred = 10**(denorm_mass_pred)-1e-5
                
                planet_count[system, 0] = np.sum(exist_true)
                planet_count[system, 1] = np.sum(exist_pred)
            
                total_mass[system, 0] = np.sum(delog_mass_true)
                total_mass[system, 1] = np.sum(delog_mass_pred)
                total_mass[system, 0] = np.log10(total_mass[system, 0]+1e-5)
                total_mass[system, 1] = np.log10(total_mass[system, 1]+1e-5)
                
        fig, ax = plt.subplots(1,1, figsize=(12, 6))
        for p in range(n_planets):
            if p < 1:
                ax.hist(true[:, p], alpha=0.3, label="true", bins=20, color="b")
                ax.hist(pred[:, p], alpha=0.3, label="predicted", bins=20, color="r")
            else:
                ax.hist(true[:, p], alpha=0.3, bins=20, color="b")
                ax.hist(pred[:, p], alpha=0.3, bins=20, color="r")
                
        ax.set(title=f"{lbl}", xlabel="value", ylabel="count")
        fig.legend()
        plt.tight_layout()
        plot_file_hist = os.path.join(plot_path, "hist_"+lbl+".png")
        plt.savefig(plot_file_hist, dpi=90)
        plt.close()
    
        fig, ax = plt.subplots(1,1, figsize=(6, 6))
        for p in range(n_planets):
            ax.scatter(true[:, p], pred[:, p], color="blue", alpha=0.3)
        ax.set(xlabel="true", ylabel="predicted", title=f"{lbl}")
        ax.plot([np.min(pred), np.max(pred)], [np.min(pred), np.max(pred)], 'r--')
        ax.set_aspect('equal', 'box')
        # fig.legend()
        plot_file_scat = os.path.join(plot_path, "scatter_"+lbl+".png")

        plt.savefig(plot_file_scat,  dpi=90)
        plt.close()
    
        # plt.show()
    
        rmse = root_mean_squared_error(true, pred)
        mae = mean_absolute_error(true, pred)
        mse = mean_squared_error(true, pred)
        metrics[f"RMSE_{lbl}"] = float(rmse)
        metrics[f"MAE_{lbl}"] = float(mae)
        metrics[f"MSE_{lbl}"] = float(mse)
        
        total_rmse += rmse
        total_mae += mae
        total_mse += mse
        
        print("rmse:", rmse)
        print("mae:", mae)
        print("mse:", mse)    

        
    calculated = [planet_count, total_mass]
    for i_l, lbl in enumerate(["system_planet_count", "system_total_mass"]):
        fig, ax = plt.subplots(1,1, figsize=(12, 6))
        ax.hist(calculated[i_l][:, 0], alpha=0.3, label="true", bins=20)
        ax.hist(calculated[i_l][:, 1], alpha=0.3, label="predicted", bins=20)
        ax.set(title=f"{lbl}", xlabel="value", ylabel="count")
        fig.legend()
        plt.tight_layout()
        plot_file_hist = os.path.join(plot_path, "hist_"+lbl+".png")
        plt.savefig(plot_file_hist, dpi=90)
        plt.close()

        fig, ax = plt.subplots(1,1, figsize=(6, 6))
        ax.scatter(calculated[i_l][:, 0], calculated[i_l][:, 1])
        ax.set(xlabel="true", ylabel="predicted", title=f"{lbl}")
        ax.plot([np.min(calculated[i_l][:, 1]), np.max(calculated[i_l][:, 1])], [np.min(calculated[i_l][:, 1]), np.max(calculated[i_l][:, 1])], 'r--')
        ax.set_aspect('equal', 'box')
        # fig.legend()
        plot_file_scat = os.path.join(plot_path, "scatter_"+lbl+".png")

        plt.savefig(plot_file_scat,  dpi=90)
        plt.close()

        rmse = root_mean_squared_error(calculated[i_l][:, 0], calculated[i_l][:, 1])
        mae = mean_absolute_error(calculated[i_l][:, 0], calculated[i_l][:, 1])
        mse = mean_squared_error(calculated[i_l][:, 0], calculated[i_l][:, 1])
        metrics[f"RMSE_{lbl}"] = float(rmse)
        metrics[f"MAE_{lbl}"] = float(mae)
        metrics[f"MSE_{lbl}"] = float(mse)

    metrics[f"total_RMSE"] = float(total_rmse)
    metrics[f"total_MAE"] = float(total_mae)
    metrics[f"total_MSE"] = float(total_mse)


else:

    for dim in range(test_dataset.output_len()):
        true, pred = labels_array[:, dim], outputs_array[:, dim]        
        fig, ax = plt.subplots(1,1, figsize=(12, 6))
        
        mined, maxed = test_dataset.mins_maxs[:, dim]
    
        true = 0.5*(true + 1)*(maxed-mined) + mined
        pred = 0.5*(pred + 1)*(maxed-mined) + mined
        
        # if "mass" in names[dim]:
        #     true = 10**(true)-1e-5
        #     pred = 10**(pred)-1e-5
            
        ax.hist(true, alpha=0.3, label="true", bins=20)
        ax.hist(pred, alpha=0.3, label="predicted", bins=20)
        ax.set(title=f"{names[dim]}", xlabel="value", ylabel="count")
        fig.legend()
        plt.tight_layout()
        plot_file_hist = os.path.join(plot_path, "hist_"+names[dim]+".png")
        plt.savefig(plot_file_hist, dpi=90)
        plt.close()
        # plt.show()
        
        fig, ax = plt.subplots(1,1, figsize=(6, 6))
        ax.scatter(true, pred)
        ax.set(xlabel="true", ylabel="predicted", title=f"{names[dim]}")
        ax.plot([np.min(pred), np.max(pred)], [np.min(pred), max(pred)], 'r--')
        ax.set_aspect('equal', 'box')
        # fig.legend()
        plot_file_scat = os.path.join(plot_path, "scatter_"+names[dim]+".png")

        plt.savefig(plot_file_scat,  dpi=90)
        plt.close()
        
        # plt.show()
        
        rmse = root_mean_squared_error(true, pred)
        mae = mean_absolute_error(true, pred)
        mse = mean_squared_error(true, pred)
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
