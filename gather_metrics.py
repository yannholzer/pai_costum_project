from src.data.utils import get_traintest_args
import os
import yaml


args = get_traintest_args()


batch_size = args.batch_size
hidden_dim = args.hidden_dim

# load the model
if args.model == "NeuralNetGenerator":
    str_hidden_dim = "_".join([str(h) for h in hidden_dim])
    model_name = f"NeuralNetGenerator_Nh{len(hidden_dim)}_hdim{str_hidden_dim}"
elif args.model == "NeuralNetGeneratorRegularized":
    str_hidden_dim = "_".join([str(h) for h in hidden_dim])
    model_name = f"NeuralNetGeneratorRegularized_Nh{len(hidden_dim)}_hdim{str_hidden_dim}_dp{args.dropout:.0e}_BN{int(args.batchnorm)}"
    if args.elementwise:
        model_name += "_EW"
else: 
    raise ValueError(f"Model {args.model} not recognized")



training_stats_name = f"batchsize{args.batch_size}_lr{args.learning_rate:.0e}_epochs{args.epochs}"

metric_path = f"./metrics/{args.labels_type}/{model_name}_{args.optimizer}_{args.loss_fn}/{training_stats_name}/metrics.yaml"

with open(metric_path, "r") as f:
    model_metrics = yaml.safe_load(f)


test_name = args.test_name

gathered_metrics_path = f"./results/{test_name}/gathered_metrics.yaml"

os.makedirs(f"./results/{test_name}", exist_ok=True)

try:
    with open(gathered_metrics_path, "r") as f:
        gathered_metrics = yaml.safe_load(f)
except FileNotFoundError:
    gathered_metrics = {}

model_key = f"{model_name}_lr{args.learning_rate:.0e}_bs{args.batch_size}"

while model_key in gathered_metrics:
    model_key = model_key + "-1"
    

gathered_metrics.update({model_key: {}})

gathered_metrics[model_key].update(model_metrics)

with open(gathered_metrics_path, "w") as f:
    yaml.dump(gathered_metrics, f)
