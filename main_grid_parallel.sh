#!/bin/bash
set -e


####################################################################################################
# == PATH PARAMETERS ===============================================================================

# THE ROOT PATH OF THE DATA
DATA_PATH_ROOT="/home/yannh/Documents/uni/phd/classes/pai/costum_project/dataset/raw_standardized"

# THE DIFFERENT DATAS
DATA_NAMES=(
    "J20_1MS"
    "J20_02MS"
    "J20_05MS"
    )

# CHOSE THE ONE I WANT TO USE
DATA_NAME=${DATA_NAMES[0]}
DATA_PATH="${DATA_PATH_ROOT}/${DATA_NAME}.csv"

# THE ROOT PATH OF THE PROCESSED DATA
PROCESSED_PATH_ROOT="/home/yannh/Documents/uni/phd/classes/pai/costum_project/dataset/processed"

# THE ROOT PATH OF THE PROCESSED DATA WITH THE DATA NAME SUBFOLDER
PROCESSED_PATH_DATA_NAME="${PROCESSED_PATH_ROOT}/${DATA_NAME}"


####################################################################################################
#== PATH TESTS ======================================================
if ! test -d "${DATA_PATH_ROOT}"; then
    echo "Invalid data root path"
    echo "${DATA_PATH_ROOT}"
    exit 1
fi

####################################################################################################
#== LABELS TYPE ======================================================

# THE LABELS TYPE
LABELS_TYPE=(
    "mass_and_count"
)
LABEL_TYPE=${LABELS_TYPE[0]}

# THE ROOT PATH OF THE PROCESSED DATA WITH THE DATA NAME AND LABELS TYPE SUBFOLDER
PROCESSED_PATH="${PROCESSED_PATH_DATA_NAME}/${LABELS_TYPE}"

####################################################################################################
#== TRAINING PARAMETERS ======================================================

SEED=-1
BATCH_SIZE=(32)
EPOCHS=150
TRAIN_RATIO=0.77
VALIDATION_RATIO=0.77
DROPOUT=(0)



HIDDEN_DIM=("32" "32,32" "32,32,32" "32,32,32,32" "32,32,32,32,32" "32,32,32,32,32,32" "32,32,32,32,32,32,32" "32,32,32,32,32,32,32,32")
LEARNING_RATE=(1e-3)



MODELS=(
    "NeuralNetGenerator"
    "NeuralNetGeneratorRegularizedNoBN"
)
#MODEL=${MODELS[1]}
MODEL=(""NeuralNetGenerator"")

LOSSES=(
    "MSELoss"
)
LOSS=${LOSSES[0]}

OPTIMIZERS=(
    "Adam"
)
OPTIMIZER=${OPTIMIZERS[0]}

TEST_NAME="hdim_test"


N_CORE=12

RESET_DATA=0
RESET_TRAIN=1
KEEP_METRICS=0


# PROCESS THE DATA IF NOT ALREADY
if [ ${RESET_DATA} -eq 1 ]; then
    echo "Resetting data"
    rm -rf "${PROCESSED_PATH}"
fi
if ! [ -f "${PROCESSED_PATH}/train.csv" ]; then
    echo "Processing data"
    python3 src/data/processing.py --data_path "${DATA_PATH}" --processed_path "${PROCESSED_PATH}" --labels_type "${LABELS_TYPE}" --seed "${SEED}" --validation_ratio "${VALIDATION_RATIO}" --train_ratio "${TRAIN_RATIO}"
else
    echo "Data already processed"
fi


#parallel -j ${N_CORE} echo {1} {2} {3}  ::: ${LEARNING_RATE[@]} ::: ${HIDDEN_DIM[@]} ::: ${MODEL[@]}

# TRAIN THE MODEL
echo "Training model"
parallel -j ${N_CORE} python3 train.py --processed_path "${PROCESSED_PATH}" --labels_type ${LABEL_TYPE} --seed ${SEED} --learning_rate {1} --batch_size {5} --epochs ${EPOCHS} --model {3} --loss_fn ${LOSS} --optimizer ${OPTIMIZER} --hidden_dim {2} --dropout {4} --reset ${RESET_TRAIN} --keep_metrics ${KEEP_METRICS} > /dev/null ::: ${LEARNING_RATE[@]}  ::: ${HIDDEN_DIM[@]} ::: ${MODEL[@]} ::: ${DROPOUT[@]} ::: ${BATCH_SIZE[@]}

# TEST THE MODEL
echo "testing model"
parallel -j ${N_CORE} python3 test.py --processed_path "${PROCESSED_PATH}" --labels_type ${LABEL_TYPE} --seed ${SEED} --learning_rate {1} --batch_size {5} --epochs ${EPOCHS} --model {3} --loss_fn ${LOSS} --optimizer ${OPTIMIZER} --hidden_dim {2} --dropout {4} --reset ${RESET_TRAIN} --keep_metrics ${KEEP_METRICS} > /dev/null ::: ${LEARNING_RATE[@]}  ::: ${HIDDEN_DIM[@]} ::: ${MODEL[@]} ::: ${DROPOUT[@]} ::: ${BATCH_SIZE[@]}




echo "gather metrics"
for h in ${HIDDEN_DIM[@]}; do
    #echo "hidden dim: ${h}"
    for lr in ${LEARNING_RATE[@]}; do
    #    echo "learning rate: ${lr}"
        for m in ${MODEL[@]}; do
    #        echo "model: ${m}"
            for d in ${DROPOUT[@]}; do
    #            echo "dropout: ${d}"
                for bs in ${BATCH_SIZE[@]}; do
python3 gather_metrics.py --processed_path "${PROCESSED_PATH}" --labels_type ${LABEL_TYPE} --seed ${SEED} --learning_rate ${lr} --batch_size ${bs} --epochs ${EPOCHS} --model ${m} --loss_fn ${LOSS} --optimizer ${OPTIMIZER} --hidden_dim ${h} --test_name ${TEST_NAME} --dropout ${d}
                done
            done
        done
    done
done

echo "done"
exit 0






