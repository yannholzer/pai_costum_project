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
LABELS_TYPE=${LABELS_TYPE[0]}

# THE ROOT PATH OF THE PROCESSED DATA WITH THE DATA NAME AND LABELS TYPE SUBFOLDER
PROCESSED_PATH="${PROCESSED_PATH_DATA_NAME}/${LABELS_TYPE}"

####################################################################################################
#== TRAINING PARAMETERS ======================================================

SEED=42
LEARNING_RATE=1e-5
BATCH_SIZE=32
EPOCHS=5000
TRAIN_RATIO=0.77
VALIDATION_RATIO=0.77

HIDDEN_DIM="128,128,128"

MODELS=(
    "NeuralNetvGenerator"
)
MODEL=${MODELS[0]}

LOSSES=(
    "MSELoss"
)
LOSS=${LOSSES[0]}

OPTIMIZERS=(
    "Adam"
)
OPTIMIZER=${OPTIMIZERS[0]}






# PROCESS THE DATA IF NOT ALREADY #typo on purpose for reseting during debugging
if ! [ -f "${PROCESSED_PATH}/train_data.npy" ]; then
    echo "Processing data"
    python3 src/data/processing.py --data_path "${DATA_PATH}" --processed_path "${PROCESSED_PATH}" --labels_type "${LABELS_TYPE}" --seed "${SEED}"
else
    echo "Data already processed"
fi


# TRAIN THE MODEL
echo "Training model"
python3 train.py --processed_path "${PROCESSED_PATH}" --seed ${SEED} --learning_rate ${LEARNING_RATE} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS} --train_ratio ${TRAIN_RATIO} --validation_ratio ${VALIDATION_RATIO} --model ${MODEL} --loss_fn ${LOSS} --optimizer ${OPTIMIZER} --hidden_dim ${HIDDEN_DIM}


# TEST THE MODEL
echo "testing model"
python3 test.py --processed_path "${PROCESSED_PATH}" --seed ${SEED} --learning_rate ${LEARNING_RATE} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS} --model ${MODEL} --loss_fn ${LOSS} --optimizer ${OPTIMIZER} --hidden_dim ${HIDDEN_DIM}


echo "done"






