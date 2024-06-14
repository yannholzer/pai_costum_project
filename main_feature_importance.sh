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


BATCH_SIZE=32
EPOCHS=200
TRAIN_RATIO=0.77
VALIDATION_RATIO=0.77
DROPOUT=0.1
BATCHNORM=0
ELEMENTWISE=1

HIDDEN_DIM="32,32"

LEARNING_RATE=1e-3


MODELS=(
    "NeuralNetGenerator"
    "NeuralNetGeneratorRegularized"
)

MODEL=${MODELS[1]}

LOSSES=(
    "MSELoss"
)
LOSS=${LOSSES[0]}

OPTIMIZERS=(
    "Adam"
)
OPTIMIZER=${OPTIMIZERS[0]}


PERMUTATION=1


# FEATURE IMPORTANCE
echo "feature importance processing"

if [ ${PERMUTATION} -eq 1 ]; then
    echo "feature importance permutation"
    python3 feature_importance_permutation.py --processed_path "${PROCESSED_PATH}" --labels_type ${LABEL_TYPE} --learning_rate ${LEARNING_RATE} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS} --model ${MODEL} --loss_fn ${LOSS} --optimizer ${OPTIMIZER} --hidden_dim ${HIDDEN_DIM} --dropout ${DROPOUT} --batchnorm ${BATCHNORM} --elementwise ${ELEMENTWISE}
else
    echo "feature importance EW weights"
    python3 feature_importance_EW_weight.py --processed_path "${PROCESSED_PATH}" --labels_type ${LABEL_TYPE} --learning_rate ${LEARNING_RATE} --batch_size ${BATCH_SIZE} --epochs ${EPOCHS} --model ${MODEL} --loss_fn ${LOSS} --optimizer ${OPTIMIZER} --hidden_dim ${HIDDEN_DIM} --dropout ${DROPOUT} --batchnorm ${BATCHNORM} --elementwise ${ELEMENTWISE}
fi

echo "done"


# main loop done

exit 0