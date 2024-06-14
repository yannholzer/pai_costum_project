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
    "SET_all"
    "SET_masses"
    "SET_FLAG_masses"
)
LABEL_TYPE=${LABELS_TYPE[0]}

# THE ROOT PATH OF THE PROCESSED DATA WITH THE DATA NAME AND LABELS TYPE SUBFOLDER
PROCESSED_PATH="${PROCESSED_PATH_DATA_NAME}/${LABEL_TYPE}"

####################################################################################################
#== TRAINING PARAMETERS ======================================================



N_LOOP=2

SEED=-1
BATCH_SIZE=(32)
EPOCHS=1000
TRAIN_RATIO=0.77
VALIDATION_RATIO=0.5
DROPOUT=(0)
BATCHNORM=0
ELEMENTWISE=0




HIDDEN_DIM=("32" "32,32" "32,32,32")

#HIDDEN_DIM=("64" "64,64" "64,64,64" "64,64,64,64" "64,64,64,64,64" "64,64,64,64,64,64" "64,64,64,64,64,64,64" "64,64,64,64,64,64,64,64")
LEARNING_RATE=(5e-5 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1)
#LEARNING_RATE=(1e-4)



MODELS=(
    "NeuralNetGenerator"
    "NeuralNetGeneratorRegularized"
)
#MODEL=${MODELS[1]}
MODEL=("NeuralNetGeneratorRegularized")

LOSSES=(
    "MSELoss"
)
LOSS=${LOSSES[0]}

OPTIMIZERS=(
    "Adam"
)
OPTIMIZER=${OPTIMIZERS[0]}

TEST_NAME="final_lr_test"



N_CORE=10

RESET_DATA=0

RESET_TRAIN=1

JOB_OUTPUT=out


rm -rf ${JOB_OUTPUT}
mkdir -p ${JOB_OUTPUT}


# PROCESS THE DATA IF NOT ALREADY
if [ ${RESET_DATA} -eq 1 ]; then
    echo "Resetting data"
    rm -rf "${PROCESSED_PATH}"
fi

if [[ $LABEL_TYPE == *"SET"* ]]; then
  
    if ! [ -f "${PROCESSED_PATH}/train.npy" ]; then
    
        echo "Processing data set format"
        python3 src/data/processing_set.py --data_path "${DATA_PATH}" --processed_path "${PROCESSED_PATH}" --labels_type "${LABEL_TYPE}" --seed "${SEED}" --validation_ratio "${VALIDATION_RATIO}" --train_ratio "${TRAIN_RATIO}"
    else
        echo "Data already processed"
    fi

else

    if ! [ -f "${PROCESSED_PATH}/train.npy" ]; then
        

        echo "Processing data"
        python3 src/data/processing.py --data_path "${DATA_PATH}" --processed_path "${PROCESSED_PATH}" --labels_type "${LABEL_TYPE}" --seed "${SEED}" --validation_ratio "${VALIDATION_RATIO}" --train_ratio "${TRAIN_RATIO}"
    else
        echo "Data already processed"
    fi

fi


# repeat training N_LOOP times
for i in $(eval echo {0..$N_LOOP}); do 


echo "=========================================="
echo "loop ${i}"
echo "=========================================="

#parallel -j ${N_CORE} echo {1} {2} {3}  ::: ${LEARNING_RATE[@]} ::: ${HIDDEN_DIM[@]} ::: ${MODEL[@]}

# TRAIN THE MODEL
echo "Training model"


command_train="python3 train.py --processed_path ${PROCESSED_PATH} --labels_type ${LABEL_TYPE} --seed ${SEED} --learning_rate {1} --batch_size {5} --epochs ${EPOCHS} --model {3} --loss_fn ${LOSS} --optimizer ${OPTIMIZER} --hidden_dim {2} --dropout {4} --batchnorm ${BATCHNORM} --elementwise ${ELEMENTWISE} --reset ${RESET_TRAIN} > out/train_out$i-{#}.txt; printf '\--- train job {#} done \---\nlr={1}\nhdim={2}\nmodel={3}\ndrpout={4}\nbatchsize={5}\n'"

parallel -j ${N_CORE} --joblog out/joblog_train.txt $command_train ::: ${LEARNING_RATE[@]}  ::: ${HIDDEN_DIM[@]} ::: ${MODEL[@]} ::: ${DROPOUT[@]} ::: ${BATCH_SIZE[@]}



# TEST THE MODEL
echo "testing model"

command_test="python3 test.py --processed_path "${PROCESSED_PATH}" --labels_type ${LABEL_TYPE} --seed ${SEED} --learning_rate {1} --batch_size {5} --epochs ${EPOCHS} --model {3} --loss_fn ${LOSS} --optimizer ${OPTIMIZER} --hidden_dim {2} --dropout {4} --batchnorm ${BATCHNORM} --elementwise ${ELEMENTWISE} --reset ${RESET_TRAIN} > out/test_out$i-{#}.txt; printf '\--- test job {#} done \---\nlr={1}\nhdim={2}\nmodel={3}\ndrpout={4}\nbatchsize={5}\n'"


parallel -j ${N_CORE} --joblog out/joblog_test.txt $command_test ::: ${LEARNING_RATE[@]}  ::: ${HIDDEN_DIM[@]} ::: ${MODEL[@]} ::: ${DROPOUT[@]} ::: ${BATCH_SIZE[@]}




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
python3 gather_metrics.py --processed_path "${PROCESSED_PATH}" --labels_type ${LABEL_TYPE} --seed ${SEED} --learning_rate ${lr} --batch_size ${bs} --epochs ${EPOCHS} --model ${m} --loss_fn ${LOSS} --optimizer ${OPTIMIZER} --hidden_dim ${h} --test_name ${TEST_NAME} --dropout ${d} --batchnorm ${BATCHNORM} --elementwise ${ELEMENTWISE}
                done
            done
        done
    done
done

echo "done"


# main loop done
done

exit 0