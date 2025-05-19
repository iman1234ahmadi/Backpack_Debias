CONDA_BASE="${HOME}/miniconda3"
CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"


if [ -f "${CONDA_SH}" ]; then
    source "${CONDA_SH}"
else
    echo "Could not find conda.sh at ${CONDA_SH}; please adjust CONDA_SH path."
    exit 1
fi

declare -A models
models=(
    ["0"]="BP_ranker_freeze_sensenet_final"
    ["1"]="BP_ranker_freeze_sensenet_final_oneGpu"
    ["2"]="BP_ranker_no_freeze_sensenet_final"
)

for j in {0..0}; do
    model_name=${models[$j]}
    for i in {0..6}; do
        session_name="session_${i}_${model_name}"
        echo "Starting screen session: ${session_name}"
        screen -dmS "${session_name}" bash -lc "source activate base && conda activate amir && cd Dev_MS_debias_Final && python main.py --id=${i} --model_name=${model_name} --GPU_CUDA=${i} "
    done
done

echo "All screen sessions started."