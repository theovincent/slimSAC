#!/bin/bash

source launch_job/parse_arguments.sh
parse_arguments $@

if ! tmux has-session -t slimsac; then
    tmux new-session -d -s slimsac
    echo "Created new tmux session - slimsac"
fi

tmux send-keys -t slimsac "cd $(pwd)" ENTER
tmux send-keys -t slimsac "source env/bin/activate" ENTER
FRACTION_GPU=$(echo "scale=2 ; 1 / ($LAST_SEED - $FIRST_SEED + 1)" | bc)
tmux send-keys -t slimsac "export XLA_PYTHON_CLIENT_MEM_FRACTION=$FRACTION_GPU" ENTER


echo "launch train $ALGO_NAME local"
for (( seed=$FIRST_SEED; seed<=$LAST_SEED; seed++ ))
do
    tmux send-keys -t slimsac\
    "python3 experiments/$ENV_NAME/$ALGO_NAME.py --experiment_name $EXPERIMENT_NAME --seed $seed $ARGS >> experiments/$ENV_NAME/logs/$EXPERIMENT_NAME/$ALGO_NAME/seed_$seed.out 2>&1 &" ENTER
done
tmux send-keys -t slimsac "wait" ENTER
