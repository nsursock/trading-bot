#!/bin/zsh

# Define the directory containing the models
MODEL_DIR="./models"

# Define the number of tests to run
NUM_TESTS=50

# Define the list of action selection methods to test
METHODS=("averaging" "bagging" "stacking" "voting" "mediating" "maximizing" "minimizing")

# Iterate over each method
for METHOD in "${METHODS[@]}"; do
    # Use AppleScript to open a new iTerm2 tab and run the test command
    osascript <<EOF
    tell application "iTerm2"
        tell current window
            create tab with default profile
            tell current session of current tab
                write text "source ~/venv-metal/bin/activate; cd Sites/trading-bot/v6; clear; python test_model.py -m $MODEL_DIR -n $NUM_TESTS -a $METHOD"
            end tell
        end tell
    end tell
EOF
done