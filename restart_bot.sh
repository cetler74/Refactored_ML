#!/bin/bash
pkill -f "python app/main.py"
python app/main.py > bot_output.log 2>&1 &
