#!/bin/sh
echo 'Starting server...'

cd ./1_ore_dispensing/src/
python build_model.py|tee log.log

echo 'Done'

