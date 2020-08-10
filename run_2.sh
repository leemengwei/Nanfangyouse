#!/bin/sh
echo 'Starting server...'

cd ./2_metal_balance/src/
python build_model.py|tee log.log

echo 'Done'

