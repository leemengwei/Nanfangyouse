#!/bin/bash
#username:mom
#ip:192.168.230.210
#password:a2298508@

echo "\nTransfering..., ip:192.168.230.210, username:mom, password:a2298508@"
echo "********************************"
scp ./data/*.csv mom@192.168.230.210:/home/mom/Nanfangyouse/1_ore_dispensing/data
echo "********************************"
echo "OK!"
