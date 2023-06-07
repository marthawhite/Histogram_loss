#! /usr/bin/bash

export KERASTUNER_TUNER_ID="chief"
export KERASTUNER_ORACLE_IP="127.0.0.1"
export KERASTUNER_ORACLE_PORT="8000"
python3 $1 $2 0
tar cf hypers.tar -C $2 hypers