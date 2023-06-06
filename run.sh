#! /usr/bin/bash
#./run.sh python_file data_folder num_workers
source chief.sh $1 $2 &
for i in {1..4}
do
    source worker.sh $1 $2 $i &
done