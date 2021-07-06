#!/bin/bash

dir_path=$1;
echo dir_path is $dir_path;

scipt_cmd='sbatch /tmp/script.sh'
for filename in $dir_path*.json; 
do
    echo start: $filename;
    cp /cs/ep/106/AI/train_in_gpu.sh /tmp/script.sh;
    echo $filename >> /tmp/script.sh;
    echo send job-;
    eval $scipt_cmd; 
    echo '********';
done 
