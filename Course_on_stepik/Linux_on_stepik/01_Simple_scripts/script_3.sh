#!/bin/bash

dir_path=~/Documents/Linux
file_path=$dir_path/test.txt

echo "Creating file $file_path"
touch $file_path
echo "Checking content of dir $dir_path"
ls $dir_path
