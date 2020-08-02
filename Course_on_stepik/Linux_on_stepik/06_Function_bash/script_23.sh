#!/bin/bash

files_creator () # dir_name file_name
{
  full_name=$1/$2
  # check first param to exist
  if [[ ! -e $1 ]]; then
    echo "Dir is not exist, creating $1"
    mkdir $1
  # check first param to eqval dir
  elif [[ ! -d $1 ]]; then
    echo "$1 is not dir, exiting"
    exit 1
  fi
  touch $1/$2
}

again="yes"
while [[ $again == "yes" ]]; do
   # read input from user
   read -p "Enter dir name and file name: " dir_name file_name
   # call function with parametrs from user
   files_creator $dir_name $file_name
   if [[ -f $full_name ]]; then echo "Created $full_name"; fi
   read -p "Again? (yes/no): " again
done

