#!/bin/bash
set -f
func () # takes three arguments
{
  if [[ $1 = "exit" ]]
  then
    echo "bye"
    exit
  elif [[ !($# -eq 3) ]]
  then
    echo "error"
    exit
  elif [[ $2 = "+" ]]
  then
    let "result = $1 $2 $3"
    echo $result
  elif [[ $2 = "-" ]]
  then
    let "result = $1 $2 $3"
    echo $result
  elif [[ $2 = "*" ]]
  then
    let "result = $1 $2 $3"
    echo $result
  elif [[ $2 = "/" ]]
  then
    let "result = $1 $2 $3"
    echo $result
  elif [[ $2 = "%" ]]
  then
    let "result = $1 $2 $3"
    echo $result
  elif [[ $2 = "**" ]]
  then
    let "result = $1 $2 $3"
    echo $result
  fi
}

while true
do
  read -p "Enter exp: " inp_1 inp_2 inp_3
  func $inp_1 $inp_2 $inp_3
done

