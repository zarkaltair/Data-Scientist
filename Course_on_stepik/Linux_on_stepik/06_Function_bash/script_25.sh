#!/bin/bash

gcd () # takes two arguments
{
  if [[ !($# -eq 2) ]]
  then
    echo "bye"
    exit
  elif [[ $1 = $2 ]]
  then
    echo "$1 and $2"
    echo "GCD is $1"
  elif [[ $1 -gt $2 ]]
  then
    let "M = $1-$2"
    echo "$M $2"
    gcd $M $2
  elif [[ $1 -lt $2 ]]
  then
    let "N = $2 - $1"
    echo "$2 $N"
    gcd $2 $N
  fi
}

while true
do
  read -p "Enter two numbers: " num_1 num_2
  gcd $num_1 $num_2
done

