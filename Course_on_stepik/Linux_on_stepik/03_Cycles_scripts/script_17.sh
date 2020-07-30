#!/bin/bash

while true
do
  echo "enter your name:"
  read name
  if [[ $name = "" ]]
  then
    echo "bye"
    break
  fi
  echo "enter your age:"
  read age
  if [[ $age = 0 ]]
  then
    echo "bye"
    break
  elif [[ $age -le 16 ]]
  then
    echo "$name, your group is child"
  elif [[ $age -le 25 ]]
  then
    echo "$name, your group is youth"
  else
    echo "$name, your group is adult"
  fi
done
