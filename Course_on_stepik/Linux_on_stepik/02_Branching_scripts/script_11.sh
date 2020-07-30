#!/bin/bash

if [[ $1 -gt 5 ]]
then
  echo "one"
elif [[ $1 -lt 3 ]]
then
  echo "two"
elif [[ $1 -eq 4 ]]
then
  echo "thee"
else
  echo "four"
fi

