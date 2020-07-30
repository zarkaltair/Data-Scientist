#!/bin/bash

if [[ $1 == 0 ]]
then
  echo "No students"
elif [[ $1 == 1 ]]
then
  echo "1 student"
elif [[ $1 == 2 ]]
then
  echo "2 students"
elif [[ $1 == 3 ]]
then
  echo "3 students"
elif [[ $1 == 4 ]]
then
  echo "4 students"
else
  echo "A lot of students"
fi
