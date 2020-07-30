#!/bin/bash

if [[ $1 -eq 0 ]]
then
    echo "No students"
elif [[ $1 -eq 1 ]]
then
    echo "1 student"
elif [[ $1 -ge 5 ]]
then
    echo "A lot of students"
else
    echo "$1 students"
fi
