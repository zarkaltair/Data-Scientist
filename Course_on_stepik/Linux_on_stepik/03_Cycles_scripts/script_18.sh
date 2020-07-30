#!/bin/bash

while true
do
    echo enter your name:; read name
    [ -z $name ] && break
    
    echo enter your age:; read age
    [ $age -eq 0 ] && break
    
    if (( $age < 17 )); then
        group=child
    elif (( $age > 25 )); then
        group=adult
    else
        group=youth
    fi
    
    echo "$name, your group is $group"
done
echo bye
