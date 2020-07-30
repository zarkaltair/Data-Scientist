#!/bin/bash

if [[ -n $0 ]]
then
  echo "True1"
fi

if [[ $# -gt 0 ]]
then
  echo "True2"
fi

if [[ !(4 -le 3) ]]
then
  echo "True3"
fi

if [[ -s $0 ]]
then
  echo "True4"
fi

if [[ -z " " ]]
then
  echo "True5"
fi

if [[ $\var1 == $var2 || $var1 != $var2 ]]
then
  echo "True6"
fi

if [[ -n $1 ]]
then
  echo "True7"
fi

if [[ -z "" ]]
then
  echo "True8"
fi

if [[ 5 -ge 5 ]]
then
  echo "True9"
fi

if [[ $# -ge 0 ]]
then
  echo "True10"
fi

if [[ $\var1 == $var2 && $var1 != $var2 ]]
then
  echo "True11"
fi

if [[ -e $0 ]]
then
  echo "True12"
fi

