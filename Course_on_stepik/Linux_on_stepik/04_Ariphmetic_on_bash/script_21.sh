#!/bin/bash

a=10
b=5

let a=$\a+$b
let "a+=b"
let a=a+b
let "a = a + b"
let "a=$\a+$b"

