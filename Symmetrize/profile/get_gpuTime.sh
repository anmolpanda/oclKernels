#!/bin/bash

awk '/oclTime:/ {print $5}'  $1 

