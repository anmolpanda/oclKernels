#!/bin/bash

PROG="symmetrize"

if [ ! -d "./result" ];
then
  mkdir "result"
fi

# read the command options

for (( i = 256; i <= 4096; i = i * 2 ))
do
  echo "Run square matrix [$i][$i]"

  if [ -f ./result/job_$i.txt ];
  then
    rm ./result/job_$i.txt
  fi

  if [ -f ./result/gpuTime_job_$i.txt ];
  then
    rm ./result/gpuTime_job_$i.txt
  fi

	
  for (( j = 1 ;  j <= 20 ; j =  j + 1 ))
  do
   	../$PROG $i  ../kernel.cl >> ./result/job_$i.txt
  done

  ./get_gpuTime.sh ./result/job_$i.txt | cut -f1 | sort -nr | tail -1 >> ./result/gpuTime_job_$i.txt

done

echo "Check the directory ./result"
  
