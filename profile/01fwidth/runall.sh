#!/bin/bash

MAXFWIDTH=30

for (( i = 1; i < ${MAXFWIDTH}; i++ )); do
	make clean
	make $i
	echo "========FILTER WIDTH $i========" >> output.log
	./cpp | tee -a output.log
	./opencl | tee -a output.log
done
