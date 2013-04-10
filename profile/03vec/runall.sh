#!/bin/bash

MAXFWIDTH=32

for (( i = 4; i < ${MAXFWIDTH}; i+=4 )); do
	make clean
	make $i
	echo "========FILTER WIDTH $i========" >> output.log
	./vec | tee -a output.log
done
