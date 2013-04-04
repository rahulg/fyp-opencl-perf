#!/bin/bash

MAXFWIDTH=30

COMMON_RES="320x240 720x480 720x576 960x540 1280x720 1920x1080 3840x2160"

for RES in ${COMMON_RES}; do
	make clean
	WIDTH=${RES%%x*} HEIGHT=${RES##*x} make
	echo $RES >> output.log
	./cpp | tee -a output.log
	./opencl | tee -a output.log
done
