CFLAG=-g -O2 -std=c++11

INPUT_IMG=trump.ppm
SORT_BY=Luminance
THRESHOLD_MIN=25.0
THRESHOLD_MAX=55.0
REVERSE_SORT_ORDER=0
ANTI_ALIASING=1
SORT_ALPHA=0
OUTPUT_FILE=output.ppm
PATTERN_PARM=Linear 0.7853981624999999

all: aesthetic

aesthetic: PixelSortGPU.cu main.cu
	nvcc ${CFLAG} -arch sm_30 main.cu PixelSortGPU.cu pgm.cpp -o $@

run:
	./aesthetic ${INPUT_IMG} ${SORT_BY} ${THRESHOLD_MIN} ${THRESHOLD_MAX} ${REVERSE_SORT_ORDER} ${ANTI_ALIASING} ${SORT_ALPHA} ${OUTPUT_FILE} ${PATTERN_PARM}

clean:
	rm -r aesthetic
