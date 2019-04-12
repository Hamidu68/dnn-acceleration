#!/bin/bash
#$ -q drgPQ


cd {IN_DIR}/{OUT_DIR}

vivado_hls -f {IN_DIR}/{OUT_DIR}/script.tcl

cp -rf {IN_DIR}/inputs/init_Input.bin {IN_DIR}/{OUT_DIR}/ml-acc-vgg/solution1/csim/build/
cp -rf {IN_DIR}/inputs/init_Weight.bin {IN_DIR}/{OUT_DIR}/ml-acc-vgg/solution1/csim/build/
cp -rf {IN_DIR}/inputs/init_Bias.bin {IN_DIR}/{OUT_DIR}/ml-acc-vgg/solution1/csim/build/


vivado_hls -f {IN_DIR}/{OUT_DIR}/script.tcl
