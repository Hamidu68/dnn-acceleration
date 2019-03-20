#!/bin/bash
#$ -q drgPQ
cd {IN_DIR}/{OUT_DIR}
vivado_hls {IN_DIR}/{OUT_DIR}/script.tcl C
