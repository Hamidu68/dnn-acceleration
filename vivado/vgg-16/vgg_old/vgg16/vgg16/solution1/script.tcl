############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2018 Xilinx, Inc. All Rights Reserved.
############################################################
open_project vgg16
set_top VGG16
add_files ../../../../Desktop/vgg16_20180710/vgg16.cpp
add_files -tb ../../../../Desktop/vgg16_20180710/vgg16_test.cpp -cflags "-Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas"
open_solution "solution1"
set_part {xczu9eg-ffvc900-3-e} -tool vivado
create_clock -period 10 -name default
#source "./vgg16/solution1/directives.tcl"
csim_design -clean
csynth_design
cosim_design -trace_level all
export_design -format ip_catalog
