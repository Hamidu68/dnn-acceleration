############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2018 Xilinx, Inc. All Rights Reserved.
############################################################
open_project stream_sample
set_top sample
add_files ../../../../Desktop/stream_sample/stream_sample.cpp
add_files -tb ../../../../Desktop/stream_sample/stream_sample_test.cpp -cflags "-Wno-unknown-pragmas -Wno-unknown-pragmas -Wno-unknown-pragmas"
open_solution "solution1"
set_part {xczu9eg-ffvc900-3-e} -tool vivado
create_clock -period 10 -name default
#source "./stream_sample/solution1/directives.tcl"
csim_design
csynth_design
cosim_design
export_design -rtl verilog -format ip_catalog
