### default setting
set SRC_PATH	{IN_DIR}/src
set Project     vgg
set Solution    sol
set Device      "xa7a12tcsg325-1q"
set Flow        ""
set Clock       10.0
set DefaultFlag 1

#### main part

# Project settings
open_project $Project -reset

# Add the file for synthesis
add_files $SRC_PATH/vgg16.cpp

# Add testbench files for co-simulation
add_files -tb  $SRC_PATH/vgg16_test.cpp

# Set top module of the design
set_top vgg19_top

# Solution settings
open_solution -reset $Solution

# Add library
set_part $Device

# Set the target clock period
create_clock -period $Clock

#################
# C SIMULATION
#################
csim_design

#############
# SYNTHESIS #
#############
csynth_design

#################
# CO-SIMULATION #
#################
cosim_design -rtl verilog -trace_level all

##################
# IMPLEMENTATION #
##################
export_design -evaluate verilog -format ipxact


exit
