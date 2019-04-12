### default setting
set SRC_PATH	{IN_DIR}/src
set Project     ml-acc-vgg
set Solution    solution1
set Device      {xczu9eg-ffvc900-3-e}
set Flow        ""
set Clock       10.0
set DefaultFlag 1

#### main part

# Project settings
open_project $Project

# Add the file for synthesis
add_files $SRC_PATH/vgg19.cpp

# Add testbench files for co-simulation
add_files -tb  $SRC_PATH/vgg19_test.cpp

# Set top module of the design
set_top vgg19_top

# Solution settings
open_solution "$Solution"

# Add library
set_part $Device

# Set the target clock period
create_clock -period $Clock

set_directive_unroll -factor {B1_UF} "HW_block1_conv1/Conv2D_1_m_loop"
set_directive_unroll -factor {B1_UF} "HW_block1_conv2/Conv2D_2_m_loop"
set_directive_unroll -factor {B2_UF} "HW_block2_conv1/Conv2D_3_m_loop"
set_directive_unroll -factor {B2_UF} "HW_block2_conv2/Conv2D_4_m_loop"
set_directive_unroll -factor {B3_UF} "HW_block3_conv1/Conv2D_5_m_loop"

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
#export_design -evaluate verilog -format ipxact


exit
