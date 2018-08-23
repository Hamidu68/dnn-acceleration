#include <iostream>
#include <ap_int.h>
#include <hls_stream.h>

typedef ap_uint<16> DATA_T;

typedef ap_uint<256> uint256_t;
typedef ap_uint<512> uint512_t;

void Stream_input(DATA_T I[3][10][10], hls::stream<DATA_T> &I_strm) {
  int m, x, y;

#pragma HLS ARRAY_PARTITION variable=I complete dim=1
	Stream_input_x_loop: for (x=0; x<10; x++) {
	 	Stream_input_y_loop: for (y=0; y<10; y++) {
#pragma HLS PIPELINE
	 		Stream_input_m_loop: for (k=0; k<3; k++) {
	 			I_strm.write(I[k][x][y]);
	    }
	  }
	}
}

void Stream_output(hls::stream<DATA_T> &O_strm, DATA_T O[3]) {
  int x;
#pragma HLS ARRAY_PARTITION variable=O complete dim=1

	Stream_input_x_loop: for (x=0; x<3; x++) {
	 
#pragma HLS PIPELINE
	 			O[x] = O_strm.read()
	}
}



static DATA_T I_i[3][10][10];
static DATA_T W1_i[4][3][3][3];
static DATA_T B1_i[4];
static DATA_T W3_i[8][4][3][3];
static DATA_T B3_i[8];
static DATA_T W8_i[8][8];
static DATA_T B8_i[8];
static DATA_T W9_i[3][8];
static DATA_T B9_i[3];


void Ex_model(DATA_T I[3][10][10], DATA_T W1_i[4][3][3][3], DATA_T B1_i[4],DATA_T W3_i[8][4][3][3], DATA_T B3_i[8],DATA_T W8_i[8][8], DATA_T B8_i[8],DATA_T W9_i[3][8], DATA_T B9_i[3],DATA_T O[3]) {

#pragma HLS ARRAY_PARTITION variable=W1_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W1_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W1_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B1_i complete
#pragma HLS ARRAY_PARTITION variable=W3_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W3_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W3_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B3_i complete
#pragma HLS ARRAY_PARTITION variable=W8_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W8_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W8_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B8_i complete
#pragma HLS ARRAY_PARTITION variable=W9_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W9_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W9_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B9_i complete


#pragma HLS DATAFLOW
  hls::stream<uint256_t> O1_packed_strm("O1_packed_strm");
  
  hls::stream<DATA_T> I_strm("I_strm");
hls::stream<DATA_T> O1_strm("O1_strm");
hls::stream<DATA_T> O2_strm("O2_strm");
hls::stream<DATA_T> O3_strm("O3_strm");
hls::stream<DATA_T> O4_strm("O4_strm");
hls::stream<DATA_T> O5_strm("O5_strm");
hls::stream<DATA_T> O6_strm("O6_strm");
hls::stream<DATA_T> O7_strm("O7_strm");
hls::stream<DATA_T> O8_strm("O8_strm");
hls::stream<DATA_T> O9_strm("O9_strm");

  
  Stream_input(I, I_strm);
  HW_block1_conv1(I_strm, W1, B1, O1_strm);
HW_block2_conv1(O2_strm, W3, B3, O3_strm);
HW_block1_pool(O3_strm, O4_strm);
HW_fc2(O7_strm, W8, B8, O8_strm);
HW_fc1000(O8_strm, W9, B9, O9_strm);

  Stream_output(O_strm, O);

}

void top(DATA_T I[3][10][10], DATA_T W1_i[4][3][3][3], DATA_T B1_i[4],DATA_T W3_i[8][4][3][3], DATA_T B3_i[8],DATA_T W8_i[8][8], DATA_T B8_i[8],DATA_T W9_i[3][8], DATA_T B9_i[3],DATA_T O[3]) {

  DATA_T O_i[3];


  int m, x, y, i, j, k;

  hls::stream<DATA_T> I_strm;
I_i_k_loop: for (k=0; k<3; k++) {
  I_i_x_loop: for (x=0; x<10; x++) {
  I_i_y_loop: for (y=0; y<10; y++) {
  I_i[k][x][y] = I[k][x][y];
//I_strm.write(I[k][x][y]);
    }
  }
}
B1_i_m_loop: for (m=0; m<4; m++) {
	B1_i[m] = B1[m];
}
W1_i_m_loop: for (m=0; m<4; m++) {
  W1_i_k_loop: for (k=0; k<3; k++) {
  W1_i_i_loop: for (i=0; i<3; i++) {
  W1_i_j_loop: for (j=0; j<3; j++) {
  W1_i[m][k][i][j] = W1[m][k][i][j];
      }
    }
  }
}
B3_i_m_loop: for (m=0; m<8; m++) {
	B3_i[m] = B3[m];
}
W3_i_m_loop: for (m=0; m<8; m++) {
  W3_i_k_loop: for (k=0; k<4; k++) {
  W3_i_i_loop: for (i=0; i<3; i++) {
  W3_i_j_loop: for (j=0; j<3; j++) {
  W3_i[m][k][i][j] = W3[m][k][i][j];
      }
    }
  }
}
B8_i_m_loop: for (m=0; m<8; m++) {
	B8_i[m] = B8[m];
}
W8_i_m_loop: for (m=0; m<8; m++) {
  W8_i_k_loop: for (k=0; k<8; k++) {
  W8_i[m][k] = W8[m][k];
  }
}
B9_i_m_loop: for (m=0; m<3; m++) {
	B9_i[m] = B9[m];
}
W9_i_m_loop: for (m=0; m<3; m++) {
  W9_i_k_loop: for (k=0; k<8; k++) {
  W9_i[m][k] = W9[m][k];
  }
}

  
  Ex_model(I_i, W1_i, B1_i, W3_i, B3_i, W8_i, B8_i, W9_i, B9_i, O_i);

  for (m=0; m<3; m++) {
          O[m] = O_i[m];
  }

}

void VGG19_sw(DATA_T I[3][10][10],DATA_T W1[4][3][3][3], DATA_T B1[4],DATA_T W3[8][4][3][3], DATA_T B3[8],DATA_T W8[8][8], DATA_T B8[8],DATA_T W9[3][8], DATA_T B9[3],DATA_T O9_SW[3]) {

  static DATA_T O1_SW[4][8][8];
static DATA_T O2_SW[4][8][8];
static DATA_T O3_SW[8][8][8];
static DATA_T O4_SW[8][7][7];
static DATA_T O5_SW[8][7][7];
static DATA_T O6_SW[8][1][1];
static DATA_T O7_SW[8];
static DATA_T O8_SW[8];


  int m, x, y, i, j, k;

  SW_block1_conv1(O0_SW,O1_SW,B1,W1);
SW_bn2a_branch2a(O1_SW,O2_SW);
SW_block2_conv1(O2_SW,O3_SW,B3,W3);
SW_block1_pool(O3_SW,O4_SW);
SW_activation_2(O4_SW,O5_SW);
SW_avg_pool(O5_SW,O6_SW);
SW_flatten_1(O6_SW,O7_SW);
SW_fc2(O7_SW,W8_SW,B8_SW,O8_SW);
SW_fc1000(O8_SW,W9_SW,B9_SW,O9_SW);


}

void SW_block1_conv1(DATA_T I[3][10][10], DATA_T O[4][8][8], DATA_T B[4], DATA_T W[4][3][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<4; m++) {
		for (x = 0; x<8; x++) {
			for (y = 0; y<8; y++) {
				ofm = B[m];
				for (k = 0; k<3; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 10 && y + j < 10) { 
								ifm = I[k][x*1 + i][y*1 + j];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_bn2a_branch2a(DATA_T I[4][8][8], DATA_T O[4][8][8]) {
	int m, x, y;
	DATA_T temp;
	DATA_T mean = 0;
	DATA_T dsd_dev = 1;
	DATA_T size = 4 * 8 * 8;
	DATA_T epsilon = 1e-3;
for (m = 0; m < 4; m++){
		for (x = 0; x < 8; x++){
			for (y = 0; y < 8; y++){
				O[m][x][y] = (I[m][x][y]-mean)/sqrt(dsd_dev+epsilon);
			}
		}
	}
}
void SW_block2_conv1(DATA_T I[4][8][8], DATA_T O[8][8][8], DATA_T B[8], DATA_T W[8][4][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(8 - 1) - 8 + 3)/2; 
	for (m = 0; m<8; m++) {
		for (x = 0; x<8; x++) {
			for (y = 0; y<8; y++) {
				ofm = B[m];
				for (k = 0; k<4; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 8 + p && y + j < 8 + p && x + i -p >= 0 && y + j -p >= 0) { 
                                    ifm = I[k][x*1 + i - p][y*1 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block1_pool(DATA_T I[8][8][8], DATA_T O[8][7][7])
{ 
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<8; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				max = I[m][x*1][y*1];
				for (i = 0; i<2; i++) {
					for (j = 0; j<2; j++) {
						if (I[m][x*1 + i][y*1 + j] > max) {
							max = I[m][x*1 + i][y*1 + j];
						}
					}
				}
				O[m][x][y] = max;
			}
		}
	}
}
void SW_activation_2(DATA_T I[8][7][7], DATA_T O[8][7][7])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<8; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_avg_pool(DATA_T I[8][7][7], DATA_T O[8][1][1])
{ 
	int m, x, y, i, j;
	DATA_T sum;
    DATA_T div = 7*7 ; 
	for (m = 0; m<8; m++) {
		for (x = 0; x<1; x++) {
			for (y = 0; y<1; y++) {
                sum = 0;
				for (i = 0; i<7; i++) {
					for (j = 0; j<7; j++) {
						sum += I[m][x*7 + i][y*7 + j];
					}
				}
				O[m][x][y] = sum/div;
			}
		}
	}
}
void SW_flatten_1(DATA_T I[8][1][1], DATA_T O[8]){
	int i, j, x, y;
	i = 0;
	 for(j=0; j< 8; j++)
		for(x=0; x<1;x++)
			for (y = 0; y < 1; y++) {
				O[i] = I[j][x][y];
				i++;
			}
}
void SW_fc2(DATA_T I[8], DATA_T W[8][8], DATA_T B[8], DATA_T O[8])
{
    //Dense
	int m, c;
	for(m=0; m<8; m++){
        O[m] = B[m];
		for (c = 0; c < 8; c++){
            O[m] += W[m][c] * I[c];
        }
        if (O[m] < 0) //Relu
            O[m] = 0;
     }
}
void SW_fc1000(DATA_T I[8], DATA_T W[3][8], DATA_T B[3] , DATA_T O[3])
{
    //Dense
	int m, c;    
	DATA_T denom = 0;
	for(m=0; m<3; m++){
        O[m] = B[m];
		for (c = 0; c < 8; c++){
			O[m] += W[m][c] * I[c];
         }
        denom += O[m]; //Sum of Output
    }
    //Softmax
    float max = 0;
	for (m = 0; m < 3; m++)
		if(O[m]>max)
          max = O[m];
        
     for (m = 0; m < 3; m++){
		if(O[m] != max)
          O[m] = 0;
        else
          O[m] = 1;}

/*
	for (m = 0; m < 3; m++)
		O[m] = O[m] / denom; 
*/

}

void HW_block2_conv1(hls::stream<DATA_T> &I_strm, DATA_T W[8][4][3][3], DATA_T B[8], hls::stream<DATA_T> &O_strm) {

#pragma HLS INLINE
  int m, x, y, i, j, k;

  uint512_t O;

  DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;

  DATA_T ifm[8], ofm[8];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

  DATA_T I[4][3][4];

#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3 

 	Conv2D_2_x_loop: for (x=0; x<4+2; x++) {
 		Conv2D_2_y_loop: for (y=0; y<4+2; y++) {
  		Conv2D_2_k_loop: for (k=0; k<4; k++) {

   	 		if (x<4 && y<4) {
   	 			I[k][x%3][y] = I_strm.read();
   	 		}
  	 		if (x>=2 && y>=2) {


      		I_0 = I[k][(x-2)%3][(y-2)];
      		I_1 = I[k][(x-2)%3][(y-1)];
      		I_2 = I[k][(x-2)%3][(y)];
      		I_3 = I[k][(x-1)%3][(y-2)];
      		I_4 = I[k][(x-1)%3][(y-1)];
      		I_5 = I[k][(x-1)%3][(y)];
      		I_6 = I[k][(x)%3][(y-2)];
      		I_7 = I[k][(x)%3][(y-1)];
      		I_8 = I[k][(x)%3][(y)];

  	 			Conv2D_2_m_loop: for (m=0; m<8; m++) {
#pragma HLS PIPELINE

  	 				if (k==0) {
  	 					ofm[m] = B[m];
  	 				}
 						if (x-2+0 < 4 && y-2+0 < 4) {
 							ifm[m] = I_0;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][0][0];

 						if (x-2+0 < 4 && y-2+1 < 4) {
 							ifm[m] = I_1;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][0][1];

 						if (x-2+0 < 4 && y-2+2 < 4) {
 							ifm[m] = I_2;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][0][2];

 						if (x-2+1 < 4 && y-2+0 < 4) {
 							ifm[m] = I_3;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][1][0];

 						if (x-2+1 < 4 && y-2+1 < 4) {
 							ifm[m] = I_4;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][1][1];

 						if (x-2+1 < 4 && y-2+2 < 4) {
 							ifm[m] = I_5;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][1][2];

 						if (x-2+2 < 4 && y-2+0 < 4) {
 							ifm[m] = I_6;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][2][0];

 						if (x-2+2 < 4 && y-2+1 < 4) {
 							ifm[m] = I_7;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][2][1];

 						if (x-2+2 < 4 && y-2+2 < 4) {
 							ifm[m] = I_8;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][2][2];

	 				}
  	 			if (k==4-1) {
  	 				for (m=0; m<8; m++) {
#pragma HLS UNROLL
  	 					O_strm.write(ofm[m]);
  	 				}
  	 			}

 	 			}
 			}
   	}
 	}
}
void HW_block1_pool(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
  int m, x, y, i, j;
  ap_uint<32> max;

  DATA_T I[8][3][8];

//#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3
//#pragma HLS ARRAY_PARTITION variable=I cyclic factor=3 dim=3

	Pool_1_x_loop: for (x=0; x<8+2; x++) {
	  Pool_1_y_loop: for (y=0; y<8+2; y++) {

	    Pool_1_m_loop: for (m=0; m<8; m++) {
#pragma HLS PIPELINE
	    	if (x<8 && y<8) {
	    		I[m][x%3][y] = I_strm.read();
	    		//std::cout << "HW: MaxPool: I[" << m << "][" << x << "][" << y << "] = " <<  I[m][x%3][y] << std::endl;
	    	}
	    	if (x>=2 && y>=2) {
	    		if (x%2==0 && y%2==0) {
	    			max = I[m][(x-2)%3][y-2];
	    			if (I[m][(x-2)%3][y-2] > max) {
	    				max = I[m][(x-2)%3][y-2];
	    			}
	    			if (I[m][(x-2)%3][y-1] > max) {
	    				max = I[m][(x-2)%3][y-1];
	    			}
	    			if (I[m][(x-1)%3][y-2] > max) {
	    				max = I[m][(x-1)%3][y-2];
	    			}
	    			if (I[m][(x-1)%3][y-1] > max) {
	    				max = I[m][(x-1)%3][y-1];
	    			}
	    			//O[m][(x-2)/2][(y-2)/2] = max;
	    			O_strm.write(max);
	    		}
	    	}
      }
    }
  }


}
void SW_fc2(DATA_T I[8], DATA_T W[8][8], DATA_T B[8] , DATA_T O[8])
{
    //Dense
	int m, c;    
	DATA_T denom = 0;
	for(m=0; m<8; m++){
        O[m] = B[m];
		for (c = 0; c < 8; c++){
			O[m] += W[m][c] * I[c];
         }
        denom += O[m]; //Sum of Output
    }
    //Softmax
    float max = 0;
	for (m = 0; m < 8; m++)
		if(O[m]>max)
          max = O[m];
        
     for (m = 0; m < 8; m++){
		if(O[m] != max)
          O[m] = 0;
        else
          O[m] = 1;}

/*
	for (m = 0; m < 8; m++)
		O[m] = O[m] / denom; 
*/

}
void SW_fc1000(DATA_T I[8], DATA_T W[3][8], DATA_T B[3] , DATA_T O[3])
{
    //Dense
	int m, c;    
	DATA_T denom = 0;
	for(m=0; m<3; m++){
        O[m] = B[m];
		for (c = 0; c < 8; c++){
			O[m] += W[m][c] * I[c];
         }
        denom += O[m]; //Sum of Output
    }
    //Softmax
    float max = 0;
	for (m = 0; m < 3; m++)
		if(O[m]>max)
          max = O[m];
        
     for (m = 0; m < 3; m++){
		if(O[m] != max)
          O[m] = 0;
        else
          O[m] = 1;}

/*
	for (m = 0; m < 3; m++)
		O[m] = O[m] / denom; 
*/

}

