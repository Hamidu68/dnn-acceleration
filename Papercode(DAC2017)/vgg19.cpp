#include <iostream>
#include <ap_int.h>
#include <hls_stream.h>

typedef int DATA_T;

typedef ap_uint<256> uint256_t;
typedef ap_uint<512> uint512_t;

void Stream_input(DATA_T I[3][16][16], hls::stream<DATA_T> &I_strm) {
  int k, x, y;

#pragma HLS ARRAY_PARTITION variable=I complete dim=1
	Stream_input_x_loop: for (x=0; x<16; x++) {
	 	Stream_input_y_loop: for (y=0; y<16; y++) {
#pragma HLS PIPELINE
	 		Stream_input_m_loop: for (k=0; k<3; k++) {
	 			I_strm.write(I[k][x][y]);
	    }
	  }
	}
}

void Stream_output(hls::stream<DATA_T> &O_strm, DATA_T O[16][8][8]) {
  int m, x, y;
#pragma HLS ARRAY_PARTITION variable=O complete dim=1

	Stream_input_x_loop: for (x=0; x<8; x++) {
	 	Stream_input_y_loop: for (y=0; y<8; y++) {
#pragma HLS PIPELINE
	 		Stream_input_m_loop: for (m=0; m<16; m++) {
	 			O[m][x][y] = O_strm.read();
	    }
	  }
	}
}



static DATA_T I_i[3][16][16];
static DATA_T W1_i[8][3][3][3];
static DATA_T B1_i[8];
static DATA_T W2_i[8][8][3][3];
static DATA_T B2_i[8];
static DATA_T W4_i[16][8][3][3];
static DATA_T B4_i[16];


void SW_block1_conv1(DATA_T I[3][16][16], DATA_T O[8][16][16], DATA_T B[8], DATA_T W[8][3][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;

	for (m = 0; m<8; m++) {
		for (x = 0; x<16; x++) {
			for (y = 0; y<16; y++) {
				ofm = B[m];
				for (k = 0; k<3; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 16 && y + j < 16 ) { 
                                    ifm = I[k][x + i][y + j];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				if (ofm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block1_conv2(DATA_T I[8][16][16], DATA_T O[8][16][16], DATA_T B[8], DATA_T W[8][8][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;

	for (m = 0; m<8; m++) {
		for (x = 0; x<16; x++) {
			for (y = 0; y<16; y++) {
				ofm = B[m];
				for (k = 0; k<8; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 16 && y + j < 16 ) { 
                                    ifm = I[k][x + i][y + j];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				if (ofm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block1_pool(DATA_T I[8][16][16], DATA_T O[8][8][8])
{ 
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<8; m++) {
		for (x = 0; x<8; x++) {
			for (y = 0; y<8; y++) {
				max = I[m][x*2][y*2];
				for (i = 0; i<2; i++) {
					for (j = 0; j<2; j++) {
						if (I[m][x*2 + i][y*2 + j] > max) {
							max = I[m][x*2 + i][y*2 + j];
						}
					}
				}
				O[m][x][y] = max;
			}
		}
	}
}
void SW_block2_conv1(DATA_T I[8][8][8], DATA_T O[16][8][8], DATA_T B[16], DATA_T W[16][8][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;

	for (m = 0; m<16; m++) {
		for (x = 0; x<8; x++) {
			for (y = 0; y<8; y++) {
				ofm = B[m];
				for (k = 0; k<8; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 8 && y + j < 8 ) { 
                                    ifm = I[k][x + i][y + j];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				if (ofm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ofm;
			}
		}
	}
}


void HW_block1_conv1(hls::stream<DATA_T> &I_strm, DATA_T W[8][3][3][3], DATA_T B[8], hls::stream<DATA_T> &O_strm) {

#pragma HLS INLINE
  int m, x, y, i, j, k;

  //uint512_t O;

  DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;

  DATA_T ifm[8], ofm[8];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

  DATA_T I[3][3][16];

#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

 	Conv2D_1_x_loop: for (x=0; x<16+2; x++) {
 		Conv2D_1_y_loop: for (y=0; y<16+2; y++) {
  		Conv2D_1_k_loop: for (k=0; k<3; k++) {

   	 		if (x<16 && y<16) {
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

  	 			Conv2D_1_m_loop: for (m=0; m<8; m++) {
#pragma HLS PIPELINE

  	 				if (k==0) {
  	 					ofm[m] = B[m];
  	 				}
 						if (x-2+0 < 16 && y-2+0 < 16) {
 							ifm[m] = I_0;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][0][0];

 						if (x-2+0 < 16 && y-2+1 < 16) {
 							ifm[m] = I_1;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][0][1];

 						if (x-2+0 < 16 && y-2+2 < 16) {
 							ifm[m] = I_2;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][0][2];

 						if (x-2+1 < 16 && y-2+0 < 16) {
 							ifm[m] = I_3;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][1][0];

 						if (x-2+1 < 16 && y-2+1 < 16) {
 							ifm[m] = I_4;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][1][1];

 						if (x-2+1 < 16 && y-2+2 < 16) {
 							ifm[m] = I_5;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][1][2];

 						if (x-2+2 < 16 && y-2+0 < 16) {
 							ifm[m] = I_6;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][2][0];

 						if (x-2+2 < 16 && y-2+1 < 16) {
 							ifm[m] = I_7;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][2][1];

 						if (x-2+2 < 16 && y-2+2 < 16) {
 							ifm[m] = I_8;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][2][2];

	 				}
  	 			if (k==3-1) {
  	 				for (m=0; m<8; m++) {
#pragma HLS UNROLL
  						if(ofm[m]<0)
							ofm[m]=0; 					
						O_strm.write(ofm[m]);
  	 				}
  	 			}

 	 			}
 			}
   	}
 	}
}

void HW_block1_conv2(hls::stream<DATA_T> &I_strm, DATA_T W[8][8][3][3], DATA_T B[8], hls::stream<DATA_T> &O_strm) {

#pragma HLS INLINE
  int m, x, y, i, j, k;

  //uint512_t O;

  DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;

  DATA_T ifm[8], ofm[8];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

  DATA_T I[8][3][16];

#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

 	Conv2D_2_x_loop: for (x=0; x<16+2; x++) {
 		Conv2D_2_y_loop: for (y=0; y<16+2; y++) {
  		Conv2D_2_k_loop: for (k=0; k<8; k++) {

   	 		if (x<16 && y<16) {
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
 						if (x-2+0 < 16 && y-2+0 < 16) {
 							ifm[m] = I_0;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][0][0];

 						if (x-2+0 < 16 && y-2+1 < 16) {
 							ifm[m] = I_1;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][0][1];

 						if (x-2+0 < 16 && y-2+2 < 16) {
 							ifm[m] = I_2;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][0][2];

 						if (x-2+1 < 16 && y-2+0 < 16) {
 							ifm[m] = I_3;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][1][0];

 						if (x-2+1 < 16 && y-2+1 < 16) {
 							ifm[m] = I_4;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][1][1];

 						if (x-2+1 < 16 && y-2+2 < 16) {
 							ifm[m] = I_5;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][1][2];

 						if (x-2+2 < 16 && y-2+0 < 16) {
 							ifm[m] = I_6;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][2][0];

 						if (x-2+2 < 16 && y-2+1 < 16) {
 							ifm[m] = I_7;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][2][1];

 						if (x-2+2 < 16 && y-2+2 < 16) {
 							ifm[m] = I_8;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][2][2];

	 				}
  	 			if (k==8-1) {
  	 				for (m=0; m<8; m++) {
#pragma HLS UNROLL
  						if(ofm[m]<0)
							ofm[m]=0; 					
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
  DATA_T max;

  DATA_T I[8][3][16];

//#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3
//#pragma HLS ARRAY_PARTITION variable=I cyclic factor=3 dim=3

	Pool_1_x_loop: for (x=0; x<16+2; x++) {
	  Pool_1_y_loop: for (y=0; y<16+2; y++) {

	    Pool_1_m_loop: for (m=0; m<8; m++) {
#pragma HLS PIPELINE
	    	if (x<16 && y<16) {
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

void HW_block2_conv1(hls::stream<DATA_T> &I_strm, DATA_T W[16][8][3][3], DATA_T B[16], hls::stream<DATA_T> &O_strm) {

#pragma HLS INLINE
  int m, x, y, i, j, k;

  //uint512_t O;

  DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;

  DATA_T ifm[16], ofm[16];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

  DATA_T I[8][3][8];

#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

 	Conv2D_3_x_loop: for (x=0; x<8+2; x++) {
 		Conv2D_3_y_loop: for (y=0; y<8+2; y++) {
  		Conv2D_3_k_loop: for (k=0; k<8; k++) {

   	 		if (x<8 && y<8) {
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

  	 			Conv2D_3_m_loop: for (m=0; m<16; m++) {
#pragma HLS PIPELINE

  	 				if (k==0) {
  	 					ofm[m] = B[m];
  	 				}
 						if (x-2+0 < 8 && y-2+0 < 8) {
 							ifm[m] = I_0;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][0][0];

 						if (x-2+0 < 8 && y-2+1 < 8) {
 							ifm[m] = I_1;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][0][1];

 						if (x-2+0 < 8 && y-2+2 < 8) {
 							ifm[m] = I_2;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][0][2];

 						if (x-2+1 < 8 && y-2+0 < 8) {
 							ifm[m] = I_3;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][1][0];

 						if (x-2+1 < 8 && y-2+1 < 8) {
 							ifm[m] = I_4;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][1][1];

 						if (x-2+1 < 8 && y-2+2 < 8) {
 							ifm[m] = I_5;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][1][2];

 						if (x-2+2 < 8 && y-2+0 < 8) {
 							ifm[m] = I_6;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][2][0];

 						if (x-2+2 < 8 && y-2+1 < 8) {
 							ifm[m] = I_7;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][2][1];

 						if (x-2+2 < 8 && y-2+2 < 8) {
 							ifm[m] = I_8;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][2][2];

	 				}
  	 			if (k==8-1) {
  	 				for (m=0; m<16; m++) {
#pragma HLS UNROLL
  						if(ofm[m]<0)
							ofm[m]=0; 					
						O_strm.write(ofm[m]);
  	 				}
  	 			}

 	 			}
 			}
   	}
 	}
}



void vgg19(DATA_T I[3][16][16], DATA_T W1_i[8][3][3][3], DATA_T B1_i[8],DATA_T W2_i[8][8][3][3], DATA_T B2_i[8],DATA_T W4_i[16][8][3][3], DATA_T B4_i[16],DATA_T O[16][8][8]) {

#pragma HLS ARRAY_PARTITION variable=W1_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W1_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W1_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B1_i complete
#pragma HLS ARRAY_PARTITION variable=W2_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W2_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W2_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B2_i complete
#pragma HLS ARRAY_PARTITION variable=W4_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W4_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W4_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B4_i complete


#pragma HLS DATAFLOW
  hls::stream<uint256_t> O1_packed_strm("O1_packed_strm");
  
  hls::stream<DATA_T> I_strm("I_strm");
hls::stream<DATA_T> O1_strm("O1_strm");
hls::stream<DATA_T> O2_strm("O2_strm");
hls::stream<DATA_T> O3_strm("O3_strm");
hls::stream<DATA_T> O4_strm("O4_strm");


  Stream_input(I, I_strm);          
  HW_block1_conv1(I_strm, W1_i, B1_i, O1_strm);
HW_block1_conv2(O1_strm, W2_i, B2_i, O2_strm);
HW_block1_pool(O2_strm, O3_strm);
HW_block2_conv1(O3_strm, W4_i, B4_i, O4_strm);

  Stream_output(O4_strm, O);

}

void vgg19_top(DATA_T I[3][16][16], DATA_T W1[8][3][3][3], DATA_T B1[8],DATA_T W2[8][8][3][3], DATA_T B2[8],DATA_T W4[16][8][3][3], DATA_T B4[16],DATA_T O[16][8][8]) {

  DATA_T O_i[16][8][8];


  int m, x, y, i, j, k;

  hls::stream<DATA_T> I_strm;
I_i_k_loop: for (k=0; k<3; k++) {
  I_i_x_loop: for (x=0; x<16; x++) {
  I_i_y_loop: for (y=0; y<16; y++) {
  I_i[k][x][y] = I[k][x][y];
//I_strm.write(I[k][x][y]);
    }
  }
}
B1_i_m_loop: for (m=0; m<8; m++) {
	B1_i[m] = B1[m];
}
W1_i_m_loop: for (m=0; m<8; m++) {
  W1_i_k_loop: for (k=0; k<3; k++) {
  W1_i_i_loop: for (i=0; i<3; i++) {
  W1_i_j_loop: for (j=0; j<3; j++) {
  W1_i[m][k][i][j] = W1[m][k][i][j];
      }
    }
  }
}
B2_i_m_loop: for (m=0; m<8; m++) {
	B2_i[m] = B2[m];
}
W2_i_m_loop: for (m=0; m<8; m++) {
  W2_i_k_loop: for (k=0; k<8; k++) {
  W2_i_i_loop: for (i=0; i<3; i++) {
  W2_i_j_loop: for (j=0; j<3; j++) {
  W2_i[m][k][i][j] = W2[m][k][i][j];
      }
    }
  }
}
B4_i_m_loop: for (m=0; m<16; m++) {
	B4_i[m] = B4[m];
}
W4_i_m_loop: for (m=0; m<16; m++) {
  W4_i_k_loop: for (k=0; k<8; k++) {
  W4_i_i_loop: for (i=0; i<3; i++) {
  W4_i_j_loop: for (j=0; j<3; j++) {
  W4_i[m][k][i][j] = W4[m][k][i][j];
      }
    }
  }
}


  vgg19(I_i, W1_i, B1_i, W2_i, B2_i, W4_i, B4_i, O_i);

  for (m=0; m<16; m++) {
    for (x=0; x<8; x++) {
      for (y=0; y<8; y++) {
          O[m][x][y] = O_i[m][x][y];
      }
    }
  }

}

void vgg19_sw(DATA_T I[3][16][16],DATA_T W1[8][3][3][3], DATA_T B1[8],DATA_T W2[8][8][3][3], DATA_T B2[8],DATA_T W4[16][8][3][3], DATA_T B4[16],DATA_T O4_SW[16][8][8]) {

  static DATA_T O1_SW[8][16][16];
static DATA_T O2_SW[8][16][16];
static DATA_T O3_SW[8][8][8];


  int m, x, y, i, j, k;

  SW_block1_conv1(I,O1_SW,B1,W1);
SW_block1_conv2(O1_SW,O2_SW,B2,W2);
SW_block1_pool(O2_SW,O3_SW);
SW_block2_conv1(O3_SW,O4_SW,B4,W4);


}


