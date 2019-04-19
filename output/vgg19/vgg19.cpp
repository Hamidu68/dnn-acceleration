#include <iostream>
#include <ap_int.h>
#include <hls_stream.h>

typedef int DATA_T;
typedef ap_uint<256> uint256_t;
typedef ap_uint<512> uint512_t;

void Stream_input(DATA_T I[3][224][224], hls::stream<DATA_T> &I_strm) {
  int k, x, y;

#pragma HLS ARRAY_PARTITION variable=I complete dim=1
	Stream_input_x_loop: for (x=0; x<224; x++) {
	 	Stream_input_y_loop: for (y=0; y<224; y++) {
#pragma HLS PIPELINE
	 		Stream_input_m_loop: for (k=0; k<3; k++) {
	 			I_strm.write(I[k][x][y]);
	    }
	  }
	}
}

void Stream_output(hls::stream<DATA_T> &O_strm, DATA_T O[128][56][56]) {
  int m, x, y;
#pragma HLS ARRAY_PARTITION variable=O complete dim=1

	Stream_input_x_loop: for (x=0; x<56; x++) {
	 	Stream_input_y_loop: for (y=0; y<56; y++) {
#pragma HLS PIPELINE
	 		Stream_input_m_loop: for (m=0; m<128; m++) {
	 			O[m][x][y] = O_strm.read();
	    }
	  }
	}
}


static DATA_T I_i[3][224][224];
	static DATA_T W1_i[64][3][3][3];
	static DATA_T B1_i[64];
	static DATA_T W2_i[64][64][3][3];
	static DATA_T B2_i[64];
	static DATA_T W4_i[128][64][3][3];
	static DATA_T B4_i[128];
	static DATA_T W5_i[128][128][3][3];
	static DATA_T B5_i[128];
	

void HW_block1_conv1(hls::stream<DATA_T> &I_strm, DATA_T W[64][3][3][3], DATA_T B[64], hls::stream<DATA_T> &O_strm) {

#pragma HLS INLINE
  int m, x, y, i, j, k;

  uint512_t O;

  DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;

  DATA_T ifm[64], ofm[64];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

  DATA_T I[3][3][224];

#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

 	HW_block1_conv1_x_loop: for (x=0; x<224+2; x++) {
 		HW_block1_conv1_y_loop: for (y=0; y<$224+2; y++) {
  		HW_block1_conv1_k_loop: for (k=0; k<3; k++) {

   	 		if (x<224 && y<224) {
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

  	 			HW_block1_conv1_m_loop: for (m=0; m<64; m++) {

#pragma HLS PIPELINE

  	 				if (k==0) {
  	 					ofm[m] = B[m];
            }
 						if (x-2+0 < 224 && y-2+0 < 224) {
 							ifm[m] = I_0;
            } else {
 							ifm[m] = 0; // zero padding
            }
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][0][0];

 						if (x-2+0 < 224 && y-2+1 < 224) {
 							ifm[m] = I_1;
            } else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][0][1];

 						if (x-2+0 < 224 && y-2+2 < 224) {
 							ifm[m] = I_2;
            } else {
 							ifm[m] = 0; // zero padding
            }
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][0][2];

 						if (x-2+1 < 224 && y-2+0 < 224) {
 							ifm[m] = I_3;
            } else {
 							ifm[m] = 0; // zero padding
            }
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][1][0];

 						if (x-2+1 < 224 && y-2+1 < 224) {
 							ifm[m] = I_4;
            } else {
 							ifm[m] = 0; // zero padding
            }
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][1][1];

 						if (x-2+1 < 224 && y-2+2 < 224) {
 							ifm[m] = I_5;
            } else {
 							ifm[m] = 0; // zero padding
            }
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][1][2];

 						if (x-2+2 < 224 && y-2+0 < 224) {
 							ifm[m] = I_6;
            } else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][2][0];

 						if (x-2+2 < 224 && y-2+1 < 224) {
 							ifm[m] = I_7;
            } else {
 							ifm[m] = 0; // zero padding
            }
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][2][1];

 						if (x-2+2 < 224 && y-2+2 < 224) {
 							ifm[m] = I_8;
            } else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][2][2];

	 				}
  	 			if (k==3-1) {
  	 				for (m=0; m<64; m++) {
#pragma HLS UNROLL
						O_strm.write(ofm[m]);
  	 				}
  	 			}

 	 			}
 			}
   	}
 	}
}

void HW_block1_conv2(hls::stream<DATA_T> &I_strm, DATA_T W[64][64][3][3], DATA_T B[64], hls::stream<DATA_T> &O_strm) {

#pragma HLS INLINE
  int m, x, y, i, j, k;

  uint512_t O;

  DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;

  DATA_T ifm[64], ofm[64];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

  DATA_T I[64][3][224];

#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

 	HW_block1_conv2_x_loop: for (x=0; x<224+2; x++) {
 		HW_block1_conv2_y_loop: for (y=0; y<$224+2; y++) {
  		HW_block1_conv2_k_loop: for (k=0; k<64; k++) {

   	 		if (x<224 && y<224) {
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

  	 			HW_block1_conv2_m_loop: for (m=0; m<64; m++) {

#pragma HLS PIPELINE

  	 				if (k==0) {
  	 					ofm[m] = B[m];
            }
 						if (x-2+0 < 224 && y-2+0 < 224) {
 							ifm[m] = I_0;
            } else {
 							ifm[m] = 0; // zero padding
            }
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][0][0];

 						if (x-2+0 < 224 && y-2+1 < 224) {
 							ifm[m] = I_1;
            } else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][0][1];

 						if (x-2+0 < 224 && y-2+2 < 224) {
 							ifm[m] = I_2;
            } else {
 							ifm[m] = 0; // zero padding
            }
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][0][2];

 						if (x-2+1 < 224 && y-2+0 < 224) {
 							ifm[m] = I_3;
            } else {
 							ifm[m] = 0; // zero padding
            }
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][1][0];

 						if (x-2+1 < 224 && y-2+1 < 224) {
 							ifm[m] = I_4;
            } else {
 							ifm[m] = 0; // zero padding
            }
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][1][1];

 						if (x-2+1 < 224 && y-2+2 < 224) {
 							ifm[m] = I_5;
            } else {
 							ifm[m] = 0; // zero padding
            }
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][1][2];

 						if (x-2+2 < 224 && y-2+0 < 224) {
 							ifm[m] = I_6;
            } else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][2][0];

 						if (x-2+2 < 224 && y-2+1 < 224) {
 							ifm[m] = I_7;
            } else {
 							ifm[m] = 0; // zero padding
            }
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][2][1];

 						if (x-2+2 < 224 && y-2+2 < 224) {
 							ifm[m] = I_8;
            } else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][2][2];

	 				}
  	 			if (k==64-1) {
  	 				for (m=0; m<64; m++) {
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

  DATA_T I[64][3][224];

//#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3
//#pragma HLS ARRAY_PARTITION variable=I cyclic factor=3 dim=3

	HW_block1_pool_x_loop: for (x=0; x<224+2; x++) {
	  HW_block1_pool_y_loop: for (y=0; y<224+2; y++) {

	    HW_block1_pool_m_loop: for (m=0; m<64; m++) {
#pragma HLS PIPELINE
	    	if (x<224 && y<224) {
	    		I[m][x%3][y] = I_strm.read();
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
	    			O_strm.write(max);
	    		}
	    	}
      }
    }
  }


}

void HW_block2_conv1(hls::stream<DATA_T> &I_strm, DATA_T W[128][64][3][3], DATA_T B[128], hls::stream<DATA_T> &O_strm) {

#pragma HLS INLINE
  int m, x, y, i, j, k;

  uint512_t O;

  DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;

  DATA_T ifm[128], ofm[128];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

  DATA_T I[64][3][112];

#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

 	HW_block2_conv1_x_loop: for (x=0; x<112+2; x++) {
 		HW_block2_conv1_y_loop: for (y=0; y<$112+2; y++) {
  		HW_block2_conv1_k_loop: for (k=0; k<64; k++) {

   	 		if (x<112 && y<112) {
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

  	 			HW_block2_conv1_m_loop: for (m=0; m<128; m++) {

#pragma HLS PIPELINE

  	 				if (k==0) {
  	 					ofm[m] = B[m];
            }
 						if (x-2+0 < 112 && y-2+0 < 112) {
 							ifm[m] = I_0;
            } else {
 							ifm[m] = 0; // zero padding
            }
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][0][0];

 						if (x-2+0 < 112 && y-2+1 < 112) {
 							ifm[m] = I_1;
            } else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][0][1];

 						if (x-2+0 < 112 && y-2+2 < 112) {
 							ifm[m] = I_2;
            } else {
 							ifm[m] = 0; // zero padding
            }
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][0][2];

 						if (x-2+1 < 112 && y-2+0 < 112) {
 							ifm[m] = I_3;
            } else {
 							ifm[m] = 0; // zero padding
            }
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][1][0];

 						if (x-2+1 < 112 && y-2+1 < 112) {
 							ifm[m] = I_4;
            } else {
 							ifm[m] = 0; // zero padding
            }
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][1][1];

 						if (x-2+1 < 112 && y-2+2 < 112) {
 							ifm[m] = I_5;
            } else {
 							ifm[m] = 0; // zero padding
            }
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][1][2];

 						if (x-2+2 < 112 && y-2+0 < 112) {
 							ifm[m] = I_6;
            } else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][2][0];

 						if (x-2+2 < 112 && y-2+1 < 112) {
 							ifm[m] = I_7;
            } else {
 							ifm[m] = 0; // zero padding
            }
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][2][1];

 						if (x-2+2 < 112 && y-2+2 < 112) {
 							ifm[m] = I_8;
            } else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][2][2];

	 				}
  	 			if (k==64-1) {
  	 				for (m=0; m<128; m++) {
#pragma HLS UNROLL
						O_strm.write(ofm[m]);
  	 				}
  	 			}

 	 			}
 			}
   	}
 	}
}

void HW_block2_conv2(hls::stream<DATA_T> &I_strm, DATA_T W[128][128][3][3], DATA_T B[128], hls::stream<DATA_T> &O_strm) {

#pragma HLS INLINE
  int m, x, y, i, j, k;

  uint512_t O;

  DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;

  DATA_T ifm[128], ofm[128];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

  DATA_T I[128][3][112];

#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

 	HW_block2_conv2_x_loop: for (x=0; x<112+2; x++) {
 		HW_block2_conv2_y_loop: for (y=0; y<$112+2; y++) {
  		HW_block2_conv2_k_loop: for (k=0; k<128; k++) {

   	 		if (x<112 && y<112) {
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

  	 			HW_block2_conv2_m_loop: for (m=0; m<128; m++) {

#pragma HLS PIPELINE

  	 				if (k==0) {
  	 					ofm[m] = B[m];
            }
 						if (x-2+0 < 112 && y-2+0 < 112) {
 							ifm[m] = I_0;
            } else {
 							ifm[m] = 0; // zero padding
            }
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][0][0];

 						if (x-2+0 < 112 && y-2+1 < 112) {
 							ifm[m] = I_1;
            } else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][0][1];

 						if (x-2+0 < 112 && y-2+2 < 112) {
 							ifm[m] = I_2;
            } else {
 							ifm[m] = 0; // zero padding
            }
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][0][2];

 						if (x-2+1 < 112 && y-2+0 < 112) {
 							ifm[m] = I_3;
            } else {
 							ifm[m] = 0; // zero padding
            }
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][1][0];

 						if (x-2+1 < 112 && y-2+1 < 112) {
 							ifm[m] = I_4;
            } else {
 							ifm[m] = 0; // zero padding
            }
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][1][1];

 						if (x-2+1 < 112 && y-2+2 < 112) {
 							ifm[m] = I_5;
            } else {
 							ifm[m] = 0; // zero padding
            }
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][1][2];

 						if (x-2+2 < 112 && y-2+0 < 112) {
 							ifm[m] = I_6;
            } else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][2][0];

 						if (x-2+2 < 112 && y-2+1 < 112) {
 							ifm[m] = I_7;
            } else {
 							ifm[m] = 0; // zero padding
            }
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][2][1];

 						if (x-2+2 < 112 && y-2+2 < 112) {
 							ifm[m] = I_8;
            } else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][2][2];

	 				}
  	 			if (k==128-1) {
  	 				for (m=0; m<128; m++) {
#pragma HLS UNROLL
						O_strm.write(ofm[m]);
  	 				}
  	 			}

 	 			}
 			}
   	}
 	}
}

void HW_block2_pool(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
  int m, x, y, i, j;
  ap_uint<32> max;

  DATA_T I[128][3][112];

//#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3
//#pragma HLS ARRAY_PARTITION variable=I cyclic factor=3 dim=3

	HW_block2_pool_x_loop: for (x=0; x<112+2; x++) {
	  HW_block2_pool_y_loop: for (y=0; y<112+2; y++) {

	    HW_block2_pool_m_loop: for (m=0; m<128; m++) {
#pragma HLS PIPELINE
	    	if (x<112 && y<112) {
	    		I[m][x%3][y] = I_strm.read();
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
	    			O_strm.write(max);
	    		}
	    	}
      }
    }
  }


}



void SW_block1_conv1(DATA_T I[3][224][224], DATA_T O[64][224][224], DATA_T W[64][3][3][3], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(224 - 1) - 224 + 3)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<224; x++) {
			for (y = 0; y<224; y++) {
				ofm = 0;
				for (k = 0; k<3; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 224 + p && y + j < 224 + p && x + i -p >= 0 && y + j -p >= 0) {
                                    ifm = I[k][x*1 + i - p][y*1 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				ofm += B[m];
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block1_conv2(DATA_T I[64][224][224], DATA_T O[64][224][224], DATA_T W[64][64][3][3], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(224 - 1) - 224 + 3)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<224; x++) {
			for (y = 0; y<224; y++) {
				ofm = 0;
				for (k = 0; k<64; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 224 + p && y + j < 224 + p && x + i -p >= 0 && y + j -p >= 0) {
                                    ifm = I[k][x*1 + i - p][y*1 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				ofm += B[m];
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block1_pool(DATA_T I[64][224][224], DATA_T O[64][112][112])
{
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<64; m++) {
		for (x = 0; x<112; x++) {
			for (y = 0; y<112; y++) {
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
void SW_block2_conv1(DATA_T I[64][112][112], DATA_T O[128][112][112], DATA_T W[128][64][3][3], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(112 - 1) - 112 + 3)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<112; x++) {
			for (y = 0; y<112; y++) {
				ofm = 0;
				for (k = 0; k<64; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 112 + p && y + j < 112 + p && x + i -p >= 0 && y + j -p >= 0) {
                                    ifm = I[k][x*1 + i - p][y*1 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				ofm += B[m];
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block2_conv2(DATA_T I[128][112][112], DATA_T O[128][112][112], DATA_T W[128][128][3][3], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(112 - 1) - 112 + 3)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<112; x++) {
			for (y = 0; y<112; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 112 + p && y + j < 112 + p && x + i -p >= 0 && y + j -p >= 0) {
                                    ifm = I[k][x*1 + i - p][y*1 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				ofm += B[m];
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block2_pool(DATA_T I[128][112][112], DATA_T O[128][56][56])
{
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<128; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
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


void vgg19(DATA_T I_i[3][224][224],DATA_T W1_i[64][3][3][3],DATA_T B1_i[64], DATA_T W2_i[64][64][3][3],DATA_T B2_i[64], DATA_T W4_i[128][64][3][3],DATA_T B4_i[128], DATA_T W5_i[128][128][3][3],DATA_T B5_i[128],  DATA_T O[128][56][56])  {
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
#pragma HLS ARRAY_PARTITION variable=W5_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W5_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W5_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B5_i complete


    #pragma HLS DATAFLOW

    	hls::stream<DATA_T> O0_strm("O0_strm");
	hls::stream<DATA_T> O1_strm("O1_strm");
	hls::stream<DATA_T> O2_strm("O2_strm");
	hls::stream<DATA_T> O3_strm("O3_strm");
	hls::stream<DATA_T> O4_strm("O4_strm");
	hls::stream<DATA_T> O5_strm("O5_strm");
	hls::stream<DATA_T> O6_strm("O6_strm");


    	Stream_input(I_i,O0_strm);
	HW_block1_conv1(O0_strm, W1_i, B1_i,O1_strm);
	HW_block1_conv2(O1_strm, W2_i, B2_i,O2_strm);
	HW_block1_pool(O2_strm ,O3_strm);
	HW_block2_conv1(O3_strm, W4_i, B4_i,O4_strm);
	HW_block2_conv2(O4_strm, W5_i, B5_i,O5_strm);
	HW_block2_pool(O5_strm ,O6_strm);
	Stream_output(O6_strm,O);
	
}

void vgg19_top(DATA_T I[3][224][224],DATA_T W1[64][3][3][3],DATA_T B1[64], DATA_T W2[64][64][3][3],DATA_T B2[64], DATA_T W4[128][64][3][3],DATA_T B4[128], DATA_T W5[128][128][3][3],DATA_T B5[128],  DATA_T O[128][56][56]) {

    DATA_T O_i[128][56][56];
    int m, x, y, i, j, k;

    	I_i_k_loop: for (k=0; k<3; k++) {
  I_i_x_loop: for (x=0; x<224; x++) {
  I_i_y_loop: for (y=0; y<224; y++) {
  I_i[k][x][y] = I[k][x][y];
    }
  }
}
W1_i_m_loop: for (m=0; m<64; m++) {
  W1_i_k_loop: for (k=0; k<3; k++) {
  W1_i_i_loop: for (i=0; i<3; i++) {
  W1_i_j_loop: for (j=0; j<3; j++) {
  W1_i[m][k][i][j] = W1[m][k][i][j];
      }
    }
  }
}
B1_i_m_loop: for (m=0; m<64; m++) {
	B1_i[m] = B1[m];
}
W2_i_m_loop: for (m=0; m<64; m++) {
  W2_i_k_loop: for (k=0; k<64; k++) {
  W2_i_i_loop: for (i=0; i<3; i++) {
  W2_i_j_loop: for (j=0; j<3; j++) {
  W2_i[m][k][i][j] = W2[m][k][i][j];
      }
    }
  }
}
B2_i_m_loop: for (m=0; m<64; m++) {
	B2_i[m] = B2[m];
}
W4_i_m_loop: for (m=0; m<128; m++) {
  W4_i_k_loop: for (k=0; k<64; k++) {
  W4_i_i_loop: for (i=0; i<3; i++) {
  W4_i_j_loop: for (j=0; j<3; j++) {
  W4_i[m][k][i][j] = W4[m][k][i][j];
      }
    }
  }
}
B4_i_m_loop: for (m=0; m<128; m++) {
	B4_i[m] = B4[m];
}
W5_i_m_loop: for (m=0; m<128; m++) {
  W5_i_k_loop: for (k=0; k<128; k++) {
  W5_i_i_loop: for (i=0; i<3; i++) {
  W5_i_j_loop: for (j=0; j<3; j++) {
  W5_i[m][k][i][j] = W5[m][k][i][j];
      }
    }
  }
}
B5_i_m_loop: for (m=0; m<128; m++) {
	B5_i[m] = B5[m];
}


    vgg19(I_i, W1_i, B1_i, W2_i, B2_i, W4_i, B4_i, W5_i, B5_i,  O_i);

    	for(m=0; m<128; m++) {
		for(x=0;x<56;x++){
			for(y=0;y<56;y++){ O[m][x][y] = O_i[m][x][y]; 
}}}

}

void vgg19_sw(DATA_T I[3][224][224],DATA_T W1[64][3][3][3],DATA_T B1[64], DATA_T W2[64][64][3][3],DATA_T B2[64], DATA_T W4[128][64][3][3],DATA_T B4[128], DATA_T W5[128][128][3][3],DATA_T B5[128],  DATA_T O6_SW[128][56][56]) {
    static DATA_T O1_SW[64][224][224];
	static DATA_T O2_SW[64][224][224];
	static DATA_T O3_SW[64][112][112];
	static DATA_T O4_SW[128][112][112];
	static DATA_T O5_SW[128][112][112];
	

    SW_block1_conv1(I,O1_SW,W1,B1);
	SW_block1_conv2(O1_SW,O2_SW,W2,B2);
	SW_block1_pool(O2_SW,O3_SW);
	SW_block2_conv1(O3_SW,O4_SW,W4,B4);
	SW_block2_conv2(O4_SW,O5_SW,W5,B5);
	SW_block2_pool(O5_SW,O6_SW);
	
}