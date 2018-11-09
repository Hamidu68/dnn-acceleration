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

void Stream_output(hls::stream<DATA_T> &O_strm, DATA_T O[256][56][56]) {
  int m, x, y;
#pragma HLS ARRAY_PARTITION variable=O complete dim=1

	Stream_input_x_loop: for (x=0; x<56; x++) {
	 	Stream_input_y_loop: for (y=0; y<56; y++) {
#pragma HLS PIPELINE
	 		Stream_input_m_loop: for (m=0; m<256; m++) {
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
static DATA_T W7_i[256][128][3][3];
static DATA_T B7_i[256];


void SW_block1_conv1(DATA_T I[3][224][224], DATA_T O[64][224][224], DATA_T B[64], DATA_T W[64][3][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    
	for (m = 0; m<64; m++) {
		for (x = 0; x<224; x++) {
			for (y = 0; y<224; y++) {
				ofm = B[m];
				for (k = 0; k<3; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 224 && y + j < 224) { 
                                    ifm = I[k][x + i][y + j];
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

void SW_block1_conv2(DATA_T I[64][224][224], DATA_T O[64][224][224], DATA_T B[64], DATA_T W[64][64][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    
	for (m = 0; m<64; m++) {
		for (x = 0; x<224; x++) {
			for (y = 0; y<224; y++) {
				ofm = B[m];
				for (k = 0; k<64; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 224 && y + j < 224) { 
                                    ifm = I[k][x + i][y + j];
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
void SW_block2_conv1(DATA_T I[64][112][112], DATA_T O[128][112][112], DATA_T B[128], DATA_T W[128][64][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    
	for (m = 0; m<128; m++) {
		for (x = 0; x<112; x++) {
			for (y = 0; y<112; y++) {
				ofm = B[m];
				for (k = 0; k<64; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 112 && y + j < 112) { 
                                    ifm = I[k][x + i][y + j];
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

void SW_block2_conv2(DATA_T I[128][112][112], DATA_T O[128][112][112], DATA_T B[128], DATA_T W[128][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    
	for (m = 0; m<128; m++) {
		for (x = 0; x<112; x++) {
			for (y = 0; y<112; y++) {
				ofm = B[m];
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 112 && y + j < 112) { 
                                    ifm = I[k][x + i][y + j];
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
void SW_block3_conv1(DATA_T I[128][56][56], DATA_T O[256][56][56], DATA_T B[256], DATA_T W[256][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    
	for (m = 0; m<256; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				ofm = B[m];
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 56 && y + j < 56) { 
                                    ifm = I[k][x + i][y + j];
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


void HW_block1_conv1(hls::stream<DATA_T> &I_strm, DATA_T W[64][3][3][3], DATA_T B[64], hls::stream<DATA_T> &O_strm) {

#pragma HLS INLINE
  int m, x, y, i, j, k;

  uint512_t O;

  DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;

  DATA_T ifm[64], ofm[64];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

  DATA_T I[3][3][224];

#pragma HLS ARRAY_PARTITION variable=I complete dim=2
    
 	Conv2D_1_x_loop: for (x=0; x<224+2; x++) {
 		Conv2D_1_y_loop: for (y=0; y<224+2; y++) {
  		Conv2D_1_k_loop: for (k=0; k<3; k++) {

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

  	 			Conv2D_1_m_loop: for (m=0; m<64; m++) {
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

#pragma HLS ARRAY_PARTITION variable=I complete dim=2

 	Conv2D_2_x_loop: for (x=0; x<224+2; x++) {
 		Conv2D_2_y_loop: for (y=0; y<224+2; y++) {
  		Conv2D_2_k_loop: for (k=0; k<64; k++) {

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

  	 			Conv2D_2_m_loop: for (m=0; m<64; m++) {
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

	Pool_1_x_loop: for (x=0; x<224+2; x++) {
	  Pool_1_y_loop: for (y=0; y<224+2; y++) {

	    Pool_1_m_loop: for (m=0; m<64; m++) {
#pragma HLS PIPELINE
	    	if (x<224 && y<224) {
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

void HW_block2_conv1(hls::stream<DATA_T> &I_strm, DATA_T W[128][64][3][3], DATA_T B[128], hls::stream<DATA_T> &O_strm) {

#pragma HLS INLINE
  int m, x, y, i, j, k;

  uint512_t O;

  DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;

  DATA_T ifm[128], ofm[128];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

  DATA_T I[64][3][112];

#pragma HLS ARRAY_PARTITION variable=I complete dim=2
    
 	Conv2D_3_x_loop: for (x=0; x<112+2; x++) {
 		Conv2D_3_y_loop: for (y=0; y<112+2; y++) {
  		Conv2D_3_k_loop: for (k=0; k<64; k++) {

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

  	 			Conv2D_3_m_loop: for (m=0; m<128; m++) {
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

#pragma HLS ARRAY_PARTITION variable=I complete dim=2

 	Conv2D_4_x_loop: for (x=0; x<112+2; x++) {
 		Conv2D_4_y_loop: for (y=0; y<112+2; y++) {
  		Conv2D_4_k_loop: for (k=0; k<128; k++) {

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

  	 			Conv2D_4_m_loop: for (m=0; m<128; m++) {
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

	Pool_2_x_loop: for (x=0; x<112+2; x++) {
	  Pool_2_y_loop: for (y=0; y<112+2; y++) {

	    Pool_2_m_loop: for (m=0; m<128; m++) {
#pragma HLS PIPELINE
	    	if (x<112 && y<112) {
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

void HW_block3_conv1(hls::stream<DATA_T> &I_strm, DATA_T W[256][128][3][3], DATA_T B[256], hls::stream<DATA_T> &O_strm) {

#pragma HLS INLINE
  int m, x, y, i, j, k;

  uint512_t O;

  DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;

  DATA_T ifm[256], ofm[256];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

  DATA_T I[128][3][56];

#pragma HLS ARRAY_PARTITION variable=I complete dim=2

 	Conv2D_5_x_loop: for (x=0; x<56+2; x++) {
 		Conv2D_5_y_loop: for (y=0; y<56+2; y++) {
  		Conv2D_5_k_loop: for (k=0; k<128; k++) {

   	 		if (x<56 && y<56) {
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

  	 			Conv2D_5_m_loop: for (m=0; m<256; m++) {
#pragma HLS PIPELINE

  	 				if (k==0) {
  	 					ofm[m] = B[m];
  	 				}
 						if (x-2+0 < 56 && y-2+0 < 56) {
 							ifm[m] = I_0;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][0][0];

 						if (x-2+0 < 56 && y-2+1 < 56) {
 							ifm[m] = I_1;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][0][1];

 						if (x-2+0 < 56 && y-2+2 < 56) {
 							ifm[m] = I_2;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][0][2];

 						if (x-2+1 < 56 && y-2+0 < 56) {
 							ifm[m] = I_3;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][1][0];

 						if (x-2+1 < 56 && y-2+1 < 56) {
 							ifm[m] = I_4;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][1][1];

 						if (x-2+1 < 56 && y-2+2 < 56) {
 							ifm[m] = I_5;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][1][2];

 						if (x-2+2 < 56 && y-2+0 < 56) {
 							ifm[m] = I_6;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][2][0];

 						if (x-2+2 < 56 && y-2+1 < 56) {
 							ifm[m] = I_7;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][2][1];

 						if (x-2+2 < 56 && y-2+2 < 56) {
 							ifm[m] = I_8;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][2][2];

	 				}
  	 			if (k==128-1) {
  	 				for (m=0; m<256; m++) {
  	 					O_strm.write(ofm[m]);
  	 				}
  	 			}

 	 			}
 			}
   	}
 	}
}



void vgg19(DATA_T I[3][224][224], DATA_T W1_i[64][3][3][3], DATA_T B1_i[64],DATA_T W2_i[64][64][3][3], DATA_T B2_i[64],DATA_T W4_i[128][64][3][3], DATA_T B4_i[128],DATA_T W5_i[128][128][3][3], DATA_T B5_i[128],DATA_T W7_i[256][128][3][3], DATA_T B7_i[256],DATA_T O[256][56][56]) {

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
#pragma HLS ARRAY_PARTITION variable=W7_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W7_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W7_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B7_i complete


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


  Stream_input(I, I_strm);          
  HW_block1_conv1(I_strm, W1_i, B1_i, O1_strm);
HW_block1_conv2(O1_strm, W2_i, B2_i, O2_strm);
HW_block1_pool(O2_strm, O3_strm);
HW_block2_conv1(O3_strm, W4_i, B4_i, O4_strm);
HW_block2_conv2(O4_strm, W5_i, B5_i, O5_strm);
HW_block2_pool(O5_strm, O6_strm);
HW_block3_conv1(O6_strm, W7_i, B7_i, O7_strm);

  Stream_output(O7_strm, O);

}

void vgg19_top(DATA_T I[3][224][224], DATA_T W1[64][3][3][3], DATA_T B1[64],DATA_T W2[64][64][3][3], DATA_T B2[64],DATA_T W4[128][64][3][3], DATA_T B4[128],DATA_T W5[128][128][3][3], DATA_T B5[128],DATA_T W7[256][128][3][3], DATA_T B7[256],DATA_T O[256][56][56]) {

  DATA_T O_i[256][56][56];


  int m, x, y, i, j, k;

  hls::stream<DATA_T> I_strm;
I_i_k_loop: for (k=0; k<3; k++) {
  I_i_x_loop: for (x=0; x<224; x++) {
  I_i_y_loop: for (y=0; y<224; y++) {
  I_i[k][x][y] = I[k][x][y];
//I_strm.write(I[k][x][y]);
    }
  }
}
B1_i_m_loop: for (m=0; m<64; m++) {
	B1_i[m] = B1[m];
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
B2_i_m_loop: for (m=0; m<64; m++) {
	B2_i[m] = B2[m];
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
B4_i_m_loop: for (m=0; m<128; m++) {
	B4_i[m] = B4[m];
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
B5_i_m_loop: for (m=0; m<128; m++) {
	B5_i[m] = B5[m];
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
B7_i_m_loop: for (m=0; m<256; m++) {
	B7_i[m] = B7[m];
}
W7_i_m_loop: for (m=0; m<256; m++) {
  W7_i_k_loop: for (k=0; k<128; k++) {
  W7_i_i_loop: for (i=0; i<3; i++) {
  W7_i_j_loop: for (j=0; j<3; j++) {
  W7_i[m][k][i][j] = W7[m][k][i][j];
      }
    }
  }
}


  vgg19(I_i, W1_i, B1_i, W2_i, B2_i, W4_i, B4_i, W5_i, B5_i, W7_i, B7_i, O_i);

  for (m=0; m<256; m++) {
    for (x=0; x<56; x++) {
      for (y=0; y<56; y++) {
          O[m][x][y] = O_i[m][x][y];
      }
    }
  }

}

void vgg19_sw(DATA_T I[3][224][224],DATA_T W1[64][3][3][3], DATA_T B1[64],DATA_T W2[64][64][3][3], DATA_T B2[64],DATA_T W4[128][64][3][3], DATA_T B4[128],DATA_T W5[128][128][3][3], DATA_T B5[128],DATA_T W7[256][128][3][3], DATA_T B7[256],DATA_T O7_SW[256][56][56]) {

  static DATA_T O1_SW[64][224][224];
static DATA_T O2_SW[64][224][224];
static DATA_T O3_SW[64][112][112];
static DATA_T O4_SW[128][112][112];
static DATA_T O5_SW[128][112][112];
static DATA_T O6_SW[128][56][56];


  int m, x, y, i, j, k;

  SW_block1_conv1(I,O1_SW,B1,W1);
SW_block1_conv2(O1_SW,O2_SW,B2,W2);
SW_block1_pool(O2_SW,O3_SW);
SW_block2_conv1(O3_SW,O4_SW,B4,W4);
SW_block2_conv2(O4_SW,O5_SW,B5,W5);
SW_block2_pool(O5_SW,O6_SW);
SW_block3_conv1(O6_SW,O7_SW,B7,W7);


}


