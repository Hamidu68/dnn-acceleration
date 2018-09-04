#include <iostream>
#include <ap_int.h>
#include <hls_stream.h>

typedef ap_uint<16> DATA_T;

typedef ap_uint<256> uint256_t;
typedef ap_uint<512> uint512_t;

void Stream_input(DATA_T I[3][32][32], hls::stream<DATA_T> &I_strm) {
  int k, x, y;

#pragma HLS ARRAY_PARTITION variable=I complete dim=1
	Stream_input_x_loop: for (x=0; x<32; x++) {
	 	Stream_input_y_loop: for (y=0; y<32; y++) {
#pragma HLS PIPELINE
	 		Stream_input_m_loop: for (k=0; k<3; k++) {
	 			I_strm.write(I[k][x][y]);
	    }
	  }
	}
}

void Stream_output(hls::stream<DATA_T> &O_strm, DATA_T O[10]) {
  int x;
#pragma HLS ARRAY_PARTITION variable=O complete dim=1

	Stream_input_x_loop: for (x=0; x<10; x++) {
	 
#pragma HLS PIPELINE
	 			O[x] = O_strm.read()
	}
}



static DATA_T I_i[3][32][32];
static DATA_T W1_i[16][3][3][3];
static DATA_T B1_i[16];
static DATA_T W2_i[16][16][3][3];
static DATA_T B2_i[16];
static DATA_T W5_i[10][4096];
static DATA_T B5_i[10];
static DATA_T W6_i[10][10];
static DATA_T B6_i[10];


void SW_block1_conv1(DATA_T I[3][32][32], DATA_T O[16][32][32], DATA_T B[16], DATA_T W[16][3][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(32 - 1) - 32 + 3)/2; 
	for (m = 0; m<16; m++) {
		for (x = 0; x<32; x++) {
			for (y = 0; y<32; y++) {
				ofm = B[m];
				for (k = 0; k<3; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 32 + p && y + j < 32 + p && x + i -p >= 0 && y + j -p >= 0) { 
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

void SW_block1_conv2(DATA_T I[16][32][32], DATA_T O[16][32][32], DATA_T B[16], DATA_T W[16][16][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(32 - 1) - 32 + 3)/2; 
	for (m = 0; m<16; m++) {
		for (x = 0; x<32; x++) {
			for (y = 0; y<32; y++) {
				ofm = B[m];
				for (k = 0; k<16; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 32 + p && y + j < 32 + p && x + i -p >= 0 && y + j -p >= 0) { 
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

void SW_block1_pool(DATA_T I[16][32][32], DATA_T O[16][16][16])
{ 
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<16; m++) {
		for (x = 0; x<16; x++) {
			for (y = 0; y<16; y++) {
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
void SW_flatten(DATA_T I[16][16][16], DATA_T O[4096]){
	int i, j, x, y;
	i = 0;
	 for(x=0; x<16;x++)
		for(y=0; y<16; y++)
			for (j=0; j<16; j++) {
				O[i] = I[j][x][y];
				i++;
			}
}

void SW_fc1(DATA_T I[4096], DATA_T W[10][4096], DATA_T B[10], DATA_T O[10])
{
    //Dense
	int m, c;
	for(m=0; m<10; m++){
        O[m] = B[m];
		for (c = 0; c < 4096; c++){
            O[m] += W[m][c] * I[c];
        }
        if (O[m] < 0) //Relu
            O[m] = 0;
     }
}
void SW_predictions(DATA_T I[10], DATA_T W[10][10], DATA_T B[10] , DATA_T O[10])
{
    //Dense
	int m, c;    
	DATA_T denom = 0;
	for(m=0; m<10; m++){
        O[m] = B[m];
		for (c = 0; c < 10; c++){
			O[m] += W[m][c] * I[c];
         }
        denom += O[m]; //Sum of Output
    }
    //Softmax
    float max = 0;
	for (m = 0; m < 10; m++)
		if(O[m]>max)
          max = O[m];
        
     for (m = 0; m < 10; m++){
		if(O[m] != max)
          O[m] = 0;
        else
          O[m] = 1;}

/*
	for (m = 0; m < 10; m++)
		O[m] = O[m] / denom; 
*/

}

void HW_block1_conv1(hls::stream<DATA_T> &I_strm, DATA_T W[16][3][3][3], DATA_T B[16], hls::stream<DATA_T> &O_strm) {

#pragma HLS INLINE
  int m, x, y, i, j, k;

  uint512_t O;

  DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;

  DATA_T ifm[16], ofm[16];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

  DATA_T I[3][3][32];

#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3 

 	Conv2D_1_x_loop: for (x=0; x<32+2; x++) {
 		Conv2D_1_y_loop: for (y=0; y<32+2; y++) {
  		Conv2D_1_k_loop: for (k=0; k<3; k++) {

   	 		if (x<32 && y<32) {
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

  	 			Conv2D_1_m_loop: for (m=0; m<16; m++) {
#pragma HLS PIPELINE

  	 				if (k==0) {
  	 					ofm[m] = B[m];
  	 				}
 						if (x-2+0 < 32 && y-2+0 < 32) {
 							ifm[m] = I_0;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][0][0];

 						if (x-2+0 < 32 && y-2+1 < 32) {
 							ifm[m] = I_1;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][0][1];

 						if (x-2+0 < 32 && y-2+2 < 32) {
 							ifm[m] = I_2;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][0][2];

 						if (x-2+1 < 32 && y-2+0 < 32) {
 							ifm[m] = I_3;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][1][0];

 						if (x-2+1 < 32 && y-2+1 < 32) {
 							ifm[m] = I_4;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][1][1];

 						if (x-2+1 < 32 && y-2+2 < 32) {
 							ifm[m] = I_5;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][1][2];

 						if (x-2+2 < 32 && y-2+0 < 32) {
 							ifm[m] = I_6;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][2][0];

 						if (x-2+2 < 32 && y-2+1 < 32) {
 							ifm[m] = I_7;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][2][1];

 						if (x-2+2 < 32 && y-2+2 < 32) {
 							ifm[m] = I_8;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][2][2];

	 				}
  	 			if (k==3-1) {
  	 				for (m=0; m<16; m++) {
#pragma HLS UNROLL
  	 					O_strm.write(ofm[m]);
  	 				}
  	 			}

 	 			}
 			}
   	}
 	}
}

void HW_block1_conv2(hls::stream<DATA_T> &I_strm, DATA_T W[16][16][3][3], DATA_T B[16], hls::stream<DATA_T> &O_strm) {

#pragma HLS INLINE
  int m, x, y, i, j, k;

  uint512_t O;

  DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;

  DATA_T ifm[16], ofm[16];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

  DATA_T I[16][3][32];

#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3 

 	Conv2D_2_x_loop: for (x=0; x<32+2; x++) {
 		Conv2D_2_y_loop: for (y=0; y<32+2; y++) {
  		Conv2D_2_k_loop: for (k=0; k<16; k++) {

   	 		if (x<32 && y<32) {
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

  	 			Conv2D_2_m_loop: for (m=0; m<16; m++) {
#pragma HLS PIPELINE

  	 				if (k==0) {
  	 					ofm[m] = B[m];
  	 				}
 						if (x-2+0 < 32 && y-2+0 < 32) {
 							ifm[m] = I_0;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][0][0];

 						if (x-2+0 < 32 && y-2+1 < 32) {
 							ifm[m] = I_1;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][0][1];

 						if (x-2+0 < 32 && y-2+2 < 32) {
 							ifm[m] = I_2;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][0][2];

 						if (x-2+1 < 32 && y-2+0 < 32) {
 							ifm[m] = I_3;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][1][0];

 						if (x-2+1 < 32 && y-2+1 < 32) {
 							ifm[m] = I_4;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][1][1];

 						if (x-2+1 < 32 && y-2+2 < 32) {
 							ifm[m] = I_5;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][1][2];

 						if (x-2+2 < 32 && y-2+0 < 32) {
 							ifm[m] = I_6;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][2][0];

 						if (x-2+2 < 32 && y-2+1 < 32) {
 							ifm[m] = I_7;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][2][1];

 						if (x-2+2 < 32 && y-2+2 < 32) {
 							ifm[m] = I_8;
 						} else {
 							ifm[m] = 0; // zero padding
 						}
 						ofm[m] = ofm[m] + ifm[m] * W[m][k][2][2];

	 				}
  	 			if (k==16-1) {
  	 				for (m=0; m<16; m++) {
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

  DATA_T I[16][3][32];

//#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3
//#pragma HLS ARRAY_PARTITION variable=I cyclic factor=3 dim=3

	Pool_1_x_loop: for (x=0; x<16+2; x++) {
	  Pool_1_y_loop: for (y=0; y<32+2; y++) {

	    Pool_1_m_loop: for (m=0; m<16; m++) {
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

void SW_fc1(DATA_T I[4096], DATA_T W[10][4096], DATA_T B[10] , DATA_T O[10])
{
    //Dense
	int m, c;    
	DATA_T denom = 0;
	for(m=0; m<10; m++){
        O[m] = B[m];
		for (c = 0; c < 4096; c++){
			O[m] += W[m][c] * I[c];
         }
        denom += O[m]; //Sum of Output
    }
    //Softmax
    float max = 0;
	for (m = 0; m < 10; m++)
		if(O[m]>max)
          max = O[m];
        
     for (m = 0; m < 10; m++){
		if(O[m] != max)
          O[m] = 0;
        else
          O[m] = 1;}

/*
	for (m = 0; m < 10; m++)
		O[m] = O[m] / denom; 
*/

}
void SW_predictions(DATA_T I[10], DATA_T W[10][10], DATA_T B[10] , DATA_T O[10])
{
    //Dense
	int m, c;    
	DATA_T denom = 0;
	for(m=0; m<10; m++){
        O[m] = B[m];
		for (c = 0; c < 10; c++){
			O[m] += W[m][c] * I[c];
         }
        denom += O[m]; //Sum of Output
    }
    //Softmax
    float max = 0;
	for (m = 0; m < 10; m++)
		if(O[m]>max)
          max = O[m];
        
     for (m = 0; m < 10; m++){
		if(O[m] != max)
          O[m] = 0;
        else
          O[m] = 1;}

/*
	for (m = 0; m < 10; m++)
		O[m] = O[m] / denom; 
*/

}


void vgg19(DATA_T I[3][32][32], DATA_T W1_i[16][3][3][3], DATA_T B1_i[16],DATA_T W2_i[16][16][3][3], DATA_T B2_i[16],DATA_T W5_i[10][4096], DATA_T B5_i[10],DATA_T W6_i[10][10], DATA_T B6_i[10],DATA_T O[10]) {

#pragma HLS ARRAY_PARTITION variable=W1_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W1_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W1_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B1_i complete
#pragma HLS ARRAY_PARTITION variable=W2_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W2_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W2_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B2_i complete
#pragma HLS ARRAY_PARTITION variable=W5_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W5_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W5_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B5_i complete
#pragma HLS ARRAY_PARTITION variable=W6_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W6_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W6_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B6_i complete


#pragma HLS DATAFLOW
  hls::stream<uint256_t> O1_packed_strm("O1_packed_strm");
  
  hls::stream<DATA_T> I_strm("I_strm");
hls::stream<DATA_T> O1_strm("O1_strm");
hls::stream<DATA_T> O2_strm("O2_strm");
hls::stream<DATA_T> O3_strm("O3_strm");
hls::stream<DATA_T> O4_strm("O4_strm");
hls::stream<DATA_T> O5_strm("O5_strm");
hls::stream<DATA_T> O6_strm("O6_strm");

  
  Stream_input(I, I_strm);
  HW_block1_conv1(I_strm, W1_i, B1_i, O1_strm);
HW_block1_conv2(O1_strm, W2_i, B2_i, O2_strm);
HW_block1_pool(O2_strm, O3_strm);
HW_fc1(O4_strm, W5_i, B5_i, O5_strm);
HW_predictions(O5_strm, W6_i, B6_i, O6_strm);

  Stream_output(O6_strm, O);

}

void vgg19_top(DATA_T I[3][32][32], DATA_T W1[16][3][3][3], DATA_T B1[16],DATA_T W2[16][16][3][3], DATA_T B2[16],DATA_T W5[10][4096], DATA_T B5[10],DATA_T W6[10][10], DATA_T B6[10],DATA_T O[10]) {

  DATA_T O_i[10];


  int m, x, y, i, j, k;

  hls::stream<DATA_T> I_strm;
I_i_k_loop: for (k=0; k<3; k++) {
  I_i_x_loop: for (x=0; x<32; x++) {
  I_i_y_loop: for (y=0; y<32; y++) {
  I_i[k][x][y] = I[k][x][y];
//I_strm.write(I[k][x][y]);
    }
  }
}
B1_i_m_loop: for (m=0; m<16; m++) {
	B1_i[m] = B1[m];
}
W1_i_m_loop: for (m=0; m<16; m++) {
  W1_i_k_loop: for (k=0; k<3; k++) {
  W1_i_i_loop: for (i=0; i<3; i++) {
  W1_i_j_loop: for (j=0; j<3; j++) {
  W1_i[m][k][i][j] = W1[m][k][i][j];
      }
    }
  }
}
B2_i_m_loop: for (m=0; m<16; m++) {
	B2_i[m] = B2[m];
}
W2_i_m_loop: for (m=0; m<16; m++) {
  W2_i_k_loop: for (k=0; k<16; k++) {
  W2_i_i_loop: for (i=0; i<3; i++) {
  W2_i_j_loop: for (j=0; j<3; j++) {
  W2_i[m][k][i][j] = W2[m][k][i][j];
      }
    }
  }
}
B5_i_m_loop: for (m=0; m<10; m++) {
	B5_i[m] = B5[m];
}
W5_i_m_loop: for (m=0; m<10; m++) {
  W5_i_k_loop: for (k=0; k<4096; k++) {
  W5_i[m][k] = W5[m][k];
  }
}
B6_i_m_loop: for (m=0; m<10; m++) {
	B6_i[m] = B6[m];
}
W6_i_m_loop: for (m=0; m<10; m++) {
  W6_i_k_loop: for (k=0; k<10; k++) {
  W6_i[m][k] = W6[m][k];
  }
}

  
  vgg19(I_i, W1_i, B1_i, W2_i, B2_i, W5_i, B5_i, W6_i, B6_i, O_i);

  for (m=0; m<10; m++) {
          O[m] = O_i[m];
  }

}

void vgg19_sw(DATA_T I[3][32][32],DATA_T W1[16][3][3][3], DATA_T B1[16],DATA_T W2[16][16][3][3], DATA_T B2[16],DATA_T W5[10][4096], DATA_T B5[10],DATA_T W6[10][10], DATA_T B6[10],DATA_T O6_SW[10]) {

  static DATA_T O1_SW[16][32][32];
static DATA_T O2_SW[16][32][32];
static DATA_T O3_SW[16][16][16];
static DATA_T O4_SW[4096];
static DATA_T O5_SW[10];


  int m, x, y, i, j, k;

  SW_block1_conv1(I,O1_SW,B1,W1);
SW_block1_conv2(O1_SW,O2_SW,B2,W2);
SW_block1_pool(O2_SW,O3_SW);
SW_flatten(O3_SW,O4_SW);
SW_fc1(O4_SW,W5_SW,B5_SW,O5_SW);
SW_predictions(O5_SW,W6_SW,B6_SW,O6_SW);


}


