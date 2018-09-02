#include <iostream>
#include <ap_int.h>
#include <hls_stream.h>


//using namespace std;
//typedef int DATA_T;
typedef ap_uint<32> DATA_T;

typedef ap_uint<256> uint256_t;


void Stream_input(DATA_T I[3][16][16], hls::stream<DATA_T> &I_strm) {
  int m, x, y, i, j, k;
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
  int m, x, y, i, j, k;
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

void Conv2D_padding_act_relu_1(hls::stream<DATA_T> &I_strm, DATA_T W[8][3][3][3], DATA_T B[8], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
  int m, x, y, i, j, k;
  DATA_T ifm, ofm;
  DATA_T I[3][3][16];
  //DATA_T I[3][16][16];

#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

  int Tm, Tk;

 	Conv2D_1_x_loop: for (x=0; x<18; x++) {
 		Conv2D_1_y_loop: for (y=0; y<18; y++) {
  	  //Conv2D_1_m_tile_loop: for (Tm=0; Tm<8; Tm+=8) {
   		  Conv2D_1_m_loop: for (m=0; m<8; m++) {

   	 			//Conv2D_1_k_tile_loop: for (Tk=0; Tk<3; Tk+=3) {
   	 				Conv2D_1_k_loop: for (k=0; k<3; k++) {
#pragma HLS PIPELINE
   	 					if (m==0) {
   	 						if (x<16 && y<16) {
   	 							I[k][x%3][y] = I_strm.read();
   	 						}
   	 					}
   	 					if (x>=2 && y>=2) {
   	 						if (k==0) {
   	 							ofm = B[m];
   	 						}
   	 						Conv2D_1_i_loop: for (i=0; i<3; i++) {
   	 							Conv2D_1_j_loop: for (j=0; j<3; j++) {
   	 								if (x-2+i < 16 && y-2+j < 16) {
   	 									ifm = I[k][(x-2+i)%3][y-2+j];
   	 								} else {
   	 									ifm = 0; // zero padding
   	 								}
   	 								ofm = ofm + ifm * W[m][k][i][j];
   	 							}
   	 						}
   	 						if (k == 2) {
   	 							//O[Tm+m][x-2][y-2] = ofm;
   	 							O_strm.write(ofm);
   	 						}
   	 					}
   	 		//		}
   	 		//	}
   		  }
  	  }
    }
  }

}


/*
void act_relu_1(DATA_T I[8][16][16], DATA_T O[8][16][16]) {
  relu_1_x_loop: for (x=0; x<16; x++) {
  	relu_1_y_loop: for (y=0; y<16; y++) {
      relu_1_k_loop: for (k=0; k<8; k++) {


      }
  	}
  }
}
*/

void Conv2D_padding_act_relu_2(hls::stream<DATA_T> &I_strm, DATA_T W[8][8][3][3], DATA_T B[8], hls::stream<uint256_t> &O_strm) {
//void Conv2D_padding_act_relu_2(hls::stream<DATA_T> &I_strm, DATA_T W[8][8][3][3], DATA_T B[8], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
  int m, x, y, i, j, k;
  //DATA_T ifm, ofm;

  uint256_t O;

  DATA_T ifm[8], ofm[8];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

  DATA_T O_buf[8];
#pragma HLS ARRAY_PARTITION variable=O_buf complete dim=0

  //DATA_T I[8][16][16];
  DATA_T I[8][3][16];

/* without the following pragma, cosim dump fatal error ...*/
//#pragma HLS ARRAY_PARTITION variable=I cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3
//#pragma HLS ARRAY_PARTITION variable=I cyclic factor=3 dim=3 // attaching this pragma increase II

  DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
  int Tm, Tk;

 	Conv2D_2_x_loop: for (x=0; x<18; x++) {
 		Conv2D_2_y_loop: for (y=0; y<18; y++) {
 			Conv2D_2_k_loop: for (k=0; k<8; k++) {
#pragma HLS PIPELINE
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

      			if (k==0) {
      				ofm[m] = B[m];
      			}

      			if (x-2 < 16 && y-2 < 16) {
      				ifm[m] = I_0;
      			} else {
      				ifm[m] = 0; // zero padding
      			}
      			ofm[m] = ofm[m] + ifm[m] * W[m][k][0][0];

      			if (x-2 < 16 && y-1 < 16) {
      				ifm[m] = I_1;
      			} else {
      				ifm[m] = 0; // zero padding
      			}
      			ofm[m] = ofm[m] + ifm[m] * W[m][k][0][1];

      			if (x-2 < 16 && y < 16) {
      				ifm[m] = I_2;
      			} else {
      				ifm[m] = 0; // zero padding
      			}
      			ofm[m] = ofm[m] + ifm[m] * W[m][k][0][2];

      			if (x-1 < 16 && y-2 < 16) {
      				ifm[m] = I_3;
      			} else {
      				ifm[m] = 0; // zero padding
      			}
      			ofm[m] = ofm[m] + ifm[m] * W[m][k][1][0];

      			if (x-1 < 16 && y-1 < 16) {
      				ifm[m] = I_4;
      			} else {
      				ifm[m] = 0; // zero padding
      			}
      			ofm[m] = ofm[m] + ifm[m] * W[m][k][1][1];

      			if (x-1 < 16 && y < 16) {
      				ifm[m] = I_5;
      			} else {
      				ifm[m] = 0; // zero padding
      			}
      			ofm[m] = ofm[m] + ifm[m] * W[m][k][1][2];

      			if (x < 16 && y-2 < 16) {
      				ifm[m] = I_6;
      			} else {
      				ifm[m] = 0; // zero padding
      			}
      			ofm[m] = ofm[m] + ifm[m] * W[m][k][2][0];

      			if (x < 16 && y-1 < 16) {
      				ifm[m] = I_7;
      			} else {
      				ifm[m] = 0; // zero padding
      			}
      			ofm[m] = ofm[m] + ifm[m] * W[m][k][2][1];

      			if (x < 16 && y < 16) {
      				ifm[m] = I_8;
      			} else {
      				ifm[m] = 0; // zero padding
      			}
      			ofm[m] = ofm[m] + ifm[m] * W[m][k][2][2];

          	/*if (k==7) {
        			//O[m][x-2][y-2] = ofm;
          		O_strm.write(ofm[m]);
          	}*/

    	    }

      		if (k==7) {
      			O = (uint256_t)ofm[0];
      			O = O | (uint256_t)ofm[1] << 32;
      			O = O | (uint256_t)ofm[2] << 64;
      			O = O | (uint256_t)ofm[3] << 96;
      			O = O | (uint256_t)ofm[4] << 128;
      			O = O | (uint256_t)ofm[5] << 160;
      			O = O | (uint256_t)ofm[6] << 192;
      			O = O | (uint256_t)ofm[7] << 224;
      			O_strm.write(O);
      		}

      	}
 			}
 		}
 	}
}


void Unpack(hls::stream<uint256_t> &I_strm, hls::stream<DATA_T> &O_strm) {
	int  x, y;
  uint256_t I;
  DATA_T O;

 	Unpack_x_loop: for (x=0; x<16; x++) {
 		Unpack_y_loop: for (y=0; y<16; y++) {
 				I = I_strm.read();

 				O = DATA_T(I);
 				O_strm.write(O);
 				O = DATA_T(I >> 32);
 				O_strm.write(O);
 				O = DATA_T(I >> 64);
 				O_strm.write(O);
 				O = DATA_T(I >> 96);
 				O_strm.write(O);
 				O = DATA_T(I >> 128);
 				O_strm.write(O);
 				O = DATA_T(I >> 160);
 				O_strm.write(O);
 				O = DATA_T(I >> 192);
 				O_strm.write(O);
 				O = DATA_T(I >> 224);
 				O_strm.write(O);

 		}
 	}

}


void MaxPooling2D_1(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
//void MaxPooling2D_1(hls::stream<DATA_T> &I_strm, DATA_T O[8][8][8]) {
#pragma HLS INLINE
  int m, x, y, i, j;
  ap_uint<32> max;

  //DATA_T I[8][16][16];
  DATA_T I[8][3][16];

//#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3
//#pragma HLS ARRAY_PARTITION variable=I cyclic factor=3 dim=3




	Pool_1_x_loop: for (x=0; x<18; x++) {
	  Pool_1_y_loop: for (y=0; y<18; y++) {



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

void Conv2D_padding_act_relu_3(hls::stream<DATA_T> &I_strm, DATA_T W[16][8][3][3], DATA_T B[16], hls::stream<DATA_T> &O_strm) {
//void Conv2D_padding_act_relu_3(hls::stream<DATA_T> &I_strm, DATA_T W[16][8][3][3], DATA_T B[16], DATA_T O[16][8][8]) {
//void Conv2D_padding_act_relu_3(DATA_T I[8][8][8], DATA_T W[16][8][3][3], DATA_T B[16], DATA_T O[16][8][8]) {
#pragma HLS INLINE
  int m, x, y, i, j, k;
  DATA_T ifm, ofm;

  DATA_T I[8][3][8];


//#pragma HLS ARRAY_PARTITION variable=I cyclic factor=4 dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3
  //#pragma HLS ARRAY_PARTITION variable=I cyclic factor=3 dim=3


  int Tm, Tk;

 	Conv2D_3_x_loop: for (x=0; x<10; x++) {
 		Conv2D_3_y_loop: for (y=0; y<10; y++) {
  	  Conv2D_3_m_tile_loop: for (Tm=0; Tm<16; Tm+=8) {
   		  Conv2D_3_m_loop: for (m=0; m<8; m++) {
   	 			Conv2D_3_k_tile_loop: for (Tk=0; Tk<8; Tk+=4) {
#pragma HLS PIPELINE
   	 				Conv2D_3_k_loop: for (k=0; k<4; k++) {

   	 					if (Tm+m==0) {
   	 						if (x<8 && y<8) {
   	 							I[Tk+k][x%3][y] = I_strm.read();
   	 						}
   	 					}

   	 					if (x>=2 && y>=2) {
   	 						if (Tk+k==0) {
   	 							ofm = B[Tm+m];
   	 						}
   	 						Conv2D_3_i_loop: for (i=0; i<3; i++) {
   	 							Conv2D_3_j_loop: for (j=0; j<3; j++) {
   	 								if (x-2+i < 8 && y-2+j < 8) {
   	 									ifm = I[Tk+k][(x-2+i)%3][y-2+j];
   	 								} else {
   	 									ifm = 0; // zero padding
   	 								}
   	 								ofm = ofm + ifm * W[Tm+m][Tk+k][i][j];
   	 							}
   	 						}
   	 						/*if (Tk+k == 7) {
   	 							//O[Tm+m][x-2][y-2] = ofm;
   	 							O_strm.write(ofm);
   	 						}*/
   	 					}
   	 				}
   	 			}
   	 			if (x>=2 && y>=2) {
   	 				O_strm.write(ofm);
   	 			}
   		  }
  	  }
    }
  }


}


void Conv2D_padding_act_relu_1_sw(DATA_T I[3][16][16], DATA_T W[8][3][3][3], DATA_T B[8], DATA_T O[8][16][16], int M, int C, int R, int S, int E, int F, int U) {
  int m, x, y, i, j, k;
  DATA_T ifm, ofm;

  for (m=0; m<8; m++) {
    for (x=0; x<16; x++) {
      for (y=0; y<16; y++) {
        ofm = B[m];
        for (k=0; k<3; k++) {
          for (i=0; i<3; i++) {
            for (j=0; j<3; j++) {
              if (x+i < 16 && y+j < 16) {
                ifm = I[k][x+i][y+j];
              } else {
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


void Conv2D_padding_act_relu_2_sw(DATA_T I[8][16][16], DATA_T W[8][8][3][3], DATA_T B[8], DATA_T O[8][16][16], int M, int C, int R, int S, int E, int F, int U) {
  int m, x, y, i, j, k;

  DATA_T ifm, ofm;

  for (m=0; m<8; m++) {
    for (x=0; x<16; x++) {
      for (y=0; y<16; y++) {
        ofm = B[m];
        for (k=0; k<8; k++) {
          for (i=0; i<3; i++) {
            for (j=0; j<3; j++) {
              if (x+i < 16 && y+j < 16) {
                ifm = I[k][x+i][y+j];
              } else {
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


void MaxPooling2D_1_sw(DATA_T I[8][16][16], DATA_T O[8][8][8], int M, int R, int S, int F, int E, int U) {

  int m, x, y, i, j;
  ap_uint<32> max;


    for (x=0; x<F; x++) {
      for (y=0; y<E; y++) {
        for (m=0; m<M; m++) {

    		//std::cout << "SW: MaxPool: I[" << m << "][" << x << "][" << y << "] = " <<  I[m][x][y] << std::endl;

        max = I[m][U*x][U*y];
        for (i=0; i<R; i++) {
          for (j=0; j<S; j++) {
            if (I[m][U*x+i][U*y+j] > max) {
              max = I[m][U*x+i][U*y+j];
            }
          }
        }
        O[m][x][y] = max;

      }
    }
  }
}



void Conv2D_padding_act_relu_3_sw(DATA_T I[8][8][8], DATA_T W[16][8][3][3], DATA_T B[16], DATA_T O[8][8][8], int M, int C, int R, int S, int E, int F, int U) {
  int m, x, y, i, j, k;
  DATA_T ifm, ofm;

  for (m=0; m<16; m++) {
    for (x=0; x<8; x++) {
      for (y=0; y<8; y++) {
        ofm = B[m];
        for (k=0; k<8; k++) {
          for (i=0; i<3; i++) {
            for (j=0; j<3; j++) {
              if (x+i < 8 && y+j < 8) {
                ifm = I[k][x+i][y+j];
              } else {
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

static DATA_T I_i[3][16][16];
static DATA_T W1_i[8][3][3][3];
static DATA_T W2_i[8][8][3][3];
static DATA_T W3_i[16][8][3][3];
static DATA_T B1_i[8];
static DATA_T B2_i[8];
static DATA_T B3_i[16];



void VGG16(DATA_T I[3][16][16], DATA_T W1_i[8][3][3][3], DATA_T W2_i[8][8][3][3], DATA_T W3_i[16][8][3][3], DATA_T B1_i[8], DATA_T B2_i[8], DATA_T B3_i[16], DATA_T O[16][8][8]) {
//void VGG16(hls::stream<DATA_T> &I_strm /*DATA_T I[3][16][16]*/, DATA_T W1_i[8][3][3][3], DATA_T W2_i[8][8][3][3], DATA_T W3_i[16][8][3][3], DATA_T B1_i[8], DATA_T B2_i[8], DATA_T B3_i[16], DATA_T O[16][8][8]) {

#pragma HLS ARRAY_PARTITION variable=W1_i complete dim=2
#pragma HLS ARRAY_PARTITION variable=W1_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W1_i complete dim=4

#pragma HLS ARRAY_PARTITION variable=W2_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W2_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W2_i complete dim=4

#pragma HLS ARRAY_PARTITION variable=B2_i complete

#pragma HLS ARRAY_PARTITION variable=W3_i cyclic factor=4 dim=2
#pragma HLS ARRAY_PARTITION variable=W3_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W3_i complete dim=4

#pragma HLS ARRAY_PARTITION variable=B3_i cyclic factor=4 dim=1


#pragma HLS DATAFLOW

  hls::stream<DATA_T> O1_strm("O1_strm");
  hls::stream<uint256_t> O2_packed_strm("O2_packed_strm");
  hls::stream<DATA_T> O2_strm("O2_strm");
  hls::stream<DATA_T> O3_strm("O3_strm");
  hls::stream<DATA_T> I_strm("I_strm");
  hls::stream<DATA_T> O_strm("O_strm");
//#pragma HLS stream variable=O1_strm depth=2


  Stream_input(I, I_strm);
  Conv2D_padding_act_relu_1(I_strm, W1_i, B1_i, O1_strm);
  //Conv2D_padding_act_relu_2(O1_strm, W2_i, B2_i, O2_strm);
  Conv2D_padding_act_relu_2(O1_strm, W2_i, B2_i, O2_packed_strm);
  Unpack(O2_packed_strm, O2_strm);
  MaxPooling2D_1(O2_strm, O3_strm);
  Conv2D_padding_act_relu_3(O3_strm, W3_i, B3_i, O_strm);
  Stream_output(O_strm, O);

}


void VGG16_top(DATA_T I[3][16][16], DATA_T W1[8][3][3][3], DATA_T W2[8][8][3][3], DATA_T W3[16][8][3][3], DATA_T B1[8], DATA_T B2[8], DATA_T B3[16], DATA_T O[16][8][8]) {

  DATA_T O_i[16][8][8];


  int m, x, y, i, j, k;

  B1_i_m_loop: for (m=0; m<8; m++) {
    B1_i[m] = B1[m];
  }
  B2_i_m_loop: for (m=0; m<8; m++) {
    B2_i[m] = B2[m];
  }
  B3_i_m_loop: for (m=0; m<16; m++) {
    B3_i[m] = B3[m];
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
  W2_i_m_loop: for (m=0; m<8; m++) {
    W2_i_k_loop: for (k=0; k<8; k++) {
	  W2_i_i_loop: for (i=0; i<3; i++) {
	    W2_i_j_loop: for (j=0; j<3; j++) {
          W2_i[m][k][i][j] = W2[m][k][i][j];
        }
      }
    }
  }
  W3_i_m_loop: for (m=0; m<16; m++) {
	W3_i_k_loop: for (k=0; k<8; k++) {
	  W3_i_i_loop: for (i=0; i<3; i++) {
	    W3_i_j_loop: for (j=0; j<3; j++) {
          W3_i[m][k][i][j] = W3[m][k][i][j];
        }
      }
    }
  }

  hls::stream<DATA_T> I_strm;
  I_i_k_loop: for (k=0; k<3; k++) {
	I_i_x_loop: for (x=0; x<16; x++) {
	  I_i_y_loop: for (y=0; y<16; y++) {
        I_i[k][x][y] = I[k][x][y];
	  		//I_strm.write(I[k][x][y]);
      }
    }
  }

  //VGG16(I_strm, W1_i, W2_i, W3_i, B1_i, B2_i, B3_i, O_i);
  VGG16(I_i, W1_i, W2_i, W3_i, B1_i, B2_i, B3_i, O_i);

  for (m=0; m<16; m++) {
    for (x=0; x<8; x++) {
      for (y=0; y<8; y++) {
    	O[m][x][y] = O_i[m][x][y];
      }
    }
  }

}


void VGG16_sw(DATA_T I[3][16][16], DATA_T W1[8][3][3][3], DATA_T W2[8][8][3][3], DATA_T W3[16][8][3][3], DATA_T B1[8], DATA_T B2[8], DATA_T B3[16], DATA_T O[16][8][8]) {

  static DATA_T O1[8][16][16];
  static DATA_T O2[8][16][16];
  static DATA_T O3[8][8][8];

  int m, x, y, i, j, k;

  Conv2D_padding_act_relu_1_sw(I, W1, B1, O1, 8, 3, 3, 3, 16, 16, 1);
  Conv2D_padding_act_relu_2_sw(O1, W2, B2, O2, 8, 8, 3, 3, 16, 16, 1);
  MaxPooling2D_1_sw(O2, O3, 8, 2, 2, 8, 8, 2);
  Conv2D_padding_act_relu_3_sw(O3, W3, B3, O, 16, 8, 3, 3, 8, 8, 1);


}

