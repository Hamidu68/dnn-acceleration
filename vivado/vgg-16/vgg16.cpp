#include <iostream>
#include <ap_int.h>
#include <hls_stream.h>


//using namespace std;
//typedef int DATA_T;
typedef ap_uint<16> DATA_T;

typedef ap_uint<256> uint256_t;
typedef ap_uint<512> uint512_t;

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
//void Conv2D_padding_act_relu_1(hls::stream<DATA_T> &I_strm, DATA_T W[8][3][3][3], DATA_T B[8], hls::stream<uint256_t> &O_strm) {
#pragma HLS INLINE
  int m, x, y, i, j, k;
  //DATA_T ifm, ofm;

  DATA_T I[3][16][16];

  DATA_T ifm[8], ofm[8];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0


  uint256_t O;

  DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;

 	int p2=0;
 	int p5=0;

 	DATA_T s0, s1, s2, s3, s4, s5, s6;
 	DATA_T s54, s55, s56, s57, s58, s59, s60;
	DATA_T s108, s109, s110, s111, s112, s113, s114;

  DATA_T buf2[48];
  DATA_T buf5[48];


//#pragma HLS ARRAY_PARTITION variable=I complete dim=1
//#pragma HLS ARRAY_PARTITION variable=I complete dim=2
//#pragma HLS ARRAY_PARTITION variable=I complete dim=3


 	Conv2D_1_x_loop: for (x=0; x<18; x++) {
 		Conv2D_1_y_loop: for (y=0; y<18; y++) {
 	 		Conv2D_1_k_loop: for (k=0; k<3; k++) {
 	 			if (x<16 && y<16) {
 	 				//I[k][x][y] = I_strm.read();
 	 				s0 = I_strm.read();
 	 			}
 	 			if (x>=2 && y>=2) {

      		I_0 = s114;
      		I_1 = s111;
      		I_2 = s108;
      		I_3 = s60;
      		I_4 = s57;
      		I_5 = s54;
      		I_6 = s6;
      		I_7 = s3;
      		I_8 = s0;

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

/*
      		if (k==2) {
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
*/

#if 1
 	 				if (k == 2) {
 	 					//O[Tm+m][x-2][y-2] = ofm;
 	 					//O_strm.write(ofm);
 	 					O_strm.write(ofm[0]);
 	 					O_strm.write(ofm[1]);
 	 					O_strm.write(ofm[2]);
 	 					O_strm.write(ofm[3]);
 	 					O_strm.write(ofm[4]);
 	 					O_strm.write(ofm[5]);
 	 					O_strm.write(ofm[6]);
 	 					O_strm.write(ofm[7]);
 	 					/*
 	 					O_strm.write(ofm[0]<(DATA_T)0 ? (DATA_T)0 : ofm[0]);
 	 					O_strm.write(ofm[1]<(DATA_T)0 ? (DATA_T)0 : ofm[1]);
 	 					O_strm.write(ofm[2]<(DATA_T)0 ? (DATA_T)0 : ofm[2]);
 	 					O_strm.write(ofm[3]<(DATA_T)0 ? (DATA_T)0 : ofm[3]);
 	 					O_strm.write(ofm[4]<(DATA_T)0 ? (DATA_T)0 : ofm[4]);
 	 					O_strm.write(ofm[5]<(DATA_T)0 ? (DATA_T)0 : ofm[5]);
 	 					O_strm.write(ofm[6]<(DATA_T)0 ? (DATA_T)0 : ofm[6]);
 	 					O_strm.write(ofm[7]<(DATA_T)0 ? (DATA_T)0 : ofm[7]);
						*/
 	 				}
#endif

   		  }
 	 			// shift register

 	 	  	s114 = s113;
 	 	  	s113 = s112;
 	 	  	s112 = s111;

 	 	  	s111 = s110;
 	 	  	s110 = s109;
 	 	  	s109 = s108;

	 	  	s108 = buf5[p5];
	 	  	buf5[p5] = s60;
	 	  	p5++;
	 	  	if (p5 == 47) p5 = 0;

 	 	  	s60 = s59;
 	 	  	s59 = s58;
 	 	  	s58 = s57;

 	 	  	s57 = s56;
 	 	  	s56 = s55;
 	 	  	s55 = s54;

 	 	  	s54 = buf2[p2];
 	 	  	buf2[p2] = s6;
 	 	  	p2++;
 	 	  	if (p2 == 47) p2 = 0;

 	 	  	s6 = s5;
 	 	  	s5 = s4;
 	 	  	s4 = s3;

 	 	  	s3 = s2;
 	 	  	s2 = s1;
 	 	  	s1 = s0;

  	  }
    }
  }

}

void Unpack1(hls::stream<uint256_t> &I_strm, hls::stream<DATA_T> &O_strm) {
	int  x, y;
  uint256_t I;
  DATA_T O;

 	Unpack1_x_loop: for (x=0; x<16; x++) {
 		Unpack1_y_loop: for (y=0; y<16; y++) {
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


#if 0
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
#endif

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


void Unpack2(hls::stream<uint256_t> &I_strm, hls::stream<DATA_T> &O_strm) {
	int  x, y;
  uint256_t I;
  DATA_T O;

 	Unpack2_x_loop: for (x=0; x<16; x++) {
 		Unpack2_y_loop: for (y=0; y<16; y++) {
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


#if 0 // scalar replaced version
void Conv2D_padding_act_relu_3(hls::stream<DATA_T> &I_strm, DATA_T W[16][8][3][3], DATA_T B[16], hls::stream<DATA_T> &O_strm) {
//void Conv2D_padding_act_relu_3(hls::stream<DATA_T> &I_strm, DATA_T W[16][8][3][3], DATA_T B[16], hls::stream<uint512_t> &O_strm) {
//void Conv2D_padding_act_relu_3(hls::stream<DATA_T> &I_strm, DATA_T W[16][8][3][3], DATA_T B[16], DATA_T O[16][8][8]) {
//void Conv2D_padding_act_relu_3(DATA_T I[8][8][8], DATA_T W[16][8][3][3], DATA_T B[16], DATA_T O[16][8][8]) {
#pragma HLS INLINE
  int m, x, y, i, j, k;
  //DATA_T ifm, ofm;

  uint512_t O;

  int p;
 	int p2=0;
 	int p5=0;

 	//DATA_T s0, s1, s2, s3, s4, s5, s6;
 	//DATA_T s54, s55, s56, s57, s58, s59, s60;
	//DATA_T s108, s109, s110, s111, s112, s113, s114;

 	DATA_T s0, s1, s2, s3, s4, s5, s6, s7;
 	DATA_T s8, s9, s10, s11, s12, s13, s14, s15, s16;
 	DATA_T s80, s81, s82, s83, s84, s85, s86, s87;
 	DATA_T s88, s89, s90, s91, s92, s93, s94, s95, s96;
 	DATA_T s160, s161, s162, s163, s164, s165, s166, s167, s168;
 	DATA_T s169, s170, s171, s172, s173, s174, s175, s176;

  //DATA_T buf0[8];
  //DATA_T buf1[8];
  DATA_T buf2[64];
  //DATA_T buf3[8];
  //DATA_T buf4[8];
  DATA_T buf5[64];
  //DATA_T buf6[8];
  //DATA_T buf7[8];
#if 0
#pragma HLS ARRAY_PARTITION variable=buf0 complete dim=0
#pragma HLS ARRAY_PARTITION variable=buf1 complete dim=0
#pragma HLS ARRAY_PARTITION variable=buf3 complete dim=0
#pragma HLS ARRAY_PARTITION variable=buf4 complete dim=0
#pragma HLS ARRAY_PARTITION variable=buf6 complete dim=0
#pragma HLS ARRAY_PARTITION variable=buf7 complete dim=0
#endif

  DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;

  DATA_T ifm[16], ofm[16];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

  //DATA_T I[8][8][8];
  DATA_T I[8][3][8];

#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

 	Conv2D_3_x_loop: for (x=0; x<10; x++) {
 		Conv2D_3_y_loop: for (y=0; y<10; y++) {
  		Conv2D_3_k_loop: for (k=0; k<8; k++) {

   	 		if (x<8 && y<8) {
   	 			//I[k][x%3][y] = I_strm.read();
   	 			s0 = I_strm.read();
   	 		}
  	 		if (x>=2 && y>=2) {


      		I_0 = s176;
      		I_1 = s168;
      		I_2 = s160;
      		I_3 = s96;
      		I_4 = s88;
      		I_5 = s80;
      		I_6 = s16;
      		I_7 = s8;
      		I_8 = s0;
      		/*
      		I_0 = I[k][(x-2)%3][(y-2)];
      		I_1 = I[k][(x-2)%3][(y-1)];
      		I_2 = I[k][(x-2)%3][(y)];
      		I_3 = I[k][(x-1)%3][(y-2)];
      		I_4 = I[k][(x-1)%3][(y-1)];
      		I_5 = I[k][(x-1)%3][(y)];
      		I_6 = I[k][(x)%3][(y-2)];
      		I_7 = I[k][(x)%3][(y-1)];
      		I_8 = I[k][(x)%3][(y)];
      		 */
  	 			Conv2D_3_m_loop: for (m=0; m<16; m++) {
#pragma HLS PIPELINE

  	 				if (k==0) {
  	 					ofm[m] = B[m];
  	 				}
  	 				/*
  	 				Conv2D_3_i_loop: for (i=0; i<3; i++) {
  	 					Conv2D_3_j_loop: for (j=0; j<3; j++) {
  	 						if (x-2+i < 8 && y-2+j < 8) {
  	 							ifm[m] = I[k][(x-2+i)%3][y-2+j];
  	 						} else {
  	 							ifm[m] = 0; // zero padding
  	 						}
  	 						ofm[m] = ofm[m] + ifm[m] * W[m][k][i][j];
  	 					}
  	 				}
  	 				*/

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

  	 				/*if (Tk+k == 7) {
   	 				//O[Tm+m][x-2][y-2] = ofm;
   	 				O_strm.write(ofm);
   	 				}*/
	 				}
  	 			/*
  	 			if (k==7) {
  	 				O = (uint512_t)ofm[0];
  	 				for (m=1; m<16; m++) {
#pragma HLS UNROLL
 							O = O | (uint512_t)ofm[m] << 32*m;
  	 				}
  	 				O_strm.write(O);
  	 			}
  	 			*/

  	 			if (k==7) {
  	 				for (m=0; m<16; m++) {
#pragma HLS UNROLL
  	 					O_strm.write(ofm[m]);
  	 				}
  	 			}

 	 			}
 	 			// shift register

  	 		s176 = s175;
  	 		s175 = s174;
  	 		s174 = s173;
  	 		s173 = s172;
  	 		s172 = s171;
  	 		s171 = s170;
  	 		s170 = s169;
  	 		s169 = s168;

  	 		s168 = s167;
  	 		s167 = s166;
  	 		s166 = s165;
  	 		s165 = s164;
  	 		s164 = s163;
  	 		s163 = s162;
  	 		s162 = s161;
  	 		s161 = s160;

	 	  	s160 = buf5[p5];
	 	  	buf5[p5] = s96;
	 	  	p5++;
	 	  	if (p5 == 63) p5 = 0;


	 	  	s96 = s95;
	 	  	s95 = s94;
	 	  	s94 = s93;
	 	  	s93 = s92;
	 	  	s92 = s91;
	 	  	s91 = s90;
	 	  	s90 = s89;
	 	  	s89 = s88;

	 	  	s88 = s87;
	 	  	s87 = s86;
	 	  	s86 = s85;
	 	  	s85 = s84;
	 	  	s84 = s83;
	 	  	s83 = s82;
	 	  	s82 = s81;
	 	  	s81 = s80;

 	 	  	s80 = buf2[p2];
 	 	  	buf2[p2] = s16;
 	 	  	p2++;
 	 	  	if (p2 == 63) p2 = 0;

 	 	  	s16 = s15;
 	 	  	s15 = s14;
 	 	  	s14 = s13;
 	 	  	s13 = s12;
 	 	  	s12 = s11;
 	 	  	s11 = s10;
 	 	  	s10 = s9;
 	 	  	s9 = s8;

 	 	  	s8 = s7;
 	 	  	s7 = s6;
 	 	  	s6 = s5;
 	 	  	s5 = s4;
 	 	  	s4 = s3;
 	 	  	s3 = s2;
 	 	  	s2 = s1;
 	 	  	s1 = s0;



 			}
   	}
 	}
}
#endif


#if 0 // without scalar replace (many LUTs (84932 LUTs))
void Conv2D_padding_act_relu_3(hls::stream<DATA_T> &I_strm, DATA_T W[16][8][3][3], DATA_T B[16], hls::stream<DATA_T> &O_strm) {
//void Conv2D_padding_act_relu_3(hls::stream<DATA_T> &I_strm, DATA_T W[16][8][3][3], DATA_T B[16], hls::stream<uint512_t> &O_strm) {
//void Conv2D_padding_act_relu_3(hls::stream<DATA_T> &I_strm, DATA_T W[16][8][3][3], DATA_T B[16], DATA_T O[16][8][8]) {
//void Conv2D_padding_act_relu_3(DATA_T I[8][8][8], DATA_T W[16][8][3][3], DATA_T B[16], DATA_T O[16][8][8]) {
#pragma HLS INLINE
  int m, x, y, i, j, k;
  //DATA_T ifm, ofm;

  uint512_t O;

  DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;

  DATA_T ifm[16], ofm[16];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

  DATA_T I[8][3][8];

#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

 	Conv2D_3_x_loop: for (x=0; x<10; x++) {
 		Conv2D_3_y_loop: for (y=0; y<10; y++) {
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
  	 				/*
  	 				Conv2D_3_i_loop: for (i=0; i<3; i++) {
  	 					Conv2D_3_j_loop: for (j=0; j<3; j++) {
  	 						if (x-2+i < 8 && y-2+j < 8) {
  	 							ifm[m] = I[k][(x-2+i)%3][y-2+j];
  	 						} else {
  	 							ifm[m] = 0; // zero padding
  	 						}
  	 						ofm[m] = ofm[m] + ifm[m] * W[m][k][i][j];
  	 					}
  	 				}
  	 				*/

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

  	 				/*if (Tk+k == 7) {
   	 				//O[Tm+m][x-2][y-2] = ofm;
   	 				O_strm.write(ofm);
   	 				}*/
	 				}
  	 			/*
  	 			if (k==7) {
  	 				O = (uint512_t)ofm[0];
  	 				for (m=1; m<16; m++) {
#pragma HLS UNROLL
 							O = O | (uint512_t)ofm[m] << 32*m;
  	 				}
  	 				O_strm.write(O);
  	 			}
  	 			*/

  	 			if (k==7) {
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
#endif




#if 0 // old version
void Conv2D_padding_act_relu_3(hls::stream<DATA_T> &I_strm, DATA_T W[16][8][3][3], DATA_T B[16], hls::stream<DATA_T> &O_strm) {
//void Conv2D_padding_act_relu_3(hls::stream<DATA_T> &I_strm, DATA_T W[16][8][3][3], DATA_T B[16], hls::stream<uint512_t> &O_strm) {
//void Conv2D_padding_act_relu_3(hls::stream<DATA_T> &I_strm, DATA_T W[16][8][3][3], DATA_T B[16], DATA_T O[16][8][8]) {
//void Conv2D_padding_act_relu_3(DATA_T I[8][8][8], DATA_T W[16][8][3][3], DATA_T B[16], DATA_T O[16][8][8]) {
#pragma HLS INLINE
  int m, x, y, i, j, k;
  //DATA_T ifm, ofm;

  uint512_t O;

  DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;

  DATA_T ifm[16], ofm[16];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

  DATA_T I[8][3][8];

#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

 	Conv2D_3_x_loop: for (x=0; x<10; x++) {
 		Conv2D_3_y_loop: for (y=0; y<10; y++) {
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
  	 				/*
  	 				Conv2D_3_i_loop: for (i=0; i<3; i++) {
  	 					Conv2D_3_j_loop: for (j=0; j<3; j++) {
  	 						if (x-2+i < 8 && y-2+j < 8) {
  	 							ifm[m] = I[k][(x-2+i)%3][y-2+j];
  	 						} else {
  	 							ifm[m] = 0; // zero padding
  	 						}
  	 						ofm[m] = ofm[m] + ifm[m] * W[m][k][i][j];
  	 					}
  	 				}
  	 				*/

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

  	 				/*if (Tk+k == 7) {
   	 				//O[Tm+m][x-2][y-2] = ofm;
   	 				O_strm.write(ofm);
   	 				}*/
	 				}
  	 			/*
  	 			if (k==7) {
  	 				O = (uint512_t)ofm[0];
  	 				for (m=1; m<16; m++) {
#pragma HLS UNROLL
 							O = O | (uint512_t)ofm[m] << 32*m;
  	 				}
  	 				O_strm.write(O);
  	 			}
  	 			*/

  	 			if (k==7) {
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
#endif


void Unpack3(hls::stream<uint512_t> &I_strm, hls::stream<DATA_T> &O_strm) {
	int  x, y, m;
  uint512_t I;
  DATA_T O;

 	Unpack3_x_loop: for (x=0; x<8; x++) {
 		Unpack3_y_loop: for (y=0; y<8; y++) {
 			I = I_strm.read();

 			Unpack3_m_loop: for (m=0; m<16; m++) {
#pragma HLS UNROLL
 				O = DATA_T(I >> 32*m);
 				O_strm.write(O);
 				/*
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
 				*/
 			}
 		}
 	}
}


#if 1 // DAC 2017 like optimized code
void Conv2D_padding_act_relu_3(hls::stream<DATA_T> &I_strm, DATA_T W[16][8][3][3], DATA_T B[16], hls::stream<DATA_T> &O_strm) {
//void Conv2D_padding_act_relu_3(hls::stream<DATA_T> &I_strm, DATA_T W[16][8][3][3], DATA_T B[16], DATA_T O[16][8][8]) {
//void Conv2D_padding_act_relu_3(DATA_T I[8][8][8], DATA_T W[16][8][3][3], DATA_T B[16], DATA_T O[16][8][8]) {
#pragma HLS INLINE
  int m, x, y, i, j, k;
  //DATA_T ifm, ofm;



  DATA_T ifm[16], ofm[16];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

  DATA_T I[8][4][8];
  DATA_T O[16][2][8];

#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

#pragma HLS ARRAY_PARTITION variable=O complete dim=1
#pragma HLS ARRAY_PARTITION variable=O complete dim=2
#pragma HLS ARRAY_PARTITION variable=O complete dim=3

  /*
 	Conv2D_3_x_init_loop: for (x=0; x<3; x++) {
 		Conv2D_3_y_init_loop: for (y=0; y<8; y++) {
  		Conv2D_3_k_init_loop: for (k=0; k<8; k++) {
#pragma HLS PIPELINE
  			I[k][x%4][y] = I_strm.read();
  		}
 		}
 	}
 	*/

 	Conv2D_3_x_loop: for (x=0; x<12; x++) {
//#pragma HLS dependence variable=I intra false
//#pragma HLS dependence variable=O intra false

 		/*
 		Conv2D_3_y_ld_loop: for (y=0; y<8; y++) {
  		Conv2D_3_k_ld_loop: for (k=0; k<8; k++) {
#pragma HLS PIPELINE
  	 		if (x < 8) {
  	 			I[k][x%4][y] = I_strm.read();
  	 		}
  		}
 		}
 		*/

 		Conv2D_3_y_loop: for (y=0; y<8; y++) {
  		Conv2D_3_k_loop: for (k=0; k<8; k++) {

  	 		if (x < 8) {
  	 			I[k][x%4][y] = I_strm.read();
  	 		}

	 			Conv2D_3_m_loop: for (m=0; m<16; m++) {
#pragma HLS PIPELINE

	 		 		if (x >= 3 && x < 11) {


	 		 			if (k==0) {
	 		 				ofm[m] = B[m];
	 		 			} else {
	 		 				ofm[m] = O[m][(x-3)%2][y];
	 		 			}
#if 0
  					Conv2D_3_i_loop: for (i=0; i<3; i++) {
  						Conv2D_3_j_loop: for (j=0; j<3; j++) {
  							if (x-3+i < 8 && y+j < 8) {
  								ifm[m] = I[k][(x-3+i)%4][y+j];
  							} else {
  								ifm[m] = 0; // zero padding
  							}
  							ofm[m] = ofm[m] + ifm[m] * W[m][k][i][j];
  						}
  					}
#else

  					if (x-3+0 < 8 && y+0 < 8) {
  						ifm[m] = I[k][(x-3+0)%4][y+0];
  					} else {
  						ifm[m] = 0; // zero padding
  					}
  					ofm[m] = ofm[m] + ifm[m] * W[m][k][0][0];

  					if (x-3+0 < 8 && y+1 < 8) {
  						ifm[m] = I[k][(x-3+0)%4][y+1];
  					} else {
  						ifm[m] = 0; // zero padding
  					}
  					ofm[m] = ofm[m] + ifm[m] * W[m][k][0][1];

  					if (x-3+0 < 8 && y+2 < 8) {
  						ifm[m] = I[k][(x-3+0)%4][y+2];
  					} else {
  						ifm[m] = 0; // zero padding
  					}
  					ofm[m] = ofm[m] + ifm[m] * W[m][k][0][2];

  					if (x-3+1 < 8 && y+0 < 8) {
  						ifm[m] = I[k][(x-3+1)%4][y+0];
  					} else {
  						ifm[m] = 0; // zero padding
  					}
  					ofm[m] = ofm[m] + ifm[m] * W[m][k][1][0];

  					if (x-3+1 < 8 && y+1 < 8) {
  						ifm[m] = I[k][(x-3+1)%4][y+1];
  					} else {
  						ifm[m] = 0; // zero padding
  					}
  					ofm[m] = ofm[m] + ifm[m] * W[m][k][1][1];

  					if (x-3+1 < 8 && y+2 < 8) {
  						ifm[m] = I[k][(x-3+1)%4][y+2];
  					} else {
  						ifm[m] = 0; // zero padding
  					}
  					ofm[m] = ofm[m] + ifm[m] * W[m][k][1][2];

  					if (x-3+2 < 8 && y+0 < 8) {
  						ifm[m] = I[k][(x-3+2)%4][y+0];
  					} else {
  						ifm[m] = 0; // zero padding
  					}
  					ofm[m] = ofm[m] + ifm[m] * W[m][k][2][0];

  					if (x-3+2 < 8 && y+1 < 8) {
  						ifm[m] = I[k][(x-3+2)%4][y+1];
  					} else {
  						ifm[m] = 0; // zero padding
  					}
  					ofm[m] = ofm[m] + ifm[m] * W[m][k][2][1];

  					if (x-3+2 < 8 && y+2 < 8) {
  						ifm[m] = I[k][(x-3+2)%4][y+2];
  					} else {
  						ifm[m] = 0; // zero padding
  					}
  					ofm[m] = ofm[m] + ifm[m] * W[m][k][2][2];

#endif


  					O[m][(x-3)%2][y] = ofm[m];

  				}
		 			if (k == 7) {
		 				if (x>=4) {
		 					O_strm.write(O[m][(x-4)%2][y]);
		 				}
		 			}
	 			}
  		}
  		/*
 			Conv2D_3_m_st_loop: for (m=0; m<16; m++) {
#pragma HLS PIPELINE
 				if (x>=4) {
 					O_strm.write(O[m][(x-4)%2][y]);
 				}
  	 	}
  	 	*/
 		}


/*
 		Conv2D_3_y_st_loop: for (y=0; y<8; y++) {
 			Conv2D_3_m_st_loop: for (m=0; m<16; m++) {
#pragma HLS PIPELINE
 				if (x>=4) {
 					O_strm.write(O[m][(x-4)%2][y]);
 				}
  	 	}
  	}
*/

 	}



}
#endif


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

#pragma HLS ARRAY_PARTITION variable=W3_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W3_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W3_i complete dim=4

#pragma HLS ARRAY_PARTITION variable=B3_i complete



#pragma HLS DATAFLOW
  hls::stream<uint256_t> O1_packed_strm("O1_packed_strm");
  hls::stream<DATA_T> O1_strm("O1_strm");
  hls::stream<uint256_t> O2_packed_strm("O2_packed_strm");
  hls::stream<DATA_T> O2_strm("O2_strm");
  hls::stream<DATA_T> O3_strm("O3_strm");
  hls::stream<DATA_T> I_strm("I_strm");
  hls::stream<uint512_t> O_packed_strm("O_packed_strm");
  hls::stream<DATA_T> O_strm("O_strm");
//#pragma HLS stream variable=O1_strm depth=2


  Stream_input(I, I_strm);
  Conv2D_padding_act_relu_1(I_strm, W1_i, B1_i, O1_strm);
  //Conv2D_padding_act_relu_1(I_strm, W1_i, B1_i, O1_packed_strm);
  //Unpack1(O1_packed_strm, O1_strm);
  //Conv2D_padding_act_relu_2(O1_strm, W2_i, B2_i, O2_strm);
  Conv2D_padding_act_relu_2(O1_strm, W2_i, B2_i, O2_packed_strm);
  Unpack2(O2_packed_strm, O2_strm);
  MaxPooling2D_1(O2_strm, O3_strm);
  //Conv2D_padding_act_relu_3(O3_strm, W3_i, B3_i, O_packed_strm);
  Conv2D_padding_act_relu_3(O3_strm, W3_i, B3_i, O_strm);
  //Unpack3(O_packed_strm, O_strm);
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

