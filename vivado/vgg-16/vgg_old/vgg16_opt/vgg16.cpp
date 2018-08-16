#include <iostream>
#include <hls_stream.h>

using namespace std;
typedef int DATA_T;

void Conv2D_padding_act_relu_1(DATA_T I[3][16][16], DATA_T W[8][3][3][3], DATA_T B[8], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
  int m, x, y, i, j, k;
  DATA_T ifm, ofm;

#pragma HLS ARRAY_PARTITION variable=I complete dim=1

  DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;

  DATA_T O_buf[8];

  Conv2D_1_x_loop: for (x=0; x<16; x++) {
  	Conv2D_1_y_loop: for (y=0; y<16; y++) {
#pragma HLS PIPELINE
      Conv2D_1_k_loop: for (k=0; k<3; k++) {
      	I_0 = I[k][x+0][y+0];
      	I_1 = I[k][x+0][y+1];
      	I_2 = I[k][x+0][y+2];
      	I_3 = I[k][x+1][y+0];
      	I_4 = I[k][x+1][y+1];
      	I_5 = I[k][x+1][y+2];
      	I_6 = I[k][x+2][y+0];
      	I_7 = I[k][x+2][y+1];
      	I_8 = I[k][x+2][y+2];

      	Conv2D_1_m_loop: for (m=0; m<8; m++) {
          if (k==0) {
          	ofm = B[m];
          } else {
          	ofm = O_buf[m];
          }

          if (x+0 < 16 && y+0 < 16) {
            ifm = I_0;
          } else {
            ifm = 0; // zero padding
          }
          ofm = ofm + ifm * W[m][k][0][0];

          if (x+0 < 16 && y+1 < 16) {
            ifm = I_1;
          } else {
            ifm = 0; // zero padding
          }
          ofm = ofm + ifm * W[m][k][0][1];

          if (x+0 < 16 && y+2 < 16) {
            ifm = I_2;
          } else {
            ifm = 0; // zero padding
          }
          ofm = ofm + ifm * W[m][k][0][2];

          if (x+1 < 16 && y+0 < 16) {
            ifm = I_3;
          } else {
            ifm = 0; // zero padding
          }
          ofm = ofm + ifm * W[m][k][1][0];

          if (x+1 < 16 && y+1 < 16) {
            ifm = I_4;
          } else {
            ifm = 0; // zero padding
          }
          ofm = ofm + ifm * W[m][k][1][1];

          if (x+1 < 16 && y+2 < 16) {
            ifm = I_5;
          } else {
            ifm = 0; // zero padding
          }
          ofm = ofm + ifm * W[m][k][1][2];

          if (x+2 < 16 && y+0 < 16) {
            ifm = I_6;
          } else {
            ifm = 0; // zero padding
          }
          ofm = ofm + ifm * W[m][k][2][0];

          if (x+2 < 16 && y+1 < 16) {
            ifm = I_7;
          } else {
            ifm = 0; // zero padding
          }
          ofm = ofm + ifm * W[m][k][2][1];

          if (x+2 < 16 && y+2 < 16) {
            ifm = I_8;
          } else {
            ifm = 0; // zero padding
          }
          ofm = ofm + ifm * W[m][k][2][2];

          if (k<2) {
          	O_buf[m] = ofm;
          } else if (k==2) { // when k==2
        		//O[m][x][y] = ofm;
          	O_strm.write(ofm);
          }
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

void Conv2D_padding_act_relu_2(hls::stream<DATA_T> &I_strm, DATA_T W[8][8][3][3], DATA_T B[8], DATA_T O[8][16][16]) {
//void Conv2D_padding_act_relu_2(DATA_T I[8][16][16], DATA_T W[8][8][3][3], DATA_T B[8], DATA_T O[8][16][16]) {
#pragma HLS INLINE
  int m, x, y, i, j, k;

  DATA_T ifm, ofm;

  DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;

  //DATA_T I[8][16][16];
  DATA_T I[8][3][16];
/* without the following pragma, cosim dump fatal error ...*/
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
//#pragma HLS ARRAY_PARTITION variable=I cyclic factor=3 dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
//#pragma HLS ARRAY_PARTITION variable=I cyclic factor=3 dim=3

/*
 	Copy_2_x_loop: for (x=0; x<18; x++) {
 		Copy_2_y_loop: for (y=0; y<18; y++) {
 			Copy_2_k_loop: for (k=0; k<8; k++) {
 				if (x<16 && y<16) {
 					I[k][x][y] = I_strm.read();
 				}
 			}
 		}
 	}
*/

 	Conv2D_2_x_loop: for (x=0; x<18; x++) {
    Conv2D_2_y_loop: for (y=0; y<18; y++) {
#pragma HLS PIPELINE
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

      			if (k==0) {
      				ofm = B[m];
      			} else {
      				ofm = O[m][x-2][y-2];
      			}

      			if (x-2 < 16 && y-2 < 16) {
      				ifm = I_0;
      			} else {
      				ifm = 0; // zero padding
      			}
      			ofm = ofm + ifm * W[m][k][0][0];

      			if (x-2 < 16 && y-1 < 16) {
      				ifm = I_1;
      			} else {
      				ifm = 0; // zero padding
      			}
      			ofm = ofm + ifm * W[m][k][0][1];

      			if (x-2 < 16 && y < 16) {
      				ifm = I_2;
      			} else {
      				ifm = 0; // zero padding
      			}
      			ofm = ofm + ifm * W[m][k][0][2];

      			if (x-1 < 16 && y-2 < 16) {
      				ifm = I_3;
      			} else {
      				ifm = 0; // zero padding
      			}
      			ofm = ofm + ifm * W[m][k][1][0];

      			if (x-1 < 16 && y-1 < 16) {
      				ifm = I_4;
      			} else {
      				ifm = 0; // zero padding
      			}
      			ofm = ofm + ifm * W[m][k][1][1];

      			if (x-1 < 16 && y < 16) {
      				ifm = I_5;
      			} else {
      				ifm = 0; // zero padding
      			}
      			ofm = ofm + ifm * W[m][k][1][2];

      			if (x < 16 && y-2 < 16) {
      				ifm = I_6;
      			} else {
      				ifm = 0; // zero padding
      			}
      			ofm = ofm + ifm * W[m][k][2][0];

      			if (x < 16 && y-1 < 16) {
      				ifm = I_7;
      			} else {
      				ifm = 0; // zero padding
      			}
      			ofm = ofm + ifm * W[m][k][2][1];

      			if (x < 16 && y < 16) {
      				ifm = I_8;
      			} else {
      				ifm = 0; // zero padding
      			}
      			ofm = ofm + ifm * W[m][k][2][2];


      			O[m][x-2][y-2] = ofm;
      		}
        }
      }
    }
  }
}


void MaxPooling2D_1(DATA_T I[8][16][16], DATA_T O[8][8][8]) {
#pragma HLS INLINE
  int m, x, y, i, j;
  double max;


	Pool_1_x_loop: for (x=0; x<8; x++) {
	  Pool_1_y_loop: for (y=0; y<8; y++) {
	    Pool_1_m_loop: for (m=0; m<8; m++) {
        max = I[m][2*x][2*y];
        Pool_1_i_loop: for (i=0; i<2; i++) {
          Pool_1_j_loop: for (j=0; j<2; j++) {
            if (I[m][2*x+i][2*y+j] > max) {
              max = I[m][2*x+i][2*y+j];
            }
          }
        }
        O[m][x][y] = max;

      }
    }
  }
}

void Conv2D_padding_act_relu_3(DATA_T I[8][8][8], DATA_T W[16][8][3][3], DATA_T B[16], DATA_T O[16][8][8]) {
#pragma HLS INLINE
  int m, x, y, i, j, k;
  DATA_T ifm, ofm;

  Conv2D_3_m_loop: for (m=0; m<16; m++) {
	Conv2D_3_x_loop: for (x=0; x<8; x++) {
	  Conv2D_3_y_loop: for (y=0; y<8; y++) {
//#pragma HLS PIPELINE
        ofm = B[m];
        Conv2D_3_k_loop: for (k=0; k<8; k++) {
          Conv2D_3_i_loop: for (i=0; i<3; i++) {
        	Conv2D_3_j_loop: for (j=0; j<3; j++) {
              if (x+i < 8 && y+j < 8) {
                ifm = I[k][x+i][y+j];
              } else {
                ifm = 0; // zero padding
              }
              ofm = ofm + ifm * W[m][k][i][j];
            }
          }
        }
        if (ofm < 0) { // relu activation
          ofm = 0;
        }
        O[m][x][y] = ofm;
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
  double max;

  for (m=0; m<M; m++) {
    for (x=0; x<F; x++) {
      for (y=0; y<E; y++) {

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
        if (ofm < 0) { // relu activation
          ofm = 0;
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
#pragma HLS DATAFLOW

  //DATA_T O1[8][16][16];
  hls::stream<DATA_T> O1;
	DATA_T O2[8][16][16];
  DATA_T O3[8][8][8];

  Conv2D_padding_act_relu_1(I, W1_i, B1_i, O1);
  Conv2D_padding_act_relu_2(O1, W2_i, B2_i, O2);
  MaxPooling2D_1(O2, O3);
  Conv2D_padding_act_relu_3(O3, W3_i, B3_i, O);

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
  I_i_k_loop: for (k=0; k<3; k++) {
	I_i_x_loop: for (x=0; x<16; x++) {
	  I_i_y_loop: for (y=0; y<16; y++) {
        I_i[k][x][y] = I[k][x][y];
      }
    }
  }

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

