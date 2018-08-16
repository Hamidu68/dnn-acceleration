#include "vgg16.h"
#include <iostream>

using namespace std;


void Conv2D_padding_act_relu_1(DATA_T I[3][16][16], DATA_T W[8][3][3][3], DATA_T B[8], DATA_T O[8][16][16], int M, int C, int R, int S, int E, int F, int U) {
#pragma HLS INLINE
  int m, x, y, i, j, k;
  DATA_T ifm, ofm;

  Conv2D_1_m_loop: for (m=0; m<8; m++) {
    Conv2D_1_x_loop: for (x=0; x<16; x++) {
	  Conv2D_1_y_loop: for (y=0; y<16; y++) {
#pragma HLS PIPELINE
        ofm = B[m];
        Conv2D_1_k_loop: for (k=0; k<3; k++) {
          Conv2D_1_i_loop:for (i=0; i<3; i++) {
        	Conv2D_1_j_loop: for (j=0; j<3; j++) {
              if (x+i < 16 && y+j < 16) {
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


void Conv2D_padding_act_relu_2(DATA_T I[8][16][16], DATA_T W[8][8][3][3], DATA_T B[8], DATA_T O[8][16][16], int M, int C, int R, int S, int E, int F, int U) {
#pragma HLS INLINE
  int m, x, y, i, j, k;

  DATA_T ifm, ofm;

  Conv2D_2_m_loop: for (m=0; m<8; m++) {
	Conv2D_2_x_loop: for (x=0; x<16; x++) {
      Conv2D_2_y_loop: for (y=0; y<16; y++) {
#pragma HLS PIPELINE
        ofm = B[m];
        Conv2D_2_k_loop: for (k=0; k<8; k++) {
          Conv2D_2_i_loop: for (i=0; i<3; i++) {
        	Conv2D_2_j_loop: for (j=0; j<3; j++) {
              if (x+i < 16 && y+j < 16) {
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


void MaxPooling2D_1(DATA_T I[8][16][16], DATA_T O[8][8][8], int M, int R, int S, int F, int E, int U) {
#pragma HLS INLINE
  int m, x, y, i, j;
  double max;

  Pool_1_m_loop: for (m=0; m<M; m++) {
	Pool_1_x_loop: for (x=0; x<F; x++) {
	  Pool_1_y_loop: for (y=0; y<E; y++) {
#pragma HLS PIPELINE
        max = I[m][U*x][U*y];
        Pool_1_i_loop: for (i=0; i<R; i++) {
          Pool_1_j_loop: for (j=0; j<S; j++) {
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

void Conv2D_padding_act_relu_3(DATA_T I[8][8][8], DATA_T W[16][8][3][3], DATA_T B[16], DATA_T O[16][8][8], int M, int C, int R, int S, int E, int F, int U) {
#pragma HLS INLINE
  int m, x, y, i, j, k;
  DATA_T ifm, ofm;

  Conv2D_3_m_loop: for (m=0; m<16; m++) {
	Conv2D_3_x_loop: for (x=0; x<8; x++) {
	  Conv2D_3_y_loop: for (y=0; y<8; y++) {
#pragma HLS PIPELINE
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
        if (ofm < 0) { // relu activation
          ofm = 0;
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
        if (ofm < 0) { // relu activation
          ofm = 0;
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


void VGG16(DATA_T I_i[3][16][16], DATA_T W1_i[8][3][3][3], DATA_T W2_i[8][8][3][3], DATA_T W3_i[16][8][3][3], DATA_T B1_i[8], DATA_T B2_i[8], DATA_T B3_i[16], DATA_T O_i[16][8][8]) {
#pragma HLS DATAFLOW

  DATA_T O1[8][16][16];
  DATA_T O2[8][16][16];
  DATA_T O3[8][8][8];

  Conv2D_padding_act_relu_1(I_i, W1_i, B1_i, O1, 8, 3, 3, 3, 16, 16, 1);
  Conv2D_padding_act_relu_2(O1, W2_i, B2_i, O2, 8, 8, 3, 3, 16, 16, 1);
  MaxPooling2D_1(O2, O3, 8, 2, 2, 8, 8, 2);
  Conv2D_padding_act_relu_3(O3, W3_i, B3_i, O_i, 16, 8, 3, 3, 8, 8, 1);

}

void VGG16_top(DATA_T I[3][16][16], DATA_T W1[8][3][3][3], DATA_T W2[8][8][3][3], DATA_T W3[16][8][3][3], DATA_T B1[8], DATA_T B2[8], DATA_T B3[16], DATA_T O[16][8][8]) {

  DATA_T W1_i[8][3][3][3];
  DATA_T W2_i[8][8][3][3];
  DATA_T W3_i[16][8][3][3];
  DATA_T B1_i[8];
  DATA_T B2_i[8];
  DATA_T B3_i[16];
  DATA_T O_i[16][8][8];
  DATA_T I_i[3][16][16];

  DATA_T O1[8][16][16];
  DATA_T O2[8][16][16];
  DATA_T O3[8][8][8];

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

