#include <iostream>
#include <ap_int.h>
#include "hls_stream.h"

using namespace std;
//typedef int DATA_T;
typedef ap_uint<32> DATA_T;

void VGG16_top(DATA_T I[3][16][16], DATA_T W1[8][3][3][3], DATA_T W2[8][8][3][3], DATA_T W3[16][8][3][3], DATA_T B1[8], DATA_T B2[8], DATA_T B3[16], DATA_T O[16][8][8]);
void VGG16_sw(DATA_T I[3][16][16], DATA_T W1[8][3][3][3], DATA_T W2[8][8][3][3], DATA_T W3[16][8][3][3], DATA_T B1[8], DATA_T B2[8], DATA_T B3[16], DATA_T O[16][8][8]);


int main() {

  static DATA_T I[3][16][16];
  static DATA_T W1[8][3][3][3];
  static DATA_T W2[8][8][3][3];
  static DATA_T W3[16][8][3][3];
  static DATA_T B1[8];
  static DATA_T B2[8];
  static DATA_T B3[16];
  static DATA_T O[16][8][8];

  static DATA_T I_sw[3][16][16];
  static DATA_T W1_sw[8][3][3][3];
  static DATA_T W2_sw[8][8][3][3];
  static DATA_T W3_sw[16][8][3][3];
  static DATA_T B1_sw[8];
  static DATA_T B2_sw[8];
  static DATA_T B3_sw[16];
  static DATA_T O_sw[16][8][8];


  int m, x, y, i, j, k;

  for (m=0; m<8; m++) {
    B1_sw[m] = ((m+3)>>1) % 255;
    B1[m]    = ((m+3)>>1) % 255;
  }
  for (m=0; m<8; m++) {
    B2_sw[m] = ((m+4)*7) % 255;
    B2[m]    = ((m+4)*7) % 255;
  }
  for (m=0; m<16; m++) {
    B3_sw[m] = ((m+5)*3) % 255;
    B3[m]    = ((m+5)*3) % 255;
  }
  for (m=0; m<8; m++) {
    for (k=0; k<3; k++) {
      for (i=0; i<3; i++) {
        for (j=0; j<3; j++) {
          W1_sw[m][k][i][j] = (((i+0)^2)+j*7+k+(m+5)) % 255;
          W1[m][k][i][j]    = (((i+0)^2)+j*7+k+(m+5)) % 255;
        }
      }
    }
  }
  for (m=0; m<8; m++) {
    for (k=0; k<8; k++) {
      for (i=0; i<3; i++) {
        for (j=0; j<3; j++) {
          W2_sw[m][k][i][j] = (((i+0)^2)+k*7+(j^2)+(m+5)) % 255;
          W2[m][k][i][j]    = (((i+0)^2)+k*7+(j^2)+(m+5)) % 255;
        }
      }
    }
  }
  for (m=0; m<16; m++) {
    for (k=0; k<8; k++) {
      for (i=0; i<3; i++) {
        for (j=0; j<3; j++) {
          W3_sw[m][k][i][j] = (((j+0)^2)+i*7+(k^2)+(m+5)) % 255;
          W3[m][k][i][j]    = (((j+0)^2)+i*7+(k^2)+(m+5)) % 255;
        }
      }
    }
  }
  for (k=0; k<3; k++) {
    for (x=0; x<16; x++) {
      for (y=0; y<16; y++) {
        I_sw[k][x][y] = (x*3+y*7+((k+1)^2)) % 255;
        I[k][x][y]    = (x*3+y*7+((k+1)^2)) % 255;
      }
    }
  }

  VGG16_sw(I_sw, W1_sw, W2_sw, W3_sw, B1_sw, B2_sw, B3_sw, O_sw);
  VGG16_top(I, W1, W2, W3, B1, B2, B3, O);


  int err_cnt = 0;
  for (m=0; m<16; m++) {
    for (x=0; x<8; x++) {
      for (y=0; y<8; y++) {
    	if (O[m][x][y] != O_sw[m][x][y]) {
          cout << "SW: O[" << m << "][" << x << "][" << y << "] = " << O_sw[m][x][y] << endl;
          cout << "HW: O[" << m << "][" << x << "][" << y << "] = " << O[m][x][y] << endl;
    	  err_cnt++;
    	}
      }
    }
  }

  int ret_val;
  if (err_cnt == 0) {
    cout << "*** TEST PASSED ***" << endl;
    ret_val = 0;
  } else {
    cout << "!!! TEST FAILED - " << err_cnt << " mismatches detected !!!";
    cout << endl;
    ret_val = -1;
  }

  return ret_val;
  //return 0;

}

