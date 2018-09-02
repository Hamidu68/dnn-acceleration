#include "hls_stream.h"
#include <iostream>

using namespace std;

typedef int DATA_T;
//typedef float DATA_T;


void sample_top(DATA_T I[16][16], DATA_T W1[16][16], DATA_T W2[16][16], DATA_T W3[16][16], DATA_T O[16][16]);
void sample_sw(DATA_T I[16][16], DATA_T W1[16][16], DATA_T W2[16][16], DATA_T W3[16][16], DATA_T O[16][16]);


int main() {

  static DATA_T I[16][16];
  static DATA_T W1[16][16];
  static DATA_T W2[16][16];
  static DATA_T W3[16][16];
  static DATA_T O[16][16];

  static DATA_T I_sw[16][16];
  static DATA_T W1_sw[16][16];
  static DATA_T W2_sw[16][16];
  static DATA_T W3_sw[16][16];
  static DATA_T O_sw[16][16];

  int i, j;


  for (i=0; i<16; i++) {
    for (j=0; j<16; j++) {
      I_sw[i][j] = (((i+0)^2)+j*7+(j^2)+(i+5)) % 255;
      I[i][j]    = (((i+0)^2)+j*7+(j^2)+(i+5)) % 255;
      //cout <<  I_sw[i][j] << "," << I[i][j] << endl;
    }
  }
  for (i=0; i<16; i++) {
    for (j=0; j<16; j++) {
      W1_sw[i][j] = (((j+0)^5)+i*3+(i^3)+(j+3)) % 255;
      W1[i][j]    = (((j+0)^5)+i*3+(i^3)+(j+3)) % 255;
      //cout <<  W1_sw[i][j] << "," << W1[i][j] << endl;
    }
  }
  for (i=0; i<16; i++) {
    for (j=0; j<16; j++) {
      W2_sw[i][j] = (((j+0)^3)+j+(i^5)+(j-3)) % 255;
      W2[i][j]    = (((j+0)^3)+j+(i^5)+(j-3)) % 255;
      //cout <<  W2_sw[i][j] << "," << W2[i][j] << endl;
    }
  }
  for (i=0; i<16; i++) {
    for (j=0; j<16; j++) {
      W3_sw[i][j] = (((j+0)^5)+3*j+(i^2)+(j-3)) % 255;
      W3[i][j]    = (((j+0)^5)+3*j+(i^2)+(j-3)) % 255;
      //cout <<  W3_sw[i][j] << "," << W3[i][j] << endl;
    }
  }

  //sample_sw(I_sw, W1_sw, W2_sw, W3_sw, O_sw);
  sample_sw(I, W1, W2, W3, O_sw);
  sample_top(I, W1, W2, W3, O);


  int err_cnt = 0;
  for (i=0; i<16; i++) {
    for (j=0; j<16; j++) {
    	if (O[i][j] != O_sw[i][j]) {
          cout << "SW: O[" << i << "][" << j << "] = " << O_sw[i][j] << endl;
          cout << "HW: O[" << i << "][" << j << "] = " << O[i][j] << endl;
    	  err_cnt++;
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


}

