#include <iostream>
#include <ap_int.h>
#include "hls_stream.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>



using namespace std;

typedef ap_uint<16> DATA_T;

void vgg19_top(DATA_T I[3][224][224],DATA_T W1[64][3][3][3], DATA_T B1[64],DATA_T W2[64][64][3][3], DATA_T B2[64],DATA_T W4[128][64][3][3], DATA_T B4[128],DATA_T W5[128][128][3][3], DATA_T B5[128], DATA_T O[128][112][112]);
void vgg19_sw(DATA_T I[3][224][224],DATA_T W1[64][3][3][3], DATA_T B1[64],DATA_T W2[64][64][3][3], DATA_T B2[64],DATA_T W4[128][64][3][3], DATA_T B4[128],DATA_T W5[128][128][3][3], DATA_T B5[128], DATA_T O[128][112][112]);


int main(){

  int m, x, y, i, j, k;
  DATA_T temp;
  //size_t i_len,b_len,w_len;
  //char *i_line,*b_line,*w_line;
  //char * i_v,*b_v,*w_v;
  static DATA_T I[3][224][224];
	static DATA_T O0_SW[3][224][224];
	static DATA_T W1[64][3][3][3];
	static DATA_T B1[64];
	static DATA_T W2[64][64][3][3];
	static DATA_T B2[64];
	static DATA_T W4[128][64][3][3];
	static DATA_T B4[128];
	static DATA_T W5[128][128][3][3];
	static DATA_T B5[128];
	static DATA_T O_SW[128][112][112];
	static DATA_T O_HW[128][112][112];


for (k = 0; k <  3 ; k++) {
	for (x = 0; x < 224 ; x++) {
		for(y = 0; y < 224 ; y++) {
			I[k][x][y] = y%20;
			O0_SW[k][x][y] = y%20;
		}
	}
}
for (m = 0; m <  64 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 3 ; i++) {
			for (j = 0; j < 3 ; j++) {
                W1[m][k][i][j] = j*i*k;
			}
		}
	}
	B1[m] = m%20;
}
for (m = 0; m <  64 ; m++) {
	for (k = 0; k < 64 ; k++) {
		for (i = 0; i < 3 ; i++) {
			for (j = 0; j < 3 ; j++) {
				W2[m][k][i][j] = (k*i*j)%20;
			}
		}
	}
	B2[m] = m%20;
}

for (m = 0; m <  128 ; m++) {
	for (k = 0; k < 64 ; k++) {
		for (i = 0; i < 3 ; i++) {
			for (j = 0; j < 3 ; j++) {
				W4[m][k][i][j] = (k*i*j)%20;
			}
		}
	}
	B4[m] = m%20;
}

for (m = 0; m <  128 ; m++) {
	for (k = 0; k < 128 ; k++) {
		for (i = 0; i < 3 ; i++) {
			for (j = 0; j < 3 ; j++) {
				W5[m][k][i][j] = (k*i*j)%20;
			}
		}
	}
	B5[m] = m%20;
}



  vgg19_top(I,W1,B1,W2,B2,W4,B4,W5,B5,O_HW);
  vgg19_sw(I,W1,B1,W2,B2,W4,B4,W5,B5,O_SW);

   int err_cnt = 0;
   for (m=0; m<128; m++) {
       for (x=0; x<112; x++) {
          for (y=0; y<112; y++) {
              if (O_HW[m][x][y] != O_SW[m][x][y]) {
                printf("SW: O[%d][%d][%d] = %d", m, x, y, O_SW[m][x][y]);
                printf("HW: O[%d][%d][%d] = %d", m, x, y, O_HW[m][x][y]);
                err_cnt++;}
           }
       }
   }


  int ret_val;
  if (err_cnt == 0) {
    printf("*** TEST PASSED ***\n");
    ret_val = 0;
  } else {
    printf("!!! TEST FAILED - %d mismatches detected !!!\n\n",err_cnt);
    ret_val = -1;
  }

  return ret_val;
}
