#include <iostream>
#include <ap_int.h>
#include "hls_stream.h"
#include <stdio.h>
#include <stdlib.h>
using namespace std;

typedef ap_uint<16> DATA_T;

void VGG19_top(DATA_T I[3][224][224],DATA_T W1[64][3][3][3], DATA_T B1[64],DATA_T W2[64][64][3][3], DATA_T B2[64],DATA_T W4[128][64][3][3], DATA_T B4[128],DATA_T W5[128][128][3][3], DATA_T B5[128], DATA_T O[128][112][112]);
void VGG19_sw(DATA_T I[3][224][224],DATA_T W1[64][3][3][3], DATA_T B1[64],DATA_T W2[64][64][3][3], DATA_T B2[64],DATA_T W4[128][64][3][3], DATA_T B4[128],DATA_T W5[128][128][3][3], DATA_T B5[128], DATA_T O[128][112][112]);


// argv[1] = init_weight.txt , argv[2] = init_bias.txt , argv[3] = init_input.txt
int main(int argc, char *argv[]){
 
  int m, x, y, i, j, k;
  DATA_T temp;
  
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
	
  
  FILE *w_stream = fopen(argv[1], "r");
  if (w_stream == NULL) printf("weight file was not opened");
  FILE *b_stream = fopen(argv[2], "r");
  if (b_stream == NULL) printf("bias file was not opened");
  FILE *i_stream = fopen(argv[3], "r");
  if (i_stream == NULL) printf("i_stream file was not opened");
  
  i_line = NULL;
i_len = 0;
getline(&i_line, &i_len, i_stream);
for (k = 0; k <  3 ; k++) {
	for (x = 0; x < 224 ; x++) {
		for(y = 0; y < 224 ; y++) {
			i_v = strtok_r(i_line, " ", &i_line);
                        I[k][x][y] = atoi(i_v);
			O0_SW[k][x][y] = atoi(i_v);
		}
	}
}

	w_line = b_line = NULL;
w_len = b_len = 0;
getline(&w_line, &w_len, w_stream);
getline(&b_line, &b_len, b_stream);
for (m = 0; m <  64 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 3 ; i++) {
			for (j = 0; j < 3 ; j++) {
                                w_v = strtok_r(w_line," ",&w_line);
				W1[m][k][i][j] = atoi(w_v);       
			}
		}
	}
	b_v = strtok_r(b_line," ",&b_line);
        B1[m] = atoi(b_v);            
}

	w_line = b_line = NULL;
w_len = b_len = 0;
getline(&w_line, &w_len, w_stream);
getline(&b_line, &b_len, b_stream);
for (m = 0; m <  64 ; m++) {
	for (k = 0; k < 64 ; k++) {
		for (i = 0; i < 3 ; i++) {
			for (j = 0; j < 3 ; j++) {
                                w_v = strtok_r(w_line," ",&w_line);
				W2[m][k][i][j] = atoi(w_v);       
			}
		}
	}
	b_v = strtok_r(b_line," ",&b_line);
        B2[m] = atoi(b_v);            
}

	w_line = b_line = NULL;
w_len = b_len = 0;
getline(&w_line, &w_len, w_stream);
getline(&b_line, &b_len, b_stream);
for (m = 0; m <  128 ; m++) {
	for (k = 0; k < 64 ; k++) {
		for (i = 0; i < 3 ; i++) {
			for (j = 0; j < 3 ; j++) {
                                w_v = strtok_r(w_line," ",&w_line);
				W4[m][k][i][j] = atoi(w_v);       
			}
		}
	}
	b_v = strtok_r(b_line," ",&b_line);
        B4[m] = atoi(b_v);            
}

	w_line = b_line = NULL;
w_len = b_len = 0;
getline(&w_line, &w_len, w_stream);
getline(&b_line, &b_len, b_stream);
for (m = 0; m <  128 ; m++) {
	for (k = 0; k < 128 ; k++) {
		for (i = 0; i < 3 ; i++) {
			for (j = 0; j < 3 ; j++) {
                                w_v = strtok_r(w_line," ",&w_line);
				W5[m][k][i][j] = atoi(w_v);       
			}
		}
	}
	b_v = strtok_r(b_line," ",&b_line);
        B5[m] = atoi(b_v);            
}

	
 
  VGG19_top(I,W1,B1,W2,B2,W4,B4,W5,B5,O_HW);
  VGG19_sw(I,W1,B1,W2,B2,W4,B4,W5,B5,O_SW);

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

  fclose(w_stream);
  fclose(b_stream);
  fclose(i_stream);
  return ret_val;
}

