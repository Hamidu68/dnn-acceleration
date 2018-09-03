#include <iostream>
#include <ap_int.h>
#include "hls_stream.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace std;

typedef ap_uint<16> DATA_T;

void vgg19_top(DATA_T I[3][32][32],DATA_T W1[16][3][3][3], DATA_T B1[16],DATA_T W2[16][16][3][3], DATA_T B2[16],DATA_T W5[10][4096], DATA_T B5[10],DATA_T W6[10][10], DATA_T B6[10], DATA_T O[10]);
void vgg19_sw(DATA_T I[3][32][32],DATA_T W1[16][3][3][3], DATA_T B1[16],DATA_T W2[16][16][3][3], DATA_T B2[16],DATA_T W5[10][4096], DATA_T B5[10],DATA_T W6[10][10], DATA_T B6[10], DATA_T O[10]);


int main(){

  DATA_T temp; 
  int m, x, y, i, j, k;
  int trash;
  
  static DATA_T I[3][32][32];
	static DATA_T O0_SW[3][32][32];
	static DATA_T W1[16][3][3][3];
	static DATA_T B1[16];
	static DATA_T W2[16][16][3][3];
	static DATA_T B2[16];
	static DATA_T W5[10][4096];
	static DATA_T B5[10];
	static DATA_T W6[10][10];
	static DATA_T B6[10];
	static DATA_T O_SW[10];
	static DATA_T O_HW[10];
	
  
  FILE *w_stream = fopen("init_weight.txt", "rb");
  if (w_stream == NULL) printf("weight file was not opened");
  FILE *b_stream = fopen("init_bias", "rb");
  if (b_stream == NULL) printf("bias file was not opened");
  FILE *i_stream = fopen("init_input.txt", "rb");
  if (i_stream == NULL) printf("input file was not opened");
  
  for (k = 0; k <  3 ; k++) {
	for (x = 0; x < 32 ; x++) {
		for(y = 0; y < 32 ; y++) {
			fread(&trash, sizeof(int), 1, i_stream);
                        I[k][x][y] = trash;
			O0_SW[k][x][y] = trash;
		}
	}
}

	for (m = 0; m <  16 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 3 ; i++) {
			for (j = 0; j < 3 ; j++) {
                                fread(&W1[m][k][i][j], sizeof(int), 1, w_stream);       
			}
		}
	}
        fread(&B1[m], sizeof(int), 1, b_stream);
}

	for (m = 0; m <  16 ; m++) {
	for (k = 0; k < 16 ; k++) {
		for (i = 0; i < 3 ; i++) {
			for (j = 0; j < 3 ; j++) {
                                fread(&W2[m][k][i][j], sizeof(int), 1, w_stream);       
			}
		}
	}
        fread(&B2[m], sizeof(int), 1, b_stream);
}

	for (m = 0; m <  10 ; m++) {
	for (k = 0; k < 4096 ; k++) {
		fread(&W5[m][k], sizeof(int), 1, w_stream);
	}   
        fread(&B5[m], sizeof(int), 1, b_stream);
}

	for (m = 0; m <  10 ; m++) {
	for (k = 0; k < 10 ; k++) {
		fread(&W6[m][k], sizeof(int), 1, w_stream);
	}   
        fread(&B6[m], sizeof(int), 1, b_stream);
}

	
 
  vgg19_top(I,W1,B1,W2,B2,W5,B5,W6,B6,O_HW);
  vgg19_sw(I,W1,B1,W2,B2,W5,B5,W6,B6,O_SW);

    int err_cnt = 0;
    for (m=0; m<10; m++) {
        if (O_HW[m] != O_SW[m]) {
            printf("SW: O[%d] = %d", m, O_SW[m]);
            printf("HW: O[%d] = %d", m, O_HW[m]);
            err_cnt++;}
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

