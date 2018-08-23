#include <iostream>
#include <ap_int.h>
#include "hls_stream.h"
#include <stdio.h>
#include <stdlib.h>
using namespace std;

typedef ap_uint<16> DATA_T;

void Ex_model_top(DATA_T I[3][10][10],DATA_T W1[4][3][3][3], DATA_T B1[4],DATA_T W3[8][4][3][3], DATA_T B3[8],DATA_T W8[8][8], DATA_T B8[8],DATA_T W9[3][8], DATA_T B9[3], DATA_T O[3]);
void Ex_model_sw(DATA_T I[3][10][10],DATA_T W1[4][3][3][3], DATA_T B1[4],DATA_T W3[8][4][3][3], DATA_T B3[8],DATA_T W8[8][8], DATA_T B8[8],DATA_T W9[3][8], DATA_T B9[3], DATA_T O[3]);


// argv[1] = init_weight.txt , argv[2] = init_bias.txt , argv[3] = init_input.txt
int main(int argc, char *argv[]){
 
  int m, x, y, i, j, k;
  DATA_T temp;
  
  static DATA_T I[3][10][10];
	static DATA_T O0_SW[3][10][10];
	static DATA_T W1[4][3][3][3];
	static DATA_T B1[4];
	static DATA_T W3[8][4][3][3];
	static DATA_T B3[8];
	static DATA_T W8[8][8];
	static DATA_T B8[8];
	static DATA_T W9[3][8];
	static DATA_T B9[3];
	static DATA_T O_SW[3];
	static DATA_T O_HW[3];
	
  
  FILE *w_stream = fopen(argv[1], "r");
  if (w_stream == NULL) printf("weight file was not opened");
  FILE *b_stream = fopen(argv[2], "r");
  if (b_stream == NULL) printf("bias file was not opened");
  FILE *i_stream = fopen(argv[3], "r");
  if (i_stream == NULL) printf("i_stream file was not opened");
  
  for (k = 0; k <  3 ; k++) {
	for (x = 0; x < 10 ; x++) {
		for(y = 0; y < 10 ; y++) {
			fscanf(i_stream, "%d", &temp);
			I[k][x][y] = temp;
			O0_SW[k][x][y] = temp;
		}
	}
}

	for (m = 0; m <  4 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 3 ; i++) {
			for (j = 0; j < 3 ; j++) {
				fscanf(w_stream, "%d", &temp);
				W1[m][k][i][j] = temp;
			}
		}
	}
	fscanf(b_stream, "%d", &temp);
	B1[m] = temp;      
}

	for (m = 0; m <  8 ; m++) {
	for (k = 0; k < 4 ; k++) {
		for (i = 0; i < 3 ; i++) {
			for (j = 0; j < 3 ; j++) {
				fscanf(w_stream, "%d", &temp);
				W3[m][k][i][j] = temp;
			}
		}
	}
	fscanf(b_stream, "%d", &temp);
	B3[m] = temp;      
}

	for (m = 0; m <  8 ; m++) {
	for (k = 0; k < 8 ; k++) {
		fscanf(w_stream, "%d", &temp);
		W8[m][k] = temp;
	}
	fscanf(b_stream, "%d", &temp);
	B8[m] = temp;      
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 8 ; k++) {
		fscanf(w_stream, "%d", &temp);
		W9[m][k] = temp;
	}
	fscanf(b_stream, "%d", &temp);
	B9[m] = temp;      
}

	
 
  Ex_model_top(I,W1,B1,W3,B3,W8,B8,W9,B9,O_HW);
  Ex_model_sw(I,W1,B1,W3,B3,W8,B8,W9,B9,O_SW);

    int err_cnt = 0;
    for (m=0; m<3; m++) {
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

