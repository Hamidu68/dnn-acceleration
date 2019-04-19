#include <iostream>
#include <ap_int.h>
#include "hls_stream.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
using namespace std;


typedef int DATA_IN;
typedef int DATA_T;

void vgg19_top(DATA_T I[3][224][224],DATA_T W1[64][3][3][3],DATA_T B1[64], DATA_T W2[64][64][3][3],DATA_T B2[64], DATA_T W4[128][64][3][3],DATA_T B4[128], DATA_T W5[128][128][3][3],DATA_T B5[128], DATA_T O[128][56][56]);
void vgg19_sw(DATA_T I[3][224][224],DATA_T W1[64][3][3][3],DATA_T B1[64], DATA_T W2[64][64][3][3],DATA_T B2[64], DATA_T W4[128][64][3][3],DATA_T B4[128], DATA_T W5[128][128][3][3],DATA_T B5[128], DATA_T O[128][56][56]);

int main() {

	int m, x, y, i, j, k;
	DATA_T temp;
	int trash;

	static DATA_T I[3][224][224];
	static DATA_T W1[64][3][3][3];
	static DATA_T B1[64];
	static DATA_T W2[64][64][3][3];
	static DATA_T B2[64];
	static DATA_T W4[128][64][3][3];
	static DATA_T B4[128];
	static DATA_T W5[128][128][3][3];
	static DATA_T B5[128];
	static DATA_T O0_SW[3][224][224];
	static DATA_T O_SW[128][56][56];
	static DATA_T O_HW[128][56][56];
	

	FILE *w_stream = fopen("init_Weight.bin", "r");
	if (w_stream == NULL) printf("weight file was not opened");
	FILE *i_stream = fopen("init_Input.bin", "r");
	if (i_stream == NULL) printf("input file was not opened");

 	for (k = 0; k <  224 ; k++) {
	for (x = 0; x < 224 ; x++) {
		for(y = 0; y < 3 ; y++) {
			fread(&trash, sizeof(int), 1, i_stream);
            I[y][k][x] = (DATA_T) trash;
			O0_SW[y][k][x] = (DATA_T) trash;
		}
	}
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 3 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W1[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B1[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W2[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B2[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W4[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B4[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W5[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B5[m] = (DATA_T) trash;
}

	

	vgg19_top(I, W1, B1, W2, B2, W4, B4, W5, B5,  O_HW);
	vgg19_sw(I, W1, B1, W2, B2, W4, B4, W5, B5,  O_SW);

	int err_cnt =0;
	for(m=0; m<128; m++){
	   for(x=0;x<56; x++){
	     for(y=0;y<56; y++){
	       cout<<"SW: O["<<m<<"]["<<"x"<<"]["<<y<<"] = "<<O_SW[m][x][y];
	       cout<<"HW: O["<<m<<"]["<<"x"<<"]["<<y<<"] = "<<O_HW[m][x][y];
	       if (O_HW[m][x][y] != O_SW[m][x][y]){
	         err_cnt++;
	       }
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
	fclose(i_stream);
	return ret_val;
}