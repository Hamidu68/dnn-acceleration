#include <iostream>
#include <ap_int.h>
#include "hls_stream.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

using namespace std;

typedef int DATA_T;

void vgg19_top(DATA_T I[3][16][16],DATA_T W1[8][3][3][3], DATA_T B1[8],DATA_T W2[8][8][3][3], DATA_T B2[8],DATA_T W4[16][8][3][3], DATA_T B4[16], DATA_T O[16][8][8]);
void vgg19_sw(DATA_T I[3][16][16],DATA_T W1[8][3][3][3], DATA_T B1[8],DATA_T W2[8][8][3][3], DATA_T B2[8],DATA_T W4[16][8][3][3], DATA_T B4[16], DATA_T O[16][8][8]);


int main(){

  DATA_T temp; 
  int m, x, y, i, j, k;
  int trash;
  
  static DATA_T I[3][16][16];
	static DATA_T O0_SW[3][16][16];
	static DATA_T W1[8][3][3][3];
	static DATA_T B1[8];
	static DATA_T W2[8][8][3][3];
	static DATA_T B2[8];
	static DATA_T W4[16][8][3][3];
	static DATA_T B4[16];
	static DATA_T O_SW[16][8][8];
	static DATA_T O_HW[16][8][8];
	
  
  FILE *w_stream = fopen("init_Weight.bin", "r");
  if (w_stream == NULL) printf("weight file was not opened");
  FILE *b_stream = fopen("init_Bias.bin", "r");
  if (b_stream == NULL) printf("bias file was not opened");
  FILE *i_stream = fopen("init_Input.bin", "r");
  if (i_stream == NULL) printf("input file was not opened");
  
  for (k = 0; k <  3 ; k++) {
	for (x = 0; x < 16 ; x++) {
		for(y = 0; y < 16 ; y++) {
			fread(&trash, sizeof(int), 1, i_stream);
            I[k][x][y] = trash;
			O0_SW[k][x][y] = trash;
		}
	}
}

	for (m = 0; m <  8 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 3 ; i++) {
			for (j = 0; j < 3 ; j++) {
                fread(&W1[m][k][i][j], sizeof(int), 1, w_stream);
			}
		}
	}
    fread(&B1[m], sizeof(int), 1, b_stream);
}

	for (m = 0; m <  8 ; m++) {
	for (k = 0; k < 8 ; k++) {
		for (i = 0; i < 3 ; i++) {
			for (j = 0; j < 3 ; j++) {
                fread(&W2[m][k][i][j], sizeof(int), 1, w_stream);
			}
		}
	}
    fread(&B2[m], sizeof(int), 1, b_stream);
}

	for (m = 0; m <  16 ; m++) {
	for (k = 0; k < 8 ; k++) {
		for (i = 0; i < 3 ; i++) {
			for (j = 0; j < 3 ; j++) {
                fread(&W4[m][k][i][j], sizeof(int), 1, w_stream);
			}
		}
	}
    fread(&B4[m], sizeof(int), 1, b_stream);
}

	
 
  vgg19_top(I,W1,B1,W2,B2,W4,B4,O_HW);
  vgg19_sw(I,W1,B1,W2,B2,W4,B4,O_SW);

   int err_cnt = 0;
   for (m=0; m<16; m++) {
       for (x=0; x<8; x++) {
          for (y=0; y<8; y++) {
	      cout<<"SW: O["<<m<<"]["<<x<<"]["<<y<<"] = "<<O_SW[m][x][y];
       	      cout<<"HW: O["<<m<<"]["<<x<<"]["<<y<<"] = "<< O_HW[m][x][y]<<endl;
              if (O_HW[m][x][y] != O_SW[m][x][y]) {
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

