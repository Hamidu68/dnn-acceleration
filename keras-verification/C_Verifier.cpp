#include <stdio.h>
#include<iostream>
#include <stdlib.h>
#include<string>
#include<string.h>
#include<math.h>
using namespace std;

typedef float DATA_T;

void SW_block1_conv1(DATA_T I[3][32][32], DATA_T O[16][32][32], DATA_T B[16], DATA_T W[16][3][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(32 - 1) - 32 + 3)/2; 
	for (m = 0; m<16; m++) {
		for (x = 0; x<32; x++) {
			for (y = 0; y<32; y++) {
				ofm = B[m];
				for (k = 0; k<3; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 32 + p && y + j < 32 + p && x + i -p >= 0 && y + j -p >= 0) { 
                                    ifm = I[k][x*1 + i - p][y*1 + j -p];
							}
							else {
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

void SW_block1_conv2(DATA_T I[16][32][32], DATA_T O[16][32][32], DATA_T B[16], DATA_T W[16][16][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(32 - 1) - 32 + 3)/2; 
	for (m = 0; m<16; m++) {
		for (x = 0; x<32; x++) {
			for (y = 0; y<32; y++) {
				ofm = B[m];
				for (k = 0; k<16; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 32 + p && y + j < 32 + p && x + i -p >= 0 && y + j -p >= 0) { 
                                    ifm = I[k][x*1 + i - p][y*1 + j -p];
							}
							else {
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

void SW_block1_pool(DATA_T I[16][32][32], DATA_T O[16][16][16])
{ 
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<16; m++) {
		for (x = 0; x<16; x++) {
			for (y = 0; y<16; y++) {
				max = I[m][x*2][y*2];
				for (i = 0; i<2; i++) {
					for (j = 0; j<2; j++) {
						if (I[m][x*2 + i][y*2 + j] > max) {
							max = I[m][x*2 + i][y*2 + j];
						}
					}
				}
				O[m][x][y] = max;
			}
		}
	}
}
void SW_flatten(DATA_T I[16][16][16], DATA_T O[4096]){
	int i, j, x, y;
	i = 0;
	 for(j=0; j< 16; j++)
		for(x=0; x<16;x++)
			for (y = 0; y < 16; y++) {
				O[i] = I[j][x][y];
				i++;
			}
}
void SW_fc1(DATA_T I[4096], DATA_T W[10][4096], DATA_T B[10], DATA_T O[10])
{
    //Dense
	int m, c;
	for(m=0; m<10; m++){
        O[m] = B[m];
		for (c = 0; c < 4096; c++){
            O[m] += W[m][c] * I[c];
        }
        if (O[m] < 0) //Relu
            O[m] = 0;
     }
}
void SW_predictions(DATA_T I[10], DATA_T W[10][10], DATA_T B[10] , DATA_T O[10])
{
    //Dense
	int m, c;    
	DATA_T denom = 0;
	for(m=0; m<10; m++){
        O[m] = B[m];
		for (c = 0; c < 10; c++){
			O[m] += W[m][c] * I[c];
         }
        denom += O[m]; //Sum of Output
    }
    //Softmax
    float max = 0;
	for (m = 0; m < 10; m++)
		if(O[m]>max)
          max = O[m];
        
     for (m = 0; m < 10; m++){
		if(O[m] != max)
          O[m] = 0;
        else
          O[m] = 1;}

/*
	for (m = 0; m < 10; m++)
		O[m] = O[m] / denom; 
*/

}


//argv[1] = init_weight.txt , argv[2] = init_bias.txt , argv[3] = init_input.txt
int main(int argc, char *argv[]){
 
  DATA_T temp;
  int m, x, y, i, j, k, l;
  int trash;

  static DATA_T I[3][32][32];
	static DATA_T W1[16][3][3][3];
	static DATA_T B1[16];
	static DATA_T W2[16][16][3][3];
	static DATA_T B2[16];
	static DATA_T B5[10];
	static DATA_T W5[10][4096];
	static DATA_T B6[10];
	static DATA_T W6[10][10];
	
 
  static DATA_T O0_SW[3][32][32];
	static DATA_T O1_SW[16][32][32];
	static DATA_T O2_SW[16][32][32];
	static DATA_T O3_SW[16][16][16];
	static DATA_T O4_SW[4096];
	static DATA_T O5_SW[10];
	static DATA_T O6_SW[10];
	 

  FILE *w_stream = fopen(argv[1], "rb");
  if (w_stream == NULL) printf("weight file was not opened");
  FILE *b_stream = fopen(argv[2], "rb");
  if (b_stream == NULL) printf("bias file was not opened");
  FILE *i_stream = fopen(argv[3], "rb");
  if (i_stream == NULL) printf("input file was not opened");
  FILE *o_stream = fopen("Output/C_output.txt", "w");
  if (o_stream == NULL) printf("Output file was not opened");
 
  printf("[C_verifier.cpp]Start Initialzation\n\n");
  for (k = 0; k <  3 ; k++) {
	for (x = 0; x < 32 ; x++) {
		for(y = 0; y < 32 ; y++) {
			fread(&trash, sizeof(int), 1, i_stream);
                        I[k][x][y] = (float) trash;
			O0_SW[k][x][y] = (float) trash;
		}
	}
}

	for (m = 0; m <  16 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 3 ; i++) {
			for (j = 0; j < 3 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                                W1[m][k][i][j] = (float) trash;       
			}
		}
	}
	fread(&trash, sizeof(int), 1, b_stream);
        B1[m] = (float) trash;            
}

	for (m = 0; m <  16 ; m++) {
	for (k = 0; k < 16 ; k++) {
		for (i = 0; i < 3 ; i++) {
			for (j = 0; j < 3 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                                W2[m][k][i][j] = (float) trash;       
			}
		}
	}
	fread(&trash, sizeof(int), 1, b_stream);
        B2[m] = (float) trash;            
}

	for (m = 0; m <  10 ; m++) {
	for (k = 0; k < 4096 ; k++) {
		fread(&trash, sizeof(int), 1, w_stream);
                W5[m][k] = (float) trash;       
	}
	fread(&trash, sizeof(int), 1, b_stream);
        B5[m] = (float) trash;         
}

	for (m = 0; m <  10 ; m++) {
	for (k = 0; k < 10 ; k++) {
		fread(&trash, sizeof(int), 1, w_stream);
                W6[m][k] = (float) trash;       
	}
	fread(&trash, sizeof(int), 1, b_stream);
        B6[m] = (float) trash;         
}

	
  printf("[C_verifier.cpp]Finish Initialization\n\n");
  
  printf("[C_verifier.cpp]InputLayer\n\n");
	printf("[C_verifier.cpp]Calculate Conv2D1\n\n");
	SW_block1_conv1(O0_SW,O1_SW,B1,W1);
	printf("[C_verifier.cpp]Calculate Conv2D2\n\n");
	SW_block1_conv2(O1_SW,O2_SW,B2,W2);
	printf("[C_verifier.cpp]Calculate MaxPooling2D3\n\n");
	SW_block1_pool(O2_SW,O3_SW);
	printf("[C_verifier.py]Calculate Flatten4\n\n");
	SW_flatten(O3_SW,O4_SW);
	printf("[C_verifier.cpp]Calculate Dense5\n\n");
	SW_fc1(O4_SW,W5,B5,O5_SW);
	printf("[C_verifier.cpp]Calculate Dense6\n\n");
	SW_predictions(O5_SW,W6,B6,O6_SW);
	

  printf("[C_verifier.cpp]Print Result\n");
  
  fprintf(o_stream,"%s","InputLayer : [[");
for (k = 0; k <  3 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 32 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%2d ",int(O0_SW[k][x][y]));
		}
		if(x != 32 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 3 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Convolution2D : [[");
for (k = 0; k <  16 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 32 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%2d ",int(O1_SW[k][x][y]));
		}
		if(x != 32 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 16 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Convolution2D : [[");
for (k = 0; k <  16 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 32 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%2d ",int(O2_SW[k][x][y]));
		}
		if(x != 32 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 16 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","MaxPooling2D : [[");
for (k = 0; k <  16 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 16 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 16 ; y++) {
			fprintf(o_stream,"%2d ",int(O3_SW[k][x][y]));
		}
		if(x != 16 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 16 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Flatten : [[");
for (k = 0; k <  4096 ; k++) {

	fprintf(o_stream,"%2d ",int(O4_SW[k]));
}
fprintf(o_stream,"%s","]]\n\n");


fprintf(o_stream,"%s","Dense : [[");
for (k = 0; k <  10 ; k++) {

	fprintf(o_stream,"%2d ",int(O5_SW[k]));
}
fprintf(o_stream,"%s","]]\n\n");


fprintf(o_stream,"%s","Dense : [[");
for (k = 0; k <  10 ; k++) {

	fprintf(o_stream,"%2d ",int(O6_SW[k]));
}
fprintf(o_stream,"%s","]]\n\n");



    
  fclose(w_stream);
  fclose(b_stream);
  fclose(i_stream);
  fclose(o_stream);
  return 0;
}

