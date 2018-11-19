#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <math.h>
    
using namespace std;

typedef float DATA_T;

void SW_conv1_pad(DATA_T I[3][224][224],DATA_T O[3][230][230]) {
	int m, x, y, i, j;
	for (m = 0; m < 3; m++) {
		for (x = 0; x < 230; x++) {
			for (y = 0; y < 230; y++) {
				O[m][x][y] = 0;
			}
		}
	}
	for (m = 0; m < 3; m++) {
		for (x = 0; x < 224; x++) {
			for (y = 0; y < 224; y++) {
				O[m][x+ 3][y+ 3] = I[m][x][y];
			}
		}
	}
}
void SW_conv1(DATA_T I[3][230][230], DATA_T O[64][112][112], DATA_T W[64][3][7][7], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<64; m++) {
		for (x = 0; x<112; x++) {
			for (y = 0; y<112; y++) {
				ofm = 0;
				for (k = 0; k<3; k++) {
					for (i = 0; i<7; i++) {
						for (j = 0; j<7; j++) {
							if (x + i <= 230 && y + j <= 230) {
								ifm = I[k][x*2 + i][y*2 + j];
							}

							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				ofm += B[m];
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_bn_conv1(DATA_T I[64][112][112], DATA_T O[64][112][112], DATA_T W[4][64]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 64; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 112; x++){
			for (y = 0; y < 112; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_1(DATA_T I[64][112][112], DATA_T O[64][112][112])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<64; m++) {
		for (x = 0; x<112; x++) {
			for (y = 0; y<112; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_max_pooling2d_1(DATA_T I[64][112][112], DATA_T O[64][55][55])
{
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<64; m++) {
		for (x = 0; x<55; x++) {
			for (y = 0; y<55; y++) {
				max = I[m][x*2][y*2];
				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {
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
void SW_res2a_branch2a(DATA_T I[64][55][55], DATA_T O[64][55][55], DATA_T W[64][64][1][1], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<64; m++) {
		for (x = 0; x<55; x++) {
			for (y = 0; y<55; y++) {
				ofm = 0;
				for (k = 0; k<64; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 55 && y + j <= 55) {
								ifm = I[k][x*1 + i][y*1 + j];
							}

							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				ofm += B[m];
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_bn2a_branch2a(DATA_T I[64][55][55], DATA_T O[64][55][55], DATA_T W[4][64]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 64; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 55; x++){
			for (y = 0; y < 55; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_2(DATA_T I[64][55][55], DATA_T O[64][55][55])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<64; m++) {
		for (x = 0; x<55; x++) {
			for (y = 0; y<55; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res2a_branch2b(DATA_T I[64][55][55], DATA_T O[64][55][55], DATA_T W[64][64][3][3], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(55 - 1) - 55 + 3)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<55; x++) {
			for (y = 0; y<55; y++) {
				ofm = 0;
				for (k = 0; k<64; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 55 + p && y + j < 55 + p && x + i -p >= 0 && y + j -p >= 0) {
                                    ifm = I[k][x*1 + i - p][y*1 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				ofm += B[m];
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_bn2a_branch2b(DATA_T I[64][55][55], DATA_T O[64][55][55], DATA_T W[4][64]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 64; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 55; x++){
			for (y = 0; y < 55; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_3(DATA_T I[64][55][55], DATA_T O[64][55][55])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<64; m++) {
		for (x = 0; x<55; x++) {
			for (y = 0; y<55; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res2a_branch2c(DATA_T I[64][55][55], DATA_T O[256][55][55], DATA_T W[256][64][1][1], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<55; x++) {
			for (y = 0; y<55; y++) {
				ofm = 0;
				for (k = 0; k<64; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 55 && y + j <= 55) {
								ifm = I[k][x*1 + i][y*1 + j];
							}

							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				ofm += B[m];
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_res2a_branch1(DATA_T I[64][55][55], DATA_T O[256][55][55], DATA_T W[256][64][1][1], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<55; x++) {
			for (y = 0; y<55; y++) {
				ofm = 0;
				for (k = 0; k<64; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 55 && y + j <= 55) {
								ifm = I[k][x*1 + i][y*1 + j];
							}

							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				ofm += B[m];
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_bn2a_branch2c(DATA_T I[256][55][55], DATA_T O[256][55][55], DATA_T W[4][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 256; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 55; x++){
			for (y = 0; y < 55; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_bn2a_branch1(DATA_T I[256][55][55], DATA_T O[256][55][55], DATA_T W[4][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 256; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 55; x++){
			for (y = 0; y < 55; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_add_1(DATA_T I1[256][55][55], DATA_T I2[256][55][55], DATA_T O[256][55][55]) {
	int m, x, y;
	for (m = 0; m< 256; m++) {
		for (x = 0; x < 55; x++) {
			for(y = 0; y < 55; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}


//argv[1] = init_weight.txt, argv[3] = init_input.txt
int main(int argc, char *argv[]){

    DATA_T temp;
    int m, x, y, i, j, k, l;
    int trash;

    static DATA_T I[3][224][224];
	static DATA_T W2[64][3][7][7];
	static DATA_T B2[64];
	static DATA_T W3[4][64];
	static DATA_T W6[64][64][1][1];
	static DATA_T B6[64];
	static DATA_T W7[4][64];
	static DATA_T W9[64][64][3][3];
	static DATA_T B9[64];
	static DATA_T W10[4][64];
	static DATA_T W12[256][64][1][1];
	static DATA_T B12[256];
	static DATA_T W13[256][64][1][1];
	static DATA_T B13[256];
	static DATA_T W14[4][256];
	static DATA_T W15[4][256];
	

    static DATA_T O0_SW[3][224][224];
	static DATA_T O1_SW[3][230][230];
	static DATA_T O2_SW[64][112][112];
	static DATA_T O3_SW[64][112][112];
	static DATA_T O4_SW[64][112][112];
	static DATA_T O5_SW[64][55][55];
	static DATA_T O6_SW[64][55][55];
	static DATA_T O7_SW[64][55][55];
	static DATA_T O8_SW[64][55][55];
	static DATA_T O9_SW[64][55][55];
	static DATA_T O10_SW[64][55][55];
	static DATA_T O11_SW[64][55][55];
	static DATA_T O12_SW[256][55][55];
	static DATA_T O13_SW[256][55][55];
	static DATA_T O14_SW[256][55][55];
	static DATA_T O15_SW[256][55][55];
	static DATA_T O16_SW[256][55][55];
	

    FILE *w_stream = fopen(argv[1], "rb");
    if (w_stream == NULL) printf("weight file was not opened");
    FILE *i_stream = fopen(argv[2], "rb");
    if (i_stream == NULL) printf("input file was not opened");
    FILE *o_stream = fopen("../../cpp_generator/resnet50/Output/C_output.txt", "w");
    if (o_stream == NULL) printf("Output file was not opened");
    FILE *c_num = fopen("../../cpp_generator/resnet50/Output/c_output_num.txt", "w");
    if (c_num == NULL) printf("Output file was not opened");

    printf("[C_verifier.cpp]Start Initialzation");
    for (k = 0; k <  224 ; k++) {
	for (x = 0; x < 224 ; x++) {
		for(y = 0; y < 3 ; y++) {
			fread(&trash, sizeof(int), 1, i_stream);
            I[y][k][x] = (DATA_T) trash;
			O0_SW[y][k][x] = (DATA_T) trash;
		}
	}
}

	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 3 ; i++) {
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


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W3[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W6[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B6[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W7[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W9[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B9[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W10[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W12[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B12[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W13[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B13[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W14[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W15[x][y] = (DATA_T) trash;
    }
}
	
    printf("[C_verifier.cpp]Finish Initialization");

    printf("[C_verifier.cpp]InputLayer\n\n");
	printf("[C_verifier.cpp]Calculate ZeroPadding2D1\n\n");
	SW_conv1_pad(O0_SW,O1_SW);
	printf("[C_verifier.cpp]Calculate Conv2D2\n\n");
	SW_conv1(O1_SW,O2_SW,W2,B2);
	printf("[C_verifier.cpp]Calculate BatchNormalization3\n\n");
	SW_bn_conv1(O2_SW,O3_SW, W3);
	printf("[C_verifier.cpp]Calculate Activation(Relu)4\n\n");
	SW_activation_1(O3_SW,O4_SW);
	printf("[C_verifier.cpp]Calculate MaxPooling2D5\n\n");
	SW_max_pooling2d_1(O4_SW,O5_SW);
	printf("[C_verifier.cpp]Calculate Conv2D6\n\n");
	SW_res2a_branch2a(O5_SW,O6_SW,W6,B6);
	printf("[C_verifier.cpp]Calculate BatchNormalization7\n\n");
	SW_bn2a_branch2a(O6_SW,O7_SW, W7);
	printf("[C_verifier.cpp]Calculate Activation(Relu)8\n\n");
	SW_activation_2(O7_SW,O8_SW);
	printf("[C_verifier.cpp]Calculate Conv2D9\n\n");
	SW_res2a_branch2b(O8_SW,O9_SW,W9,B9);
	printf("[C_verifier.cpp]Calculate BatchNormalization10\n\n");
	SW_bn2a_branch2b(O9_SW,O10_SW, W10);
	printf("[C_verifier.cpp]Calculate Activation(Relu)11\n\n");
	SW_activation_3(O10_SW,O11_SW);
	printf("[C_verifier.cpp]Calculate Conv2D12\n\n");
	SW_res2a_branch2c(O11_SW,O12_SW,W12,B12);
	printf("[C_verifier.cpp]Calculate Conv2D13\n\n");
	SW_res2a_branch1(O5_SW,O13_SW,W13,B13);
	printf("[C_verifier.cpp]Calculate BatchNormalization14\n\n");
	SW_bn2a_branch2c(O12_SW,O14_SW, W14);
	printf("[C_verifier.cpp]Calculate BatchNormalization15\n\n");
	SW_bn2a_branch1(O13_SW,O15_SW, W15);
	printf("[C_verifier.cpp]Calculate Add16\n\n");
	SW_add_1(O14_SW,O15_SW,O16_SW);
	

    printf("[C_verifier.cpp]Print Result");


    fprintf(o_stream,"%s","InputLayer : [[");
for (k = 0; k < 224 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 224 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 3 ; y++) {
			fprintf(o_stream,"%.6f ",O0_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O0_SW[y][k][x]);
		}
		if(x != 224 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 224 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","ZeroPadding2D : [[");
for (k = 0; k < 230 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 230 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 3 ; y++) {
			fprintf(o_stream,"%.6f ",O1_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O1_SW[y][k][x]);
		}
		if(x != 230 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 230 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 112 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 112 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O2_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O2_SW[y][k][x]);
		}
		if(x != 112 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 112 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 112 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 112 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O3_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O3_SW[y][k][x]);
		}
		if(x != 112 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 112 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 112 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 112 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O4_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O4_SW[y][k][x]);
		}
		if(x != 112 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 112 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","MaxPooling2D : [[");
for (k = 0; k < 55 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 55 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O5_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O5_SW[y][k][x]);
		}
		if(x != 55 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 55 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 55 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 55 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O6_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O6_SW[y][k][x]);
		}
		if(x != 55 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 55 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 55 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 55 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O7_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O7_SW[y][k][x]);
		}
		if(x != 55 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 55 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 55 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 55 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O8_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O8_SW[y][k][x]);
		}
		if(x != 55 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 55 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 55 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 55 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O9_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O9_SW[y][k][x]);
		}
		if(x != 55 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 55 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 55 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 55 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O10_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O10_SW[y][k][x]);
		}
		if(x != 55 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 55 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 55 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 55 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O11_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O11_SW[y][k][x]);
		}
		if(x != 55 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 55 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 55 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 55 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O12_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O12_SW[y][k][x]);
		}
		if(x != 55 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 55 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 55 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 55 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O13_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O13_SW[y][k][x]);
		}
		if(x != 55 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 55 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 55 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 55 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O14_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O14_SW[y][k][x]);
		}
		if(x != 55 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 55 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 55 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 55 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O15_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O15_SW[y][k][x]);
		}
		if(x != 55 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 55 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 55 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 55 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O16_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O16_SW[y][k][x]);
		}
		if(x != 55 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 55 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");





    fclose(w_stream);
    fclose(i_stream);
    fclose(o_stream);
    fclose(c_num);

    return 0;
}
