#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <math.h>
    
using namespace std;

typedef float DATA_T;

void SW_conv1_pad(DATA_T I[3][224][224],DATA_T O[3][225][225]) {
	int m, x, y, i, j;
	for (m = 0; m < 3; m++) {
		for (x = 0; x < 225; x++) {
			for (y = 0; y < 225; y++) {
				O[m][x][y] = 0;
			}
		}
	}
	for (m = 0; m < 3; m++) {
		for (x = 0; x < 224; x++) {
			for (y = 0; y < 224; y++) {
				O[m][x+ 0][y+ 0] = I[m][x][y];
			}
		}
	}
}
void SW_conv1(DATA_T I[3][225][225], DATA_T O[32][112][112], DATA_T W[32][3][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<32; m++) {
		for (x = 0; x<112; x++) {
			for (y = 0; y<112; y++) {
				ofm = 0;
				for (k = 0; k<3; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i <= 225 && y + j <= 225) {
								ifm = I[k][x*2 + i][y*2 + j];
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

void SW_conv1_bn(DATA_T I[32][112][112], DATA_T O[32][112][112], DATA_T W[4][32]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 32; m++){
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
void SW_conv1_relu(DATA_T I[32][112][112], DATA_T O[32][112][112])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<32; m++) {
		for (x = 0; x<112; x++) {
			for (y = 0; y<112; y++) {
				ifm = I[m][x][y];
				if (ifm >= mv)
					O[m][x][y] = mv;
				else if(ifm>=0.0 && ifm<mv)
					O[m][x][y] = ifm;
				else
				    O[m][x][y] = 0.0*(ifm-0.0);
			}
		}
	}
}
void SW_conv_dw_1(DATA_T I[32][112][112], DATA_T O[32][112][112], DATA_T W[32][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(112 - 1) - 112 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<112; x++) {
			for (y = 0; y<112; y++) {
				ofm = 0;

				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {
						if (x + i < 112 + p && y + j < 112 + p && x + i -p >= 0 && y + j -p >= 0) {
                            ifm = I[m][x*1 + i - p][y*1 + j -p];
						}
						else {
							ifm = 0; // zero padding
						}
						ofm = ofm + ifm * W[m][i][j];
					}
				}
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_conv_dw_1_bn(DATA_T I[32][112][112], DATA_T O[32][112][112], DATA_T W[4][32]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 32; m++){
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
void SW_conv_dw_1_relu(DATA_T I[32][112][112], DATA_T O[32][112][112])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<32; m++) {
		for (x = 0; x<112; x++) {
			for (y = 0; y<112; y++) {
				ifm = I[m][x][y];
				if (ifm >= mv)
					O[m][x][y] = mv;
				else if(ifm>=0.0 && ifm<mv)
					O[m][x][y] = ifm;
				else
				    O[m][x][y] = 0.0*(ifm-0.0);
			}
		}
	}
}
void SW_conv_pw_1(DATA_T I[32][112][112], DATA_T O[64][112][112], DATA_T W[64][32][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(112 - 1) - 112 + 1)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<112; x++) {
			for (y = 0; y<112; y++) {
				ofm = 0;
				for (k = 0; k<32; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 112 + p && y + j < 112 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_conv_pw_1_bn(DATA_T I[64][112][112], DATA_T O[64][112][112], DATA_T W[4][64]) {
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
void SW_conv_pw_1_relu(DATA_T I[64][112][112], DATA_T O[64][112][112])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<64; m++) {
		for (x = 0; x<112; x++) {
			for (y = 0; y<112; y++) {
				ifm = I[m][x][y];
				if (ifm >= mv)
					O[m][x][y] = mv;
				else if(ifm>=0.0 && ifm<mv)
					O[m][x][y] = ifm;
				else
				    O[m][x][y] = 0.0*(ifm-0.0);
			}
		}
	}
}
void SW_conv_pad_2(DATA_T I[64][112][112],DATA_T O[64][113][113]) {
	int m, x, y, i, j;
	for (m = 0; m < 64; m++) {
		for (x = 0; x < 113; x++) {
			for (y = 0; y < 113; y++) {
				O[m][x][y] = 0;
			}
		}
	}
	for (m = 0; m < 64; m++) {
		for (x = 0; x < 112; x++) {
			for (y = 0; y < 112; y++) {
				O[m][x+ 0][y+ 0] = I[m][x][y];
			}
		}
	}
}
void SW_conv_dw_2(DATA_T I[64][113][113], DATA_T O[64][56][56], DATA_T W[64][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<64; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				ofm = 0;
				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {
						if (x + i <= 113 && y + j <= 113) {
							ifm = I[m][x*2 + i][y*2 + j];
						}

						ofm = ofm + ifm * W[m][i][j];
					}
				}
				O[m][x][y] = ofm;
			}
		}
	}
}
void SW_conv_dw_2_bn(DATA_T I[64][56][56], DATA_T O[64][56][56], DATA_T W[4][64]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 64; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 56; x++){
			for (y = 0; y < 56; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_conv_dw_2_relu(DATA_T I[64][56][56], DATA_T O[64][56][56])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<64; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				ifm = I[m][x][y];
				if (ifm >= mv)
					O[m][x][y] = mv;
				else if(ifm>=0.0 && ifm<mv)
					O[m][x][y] = ifm;
				else
				    O[m][x][y] = 0.0*(ifm-0.0);
			}
		}
	}
}
void SW_conv_pw_2(DATA_T I[64][56][56], DATA_T O[128][56][56], DATA_T W[128][64][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(56 - 1) - 56 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				ofm = 0;
				for (k = 0; k<64; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 56 + p && y + j < 56 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_conv_pw_2_bn(DATA_T I[128][56][56], DATA_T O[128][56][56], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 128; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 56; x++){
			for (y = 0; y < 56; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_conv_pw_2_relu(DATA_T I[128][56][56], DATA_T O[128][56][56])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<128; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				ifm = I[m][x][y];
				if (ifm >= mv)
					O[m][x][y] = mv;
				else if(ifm>=0.0 && ifm<mv)
					O[m][x][y] = ifm;
				else
				    O[m][x][y] = 0.0*(ifm-0.0);
			}
		}
	}
}
void SW_conv_dw_3(DATA_T I[128][56][56], DATA_T O[128][56][56], DATA_T W[128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(56 - 1) - 56 + 3)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				ofm = 0;

				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {
						if (x + i < 56 + p && y + j < 56 + p && x + i -p >= 0 && y + j -p >= 0) {
                            ifm = I[m][x*1 + i - p][y*1 + j -p];
						}
						else {
							ifm = 0; // zero padding
						}
						ofm = ofm + ifm * W[m][i][j];
					}
				}
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_conv_dw_3_bn(DATA_T I[128][56][56], DATA_T O[128][56][56], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 128; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 56; x++){
			for (y = 0; y < 56; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
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
	static DATA_T W2[32][3][3][3];
	static DATA_T W3[4][32];
	static DATA_T W5[32][3][3];
	static DATA_T W6[4][32];
	static DATA_T W8[64][32][1][1];
	static DATA_T W9[4][64];
	static DATA_T W12[64][3][3];
	static DATA_T W13[4][64];
	static DATA_T W15[128][64][1][1];
	static DATA_T W16[4][128];
	static DATA_T W18[128][3][3];
	static DATA_T W19[4][128];
	

    static DATA_T O0_SW[3][224][224];
	static DATA_T O1_SW[3][225][225];
	static DATA_T O2_SW[32][112][112];
	static DATA_T O3_SW[32][112][112];
	static DATA_T O4_SW[32][112][112];
	static DATA_T O5_SW[32][112][112];
	static DATA_T O6_SW[32][112][112];
	static DATA_T O7_SW[32][112][112];
	static DATA_T O8_SW[64][112][112];
	static DATA_T O9_SW[64][112][112];
	static DATA_T O10_SW[64][112][112];
	static DATA_T O11_SW[64][113][113];
	static DATA_T O12_SW[64][56][56];
	static DATA_T O13_SW[64][56][56];
	static DATA_T O14_SW[64][56][56];
	static DATA_T O15_SW[128][56][56];
	static DATA_T O16_SW[128][56][56];
	static DATA_T O17_SW[128][56][56];
	static DATA_T O18_SW[128][56][56];
	static DATA_T O19_SW[128][56][56];
	

    FILE *w_stream = fopen(argv[1], "rb");
    if (w_stream == NULL) printf("weight file was not opened");
    FILE *i_stream = fopen(argv[2], "rb");
    if (i_stream == NULL) printf("input file was not opened");
    FILE *o_stream = fopen("../../cpp_generator/mobilenet/Output/C_output.txt", "w");
    if (o_stream == NULL) printf("Output file was not opened");
    FILE *c_num = fopen("../../cpp_generator/mobilenet/Output/c_output_num.txt", "w");
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

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 3 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W2[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B2[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W3[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W5[j][m][k] = (DATA_T) trash;
			}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B5[m] = (DATA_T) trash;
}
*/
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W6[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W8[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B8[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W9[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W12[j][m][k] = (DATA_T) trash;
			}
	}
}

/*
for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B12[m] = (DATA_T) trash;
}
*/
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W13[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W15[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B15[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W16[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W18[j][m][k] = (DATA_T) trash;
			}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B18[m] = (DATA_T) trash;
}
*/
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W19[x][y] = (DATA_T) trash;
    }
}
	
    printf("[C_verifier.cpp]Finish Initialization");

    printf("[C_verifier.cpp]InputLayer\n\n");
	printf("[C_verifier.cpp]Calculate ZeroPadding2D1\n\n");
	SW_conv1_pad(O0_SW,O1_SW);
	printf("[C_verifier.cpp]Calculate Conv2D2\n\n");
	SW_conv1(O1_SW,O2_SW,W2);
	printf("[C_verifier.cpp]Calculate BatchNormalization3\n\n");
	SW_conv1_bn(O2_SW,O3_SW, W3);
	printf("[C_verifier.cpp]Calculate Relu4\n\n");
	SW_conv1_relu(O3_SW,O4_SW);
	printf("[C_verifier.cpp]Calculate DepthwiseConv2D5\n\n");
	SW_conv_dw_1(O4_SW,O5_SW,W5);
	printf("[C_verifier.cpp]Calculate BatchNormalization6\n\n");
	SW_conv_dw_1_bn(O5_SW,O6_SW, W6);
	printf("[C_verifier.cpp]Calculate Relu7\n\n");
	SW_conv_dw_1_relu(O6_SW,O7_SW);
	printf("[C_verifier.cpp]Calculate Conv2D8\n\n");
	SW_conv_pw_1(O7_SW,O8_SW,W8);
	printf("[C_verifier.cpp]Calculate BatchNormalization9\n\n");
	SW_conv_pw_1_bn(O8_SW,O9_SW, W9);
	printf("[C_verifier.cpp]Calculate Relu10\n\n");
	SW_conv_pw_1_relu(O9_SW,O10_SW);
	printf("[C_verifier.cpp]Calculate ZeroPadding2D11\n\n");
	SW_conv_pad_2(O10_SW,O11_SW);
	printf("[C_verifier.cpp]Calculate DepthwiseConv2D12\n\n");
	SW_conv_dw_2(O11_SW,O12_SW,W12);
	printf("[C_verifier.cpp]Calculate BatchNormalization13\n\n");
	SW_conv_dw_2_bn(O12_SW,O13_SW, W13);
	printf("[C_verifier.cpp]Calculate Relu14\n\n");
	SW_conv_dw_2_relu(O13_SW,O14_SW);
	printf("[C_verifier.cpp]Calculate Conv2D15\n\n");
	SW_conv_pw_2(O14_SW,O15_SW,W15);
	printf("[C_verifier.cpp]Calculate BatchNormalization16\n\n");
	SW_conv_pw_2_bn(O15_SW,O16_SW, W16);
	printf("[C_verifier.cpp]Calculate Relu17\n\n");
	SW_conv_pw_2_relu(O16_SW,O17_SW);
	printf("[C_verifier.cpp]Calculate DepthwiseConv2D18\n\n");
	SW_conv_dw_3(O17_SW,O18_SW,W18);
	printf("[C_verifier.cpp]Calculate BatchNormalization19\n\n");
	SW_conv_dw_3_bn(O18_SW,O19_SW, W19);
	

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
for (k = 0; k < 225 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 225 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 3 ; y++) {
			fprintf(o_stream,"%.6f ",O1_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O1_SW[y][k][x]);
		}
		if(x != 225 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 225 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 112 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 112 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
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
		for(y = 0; y < 32 ; y++) {
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


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 112 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 112 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
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


fprintf(o_stream,"%s","DepthwiseConv2D : [[");
for (k = 0; k < 112 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 112 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O5_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O5_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O6_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O6_SW[y][k][x]);
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


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 112 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 112 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O7_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O7_SW[y][k][x]);
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 112 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 112 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O8_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O8_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O9_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O9_SW[y][k][x]);
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


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 112 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 112 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O10_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O10_SW[y][k][x]);
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


fprintf(o_stream,"%s","ZeroPadding2D : [[");
for (k = 0; k < 113 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 113 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O11_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O11_SW[y][k][x]);
		}
		if(x != 113 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 113 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","DepthwiseConv2D : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O12_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O12_SW[y][k][x]);
		}
		if(x != 56 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 56 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O13_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O13_SW[y][k][x]);
		}
		if(x != 56 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 56 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O14_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O14_SW[y][k][x]);
		}
		if(x != 56 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 56 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O15_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O15_SW[y][k][x]);
		}
		if(x != 56 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 56 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O16_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O16_SW[y][k][x]);
		}
		if(x != 56 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 56 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O17_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O17_SW[y][k][x]);
		}
		if(x != 56 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 56 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","DepthwiseConv2D : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O18_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O18_SW[y][k][x]);
		}
		if(x != 56 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 56 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O19_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O19_SW[y][k][x]);
		}
		if(x != 56 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 56 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");





    fclose(w_stream);
    fclose(i_stream);
    fclose(o_stream);
    fclose(c_num);

    return 0;
}
