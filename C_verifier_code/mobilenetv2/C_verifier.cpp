#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <math.h>
    
using namespace std;

typedef float DATA_T;

void SW_Conv1(DATA_T I[3][224][224], DATA_T O[32][112][112], DATA_T W[32][3][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (2 *(112 - 1) - 224 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<112; x++) {
			for (y = 0; y<112; y++) {
				ofm = 0;
				for (k = 0; k<3; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*2 + i < 224 + p && y*2 + j < 224 + p && x*2 + i -p >= 0 && y*2 + j -p >= 0) {
                                    ifm = I[k][x*2 + i - p][y*2 + j - p];
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

void SW_bn_Conv1(DATA_T I[32][112][112], DATA_T O[32][112][112], DATA_T W[4][32]) {
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
void SW_Conv1_relu(DATA_T I[32][112][112], DATA_T O[32][112][112])
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
void SW_expanded_conv_depthwise(DATA_T I[32][112][112], DATA_T O[32][112][112], DATA_T W[32][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(112 - 1) - 112 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<112; x++) {
			for (y = 0; y<112; y++) {
				ofm = 0;

				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {
						if (x*1 + i < 112 + p && y*1 + j < 112 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_expanded_conv_depthwise_BN(DATA_T I[32][112][112], DATA_T O[32][112][112], DATA_T W[4][32]) {
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
void SW_expanded_conv_depthwise_relu(DATA_T I[32][112][112], DATA_T O[32][112][112])
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
void SW_expanded_conv_project(DATA_T I[32][112][112], DATA_T O[16][112][112], DATA_T W[16][32][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(112 - 1) - 112 + 1)/2;
	for (m = 0; m<16; m++) {
		for (x = 0; x<112; x++) {
			for (y = 0; y<112; y++) {
				ofm = 0;
				for (k = 0; k<32; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 112 + p && y*1 + j < 112 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                                    ifm = I[k][x*1 + i - p][y*1 + j - p];
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

void SW_expanded_conv_project_BN(DATA_T I[16][112][112], DATA_T O[16][112][112], DATA_T W[4][16]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 16; m++){
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
void SW_block_1_expand(DATA_T I[16][112][112], DATA_T O[96][112][112], DATA_T W[96][16][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(112 - 1) - 112 + 1)/2;
	for (m = 0; m<96; m++) {
		for (x = 0; x<112; x++) {
			for (y = 0; y<112; y++) {
				ofm = 0;
				for (k = 0; k<16; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 112 + p && y*1 + j < 112 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                                    ifm = I[k][x*1 + i - p][y*1 + j - p];
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

void SW_block_1_expand_BN(DATA_T I[96][112][112], DATA_T O[96][112][112], DATA_T W[4][96]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 96; m++){
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
void SW_block_1_expand_relu(DATA_T I[96][112][112], DATA_T O[96][112][112])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<96; m++) {
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
void SW_block_1_depthwise(DATA_T I[96][112][112], DATA_T O[96][56][56], DATA_T W[96][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (2 *(56 - 1) - 112 + 3)/2;
	for (m = 0; m<96; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				ofm = 0;

				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {
						if (x*2 + i < 112 + p && y*2 + j < 112 + p && x*2 + i -p >= 0 && y*2 + j -p >= 0) {
                            ifm = I[m][x*2 + i - p][y*2 + j -p];
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

void SW_block_1_depthwise_BN(DATA_T I[96][56][56], DATA_T O[96][56][56], DATA_T W[4][96]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 96; m++){
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
void SW_block_1_depthwise_relu(DATA_T I[96][56][56], DATA_T O[96][56][56])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<96; m++) {
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
void SW_block_1_project(DATA_T I[96][56][56], DATA_T O[24][56][56], DATA_T W[24][96][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(56 - 1) - 56 + 1)/2;
	for (m = 0; m<24; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				ofm = 0;
				for (k = 0; k<96; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 56 + p && y*1 + j < 56 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                                    ifm = I[k][x*1 + i - p][y*1 + j - p];
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

void SW_block_1_project_BN(DATA_T I[24][56][56], DATA_T O[24][56][56], DATA_T W[4][24]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 24; m++){
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
void SW_block_2_expand(DATA_T I[24][56][56], DATA_T O[144][56][56], DATA_T W[144][24][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(56 - 1) - 56 + 1)/2;
	for (m = 0; m<144; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				ofm = 0;
				for (k = 0; k<24; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 56 + p && y*1 + j < 56 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                                    ifm = I[k][x*1 + i - p][y*1 + j - p];
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

void SW_block_2_expand_BN(DATA_T I[144][56][56], DATA_T O[144][56][56], DATA_T W[4][144]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 144; m++){
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
void SW_block_2_expand_relu(DATA_T I[144][56][56], DATA_T O[144][56][56])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<144; m++) {
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
void SW_block_2_depthwise(DATA_T I[144][56][56], DATA_T O[144][56][56], DATA_T W[144][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(56 - 1) - 56 + 3)/2;
	for (m = 0; m<144; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				ofm = 0;

				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {
						if (x*1 + i < 56 + p && y*1 + j < 56 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_block_2_depthwise_BN(DATA_T I[144][56][56], DATA_T O[144][56][56], DATA_T W[4][144]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 144; m++){
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
void SW_block_2_depthwise_relu(DATA_T I[144][56][56], DATA_T O[144][56][56])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<144; m++) {
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
void SW_block_2_project(DATA_T I[144][56][56], DATA_T O[24][56][56], DATA_T W[24][144][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(56 - 1) - 56 + 1)/2;
	for (m = 0; m<24; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				ofm = 0;
				for (k = 0; k<144; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 56 + p && y*1 + j < 56 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                                    ifm = I[k][x*1 + i - p][y*1 + j - p];
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

void SW_block_2_project_BN(DATA_T I[24][56][56], DATA_T O[24][56][56], DATA_T W[4][24]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 24; m++){
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
void SW_block_2_add(DATA_T I1[24][56][56], DATA_T I2[24][56][56], DATA_T O[24][56][56]) {
	int m, x, y;
	for (m = 0; m< 24; m++) {
		for (x = 0; x < 56; x++) {
			for(y = 0; y < 56; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_block_3_expand(DATA_T I[24][56][56], DATA_T O[144][56][56], DATA_T W[144][24][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(56 - 1) - 56 + 1)/2;
	for (m = 0; m<144; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				ofm = 0;
				for (k = 0; k<24; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 56 + p && y*1 + j < 56 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                                    ifm = I[k][x*1 + i - p][y*1 + j - p];
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



//argv[1] = init_weight.txt, argv[3] = init_input.txt
int main(int argc, char *argv[]){

    DATA_T temp;
    int m, x, y, i, j, k, l;
    int trash;

    static DATA_T I[3][224][224];
	static DATA_T W1[32][3][3][3];
	static DATA_T W2[4][32];
	static DATA_T W4[32][3][3];
	static DATA_T W5[4][32];
	static DATA_T W7[16][32][1][1];
	static DATA_T W8[4][16];
	static DATA_T W9[96][16][1][1];
	static DATA_T W10[4][96];
	static DATA_T W12[96][3][3];
	static DATA_T W13[4][96];
	static DATA_T W15[24][96][1][1];
	static DATA_T W16[4][24];
	static DATA_T W17[144][24][1][1];
	static DATA_T W18[4][144];
	static DATA_T W20[144][3][3];
	static DATA_T W21[4][144];
	static DATA_T W23[24][144][1][1];
	static DATA_T W24[4][24];
	static DATA_T W26[144][24][1][1];
	

    static DATA_T O0_SW[3][224][224];
	static DATA_T O1_SW[32][112][112];
	static DATA_T O2_SW[32][112][112];
	static DATA_T O3_SW[32][112][112];
	static DATA_T O4_SW[32][112][112];
	static DATA_T O5_SW[32][112][112];
	static DATA_T O6_SW[32][112][112];
	static DATA_T O7_SW[16][112][112];
	static DATA_T O8_SW[16][112][112];
	static DATA_T O9_SW[96][112][112];
	static DATA_T O10_SW[96][112][112];
	static DATA_T O11_SW[96][112][112];
	static DATA_T O12_SW[96][56][56];
	static DATA_T O13_SW[96][56][56];
	static DATA_T O14_SW[96][56][56];
	static DATA_T O15_SW[24][56][56];
	static DATA_T O16_SW[24][56][56];
	static DATA_T O17_SW[144][56][56];
	static DATA_T O18_SW[144][56][56];
	static DATA_T O19_SW[144][56][56];
	static DATA_T O20_SW[144][56][56];
	static DATA_T O21_SW[144][56][56];
	static DATA_T O22_SW[144][56][56];
	static DATA_T O23_SW[24][56][56];
	static DATA_T O24_SW[24][56][56];
	static DATA_T O25_SW[24][56][56];
	static DATA_T O26_SW[144][56][56];
	

    FILE *w_stream = fopen(argv[1], "rb");
    if (w_stream == NULL) printf("weight file was not opened");
    FILE *i_stream = fopen(argv[2], "rb");
    if (i_stream == NULL) printf("input file was not opened");
    FILE *o_stream = fopen("../../cpp_generator/mobilenetv2/Output/C_output.txt", "w");
    if (o_stream == NULL) printf("Output file was not opened");
    FILE *c_num = fopen("../../cpp_generator/mobilenetv2/Output/c_output_num.txt", "w");
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
                W1[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B1[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W2[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W4[j][m][k] = (DATA_T) trash;
			}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B4[m] = (DATA_T) trash;
}
*/
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W5[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 16 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W7[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 16 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B7[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 16 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W8[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 16 ; i++) {
			for (j = 0; j < 96 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W9[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 96 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B9[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 96 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W10[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
			for (j = 0; j < 96 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W12[j][m][k] = (DATA_T) trash;
			}
	}
}

/*
for (m = 0; m < 96 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B12[m] = (DATA_T) trash;
}
*/
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 96 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W13[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 96 ; i++) {
			for (j = 0; j < 24 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W15[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 24 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B15[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 24 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W16[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 24 ; i++) {
			for (j = 0; j < 144 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W17[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 144 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B17[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 144 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W18[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
			for (j = 0; j < 144 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W20[j][m][k] = (DATA_T) trash;
			}
	}
}

/*
for (m = 0; m < 144 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B20[m] = (DATA_T) trash;
}
*/
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 144 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W21[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 144 ; i++) {
			for (j = 0; j < 24 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W23[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 24 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B23[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 24 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W24[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 24 ; i++) {
			for (j = 0; j < 144 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W26[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 144 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B26[m] = (DATA_T) trash;
}
*/

	
    printf("[C_verifier.cpp]Finish Initialization");

    printf("[C_verifier.cpp]InputLayer\n\n");
	printf("[C_verifier.cpp]Calculate Conv2D1\n\n");
	SW_Conv1(O0_SW,O1_SW,W1);
	printf("[C_verifier.cpp]Calculate BatchNormalization2\n\n");
	SW_bn_Conv1(O1_SW,O2_SW, W2);
	printf("[C_verifier.cpp]Calculate Relu3\n\n");
	SW_Conv1_relu(O2_SW,O3_SW);
	printf("[C_verifier.cpp]Calculate DepthwiseConv2D4\n\n");
	SW_expanded_conv_depthwise(O3_SW,O4_SW,W4);
	printf("[C_verifier.cpp]Calculate BatchNormalization5\n\n");
	SW_expanded_conv_depthwise_BN(O4_SW,O5_SW, W5);
	printf("[C_verifier.cpp]Calculate Relu6\n\n");
	SW_expanded_conv_depthwise_relu(O5_SW,O6_SW);
	printf("[C_verifier.cpp]Calculate Conv2D7\n\n");
	SW_expanded_conv_project(O6_SW,O7_SW,W7);
	printf("[C_verifier.cpp]Calculate BatchNormalization8\n\n");
	SW_expanded_conv_project_BN(O7_SW,O8_SW, W8);
	printf("[C_verifier.cpp]Calculate Conv2D9\n\n");
	SW_block_1_expand(O8_SW,O9_SW,W9);
	printf("[C_verifier.cpp]Calculate BatchNormalization10\n\n");
	SW_block_1_expand_BN(O9_SW,O10_SW, W10);
	printf("[C_verifier.cpp]Calculate Relu11\n\n");
	SW_block_1_expand_relu(O10_SW,O11_SW);
	printf("[C_verifier.cpp]Calculate DepthwiseConv2D12\n\n");
	SW_block_1_depthwise(O11_SW,O12_SW,W12);
	printf("[C_verifier.cpp]Calculate BatchNormalization13\n\n");
	SW_block_1_depthwise_BN(O12_SW,O13_SW, W13);
	printf("[C_verifier.cpp]Calculate Relu14\n\n");
	SW_block_1_depthwise_relu(O13_SW,O14_SW);
	printf("[C_verifier.cpp]Calculate Conv2D15\n\n");
	SW_block_1_project(O14_SW,O15_SW,W15);
	printf("[C_verifier.cpp]Calculate BatchNormalization16\n\n");
	SW_block_1_project_BN(O15_SW,O16_SW, W16);
	printf("[C_verifier.cpp]Calculate Conv2D17\n\n");
	SW_block_2_expand(O16_SW,O17_SW,W17);
	printf("[C_verifier.cpp]Calculate BatchNormalization18\n\n");
	SW_block_2_expand_BN(O17_SW,O18_SW, W18);
	printf("[C_verifier.cpp]Calculate Relu19\n\n");
	SW_block_2_expand_relu(O18_SW,O19_SW);
	printf("[C_verifier.cpp]Calculate DepthwiseConv2D20\n\n");
	SW_block_2_depthwise(O19_SW,O20_SW,W20);
	printf("[C_verifier.cpp]Calculate BatchNormalization21\n\n");
	SW_block_2_depthwise_BN(O20_SW,O21_SW, W21);
	printf("[C_verifier.cpp]Calculate Relu22\n\n");
	SW_block_2_depthwise_relu(O21_SW,O22_SW);
	printf("[C_verifier.cpp]Calculate Conv2D23\n\n");
	SW_block_2_project(O22_SW,O23_SW,W23);
	printf("[C_verifier.cpp]Calculate BatchNormalization24\n\n");
	SW_block_2_project_BN(O23_SW,O24_SW, W24);
	printf("[C_verifier.cpp]Calculate Add25\n\n");
	SW_block_2_add(O16_SW,O24_SW,O25_SW);
	printf("[C_verifier.cpp]Calculate Conv2D26\n\n");
	SW_block_3_expand(O25_SW,O26_SW,W26);
	

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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 112 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 112 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O1_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O1_SW[y][k][x]);
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


fprintf(o_stream,"%s","ReLU : [[");
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


fprintf(o_stream,"%s","DepthwiseConv2D : [[");
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


fprintf(o_stream,"%s","BatchNormalization : [[");
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


fprintf(o_stream,"%s","ReLU : [[");
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 112 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 112 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 16 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 112 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 112 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 16 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 112 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 112 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 96 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 112 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 112 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 96 ; y++) {
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


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 112 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 112 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 96 ; y++) {
			fprintf(o_stream,"%.6f ",O11_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O11_SW[y][k][x]);
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
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 96 ; y++) {
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
		for(y = 0; y < 96 ; y++) {
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
		for(y = 0; y < 96 ; y++) {
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
		for(y = 0; y < 24 ; y++) {
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
		for(y = 0; y < 24 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 144 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 144 ; y++) {
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


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 144 ; y++) {
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


fprintf(o_stream,"%s","DepthwiseConv2D : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 144 ; y++) {
			fprintf(o_stream,"%.6f ",O20_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O20_SW[y][k][x]);
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
		for(y = 0; y < 144 ; y++) {
			fprintf(o_stream,"%.6f ",O21_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O21_SW[y][k][x]);
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
		for(y = 0; y < 144 ; y++) {
			fprintf(o_stream,"%.6f ",O22_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O22_SW[y][k][x]);
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
		for(y = 0; y < 24 ; y++) {
			fprintf(o_stream,"%.6f ",O23_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O23_SW[y][k][x]);
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
		for(y = 0; y < 24 ; y++) {
			fprintf(o_stream,"%.6f ",O24_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O24_SW[y][k][x]);
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


fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 24 ; y++) {
			fprintf(o_stream,"%.6f ",O25_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O25_SW[y][k][x]);
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
		for(y = 0; y < 144 ; y++) {
			fprintf(o_stream,"%.6f ",O26_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O26_SW[y][k][x]);
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
