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
void SW_activation_4(DATA_T I[256][55][55], DATA_T O[256][55][55])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<256; m++) {
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
void SW_res2b_branch2a(DATA_T I[256][55][55], DATA_T O[64][55][55], DATA_T W[64][256][1][1], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<64; m++) {
		for (x = 0; x<55; x++) {
			for (y = 0; y<55; y++) {
				ofm = 0;
				for (k = 0; k<256; k++) {
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

void SW_bn2b_branch2a(DATA_T I[64][55][55], DATA_T O[64][55][55], DATA_T W[4][64]) {
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
void SW_activation_5(DATA_T I[64][55][55], DATA_T O[64][55][55])
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
void SW_res2b_branch2b(DATA_T I[64][55][55], DATA_T O[64][55][55], DATA_T W[64][64][3][3], DATA_T B[64]) {
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

void SW_bn2b_branch2b(DATA_T I[64][55][55], DATA_T O[64][55][55], DATA_T W[4][64]) {
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
void SW_activation_6(DATA_T I[64][55][55], DATA_T O[64][55][55])
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
void SW_res2b_branch2c(DATA_T I[64][55][55], DATA_T O[256][55][55], DATA_T W[256][64][1][1], DATA_T B[256]) {
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

void SW_bn2b_branch2c(DATA_T I[256][55][55], DATA_T O[256][55][55], DATA_T W[4][256]) {
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
void SW_add_2(DATA_T I1[256][55][55], DATA_T I2[256][55][55], DATA_T O[256][55][55]) {
	int m, x, y;
	for (m = 0; m< 256; m++) {
		for (x = 0; x < 55; x++) {
			for(y = 0; y < 55; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_activation_7(DATA_T I[256][55][55], DATA_T O[256][55][55])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<256; m++) {
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
void SW_res2c_branch2a(DATA_T I[256][55][55], DATA_T O[64][55][55], DATA_T W[64][256][1][1], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<64; m++) {
		for (x = 0; x<55; x++) {
			for (y = 0; y<55; y++) {
				ofm = 0;
				for (k = 0; k<256; k++) {
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

void SW_bn2c_branch2a(DATA_T I[64][55][55], DATA_T O[64][55][55], DATA_T W[4][64]) {
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
void SW_activation_8(DATA_T I[64][55][55], DATA_T O[64][55][55])
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
void SW_res2c_branch2b(DATA_T I[64][55][55], DATA_T O[64][55][55], DATA_T W[64][64][3][3], DATA_T B[64]) {
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

void SW_bn2c_branch2b(DATA_T I[64][55][55], DATA_T O[64][55][55], DATA_T W[4][64]) {
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
void SW_activation_9(DATA_T I[64][55][55], DATA_T O[64][55][55])
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
void SW_res2c_branch2c(DATA_T I[64][55][55], DATA_T O[256][55][55], DATA_T W[256][64][1][1], DATA_T B[256]) {
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

void SW_bn2c_branch2c(DATA_T I[256][55][55], DATA_T O[256][55][55], DATA_T W[4][256]) {
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
void SW_add_3(DATA_T I1[256][55][55], DATA_T I2[256][55][55], DATA_T O[256][55][55]) {
	int m, x, y;
	for (m = 0; m< 256; m++) {
		for (x = 0; x < 55; x++) {
			for(y = 0; y < 55; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_activation_10(DATA_T I[256][55][55], DATA_T O[256][55][55])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<256; m++) {
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
void SW_res3a_branch2a(DATA_T I[256][55][55], DATA_T O[128][28][28], DATA_T W[128][256][1][1], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<256; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 55 && y + j <= 55) {
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

void SW_bn3a_branch2a(DATA_T I[128][28][28], DATA_T O[128][28][28], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 128; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 28; x++){
			for (y = 0; y < 28; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_11(DATA_T I[128][28][28], DATA_T O[128][28][28])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res3a_branch2b(DATA_T I[128][28][28], DATA_T O[128][28][28], DATA_T W[128][128][3][3], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 3)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 28 + p && y + j < 28 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_bn3a_branch2b(DATA_T I[128][28][28], DATA_T O[128][28][28], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 128; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 28; x++){
			for (y = 0; y < 28; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_12(DATA_T I[128][28][28], DATA_T O[128][28][28])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res3a_branch2c(DATA_T I[128][28][28], DATA_T O[512][28][28], DATA_T W[512][128][1][1], DATA_T B[512]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<512; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 28 && y + j <= 28) {
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

void SW_res3a_branch1(DATA_T I[256][55][55], DATA_T O[512][28][28], DATA_T W[512][256][1][1], DATA_T B[512]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<512; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<256; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 55 && y + j <= 55) {
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

void SW_bn3a_branch2c(DATA_T I[512][28][28], DATA_T O[512][28][28], DATA_T W[4][512]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 512; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 28; x++){
			for (y = 0; y < 28; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_bn3a_branch1(DATA_T I[512][28][28], DATA_T O[512][28][28], DATA_T W[4][512]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 512; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 28; x++){
			for (y = 0; y < 28; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_add_4(DATA_T I1[512][28][28], DATA_T I2[512][28][28], DATA_T O[512][28][28]) {
	int m, x, y;
	for (m = 0; m< 512; m++) {
		for (x = 0; x < 28; x++) {
			for(y = 0; y < 28; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_activation_13(DATA_T I[512][28][28], DATA_T O[512][28][28])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<512; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res3b_branch2a(DATA_T I[512][28][28], DATA_T O[128][28][28], DATA_T W[128][512][1][1], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<512; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 28 && y + j <= 28) {
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

void SW_bn3b_branch2a(DATA_T I[128][28][28], DATA_T O[128][28][28], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 128; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 28; x++){
			for (y = 0; y < 28; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_14(DATA_T I[128][28][28], DATA_T O[128][28][28])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res3b_branch2b(DATA_T I[128][28][28], DATA_T O[128][28][28], DATA_T W[128][128][3][3], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 3)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 28 + p && y + j < 28 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_bn3b_branch2b(DATA_T I[128][28][28], DATA_T O[128][28][28], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 128; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 28; x++){
			for (y = 0; y < 28; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_15(DATA_T I[128][28][28], DATA_T O[128][28][28])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res3b_branch2c(DATA_T I[128][28][28], DATA_T O[512][28][28], DATA_T W[512][128][1][1], DATA_T B[512]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<512; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 28 && y + j <= 28) {
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

void SW_bn3b_branch2c(DATA_T I[512][28][28], DATA_T O[512][28][28], DATA_T W[4][512]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 512; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 28; x++){
			for (y = 0; y < 28; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_add_5(DATA_T I1[512][28][28], DATA_T I2[512][28][28], DATA_T O[512][28][28]) {
	int m, x, y;
	for (m = 0; m< 512; m++) {
		for (x = 0; x < 28; x++) {
			for(y = 0; y < 28; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_activation_16(DATA_T I[512][28][28], DATA_T O[512][28][28])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<512; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res3c_branch2a(DATA_T I[512][28][28], DATA_T O[128][28][28], DATA_T W[128][512][1][1], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<512; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 28 && y + j <= 28) {
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

void SW_bn3c_branch2a(DATA_T I[128][28][28], DATA_T O[128][28][28], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 128; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 28; x++){
			for (y = 0; y < 28; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_17(DATA_T I[128][28][28], DATA_T O[128][28][28])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res3c_branch2b(DATA_T I[128][28][28], DATA_T O[128][28][28], DATA_T W[128][128][3][3], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 3)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 28 + p && y + j < 28 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_bn3c_branch2b(DATA_T I[128][28][28], DATA_T O[128][28][28], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 128; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 28; x++){
			for (y = 0; y < 28; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_18(DATA_T I[128][28][28], DATA_T O[128][28][28])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res3c_branch2c(DATA_T I[128][28][28], DATA_T O[512][28][28], DATA_T W[512][128][1][1], DATA_T B[512]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<512; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 28 && y + j <= 28) {
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

void SW_bn3c_branch2c(DATA_T I[512][28][28], DATA_T O[512][28][28], DATA_T W[4][512]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 512; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 28; x++){
			for (y = 0; y < 28; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_add_6(DATA_T I1[512][28][28], DATA_T I2[512][28][28], DATA_T O[512][28][28]) {
	int m, x, y;
	for (m = 0; m< 512; m++) {
		for (x = 0; x < 28; x++) {
			for(y = 0; y < 28; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_activation_19(DATA_T I[512][28][28], DATA_T O[512][28][28])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<512; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res3d_branch2a(DATA_T I[512][28][28], DATA_T O[128][28][28], DATA_T W[128][512][1][1], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<512; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 28 && y + j <= 28) {
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

void SW_bn3d_branch2a(DATA_T I[128][28][28], DATA_T O[128][28][28], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 128; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 28; x++){
			for (y = 0; y < 28; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_20(DATA_T I[128][28][28], DATA_T O[128][28][28])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res3d_branch2b(DATA_T I[128][28][28], DATA_T O[128][28][28], DATA_T W[128][128][3][3], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 3)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 28 + p && y + j < 28 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_bn3d_branch2b(DATA_T I[128][28][28], DATA_T O[128][28][28], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 128; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 28; x++){
			for (y = 0; y < 28; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_21(DATA_T I[128][28][28], DATA_T O[128][28][28])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res3d_branch2c(DATA_T I[128][28][28], DATA_T O[512][28][28], DATA_T W[512][128][1][1], DATA_T B[512]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<512; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 28 && y + j <= 28) {
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

void SW_bn3d_branch2c(DATA_T I[512][28][28], DATA_T O[512][28][28], DATA_T W[4][512]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 512; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 28; x++){
			for (y = 0; y < 28; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_add_7(DATA_T I1[512][28][28], DATA_T I2[512][28][28], DATA_T O[512][28][28]) {
	int m, x, y;
	for (m = 0; m< 512; m++) {
		for (x = 0; x < 28; x++) {
			for(y = 0; y < 28; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_activation_22(DATA_T I[512][28][28], DATA_T O[512][28][28])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<512; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res4a_branch2a(DATA_T I[512][28][28], DATA_T O[256][14][14], DATA_T W[256][512][1][1], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<512; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 28 && y + j <= 28) {
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

void SW_bn4a_branch2a(DATA_T I[256][14][14], DATA_T O[256][14][14], DATA_T W[4][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 256; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 14; x++){
			for (y = 0; y < 14; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_23(DATA_T I[256][14][14], DATA_T O[256][14][14])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res4a_branch2b(DATA_T I[256][14][14], DATA_T O[256][14][14], DATA_T W[256][256][3][3], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<256; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 14 + p && y + j < 14 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_bn4a_branch2b(DATA_T I[256][14][14], DATA_T O[256][14][14], DATA_T W[4][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 256; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 14; x++){
			for (y = 0; y < 14; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_24(DATA_T I[256][14][14], DATA_T O[256][14][14])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res4a_branch2c(DATA_T I[256][14][14], DATA_T O[1024][14][14], DATA_T W[1024][256][1][1], DATA_T B[1024]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<1024; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<256; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_res4a_branch1(DATA_T I[512][28][28], DATA_T O[1024][14][14], DATA_T W[1024][512][1][1], DATA_T B[1024]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<1024; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<512; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 28 && y + j <= 28) {
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

void SW_bn4a_branch2c(DATA_T I[1024][14][14], DATA_T O[1024][14][14], DATA_T W[4][1024]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 1024; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 14; x++){
			for (y = 0; y < 14; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_bn4a_branch1(DATA_T I[1024][14][14], DATA_T O[1024][14][14], DATA_T W[4][1024]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 1024; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 14; x++){
			for (y = 0; y < 14; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_add_8(DATA_T I1[1024][14][14], DATA_T I2[1024][14][14], DATA_T O[1024][14][14]) {
	int m, x, y;
	for (m = 0; m< 1024; m++) {
		for (x = 0; x < 14; x++) {
			for(y = 0; y < 14; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_activation_25(DATA_T I[1024][14][14], DATA_T O[1024][14][14])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<1024; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res4b_branch2a(DATA_T I[1024][14][14], DATA_T O[256][14][14], DATA_T W[256][1024][1][1], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<1024; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_bn4b_branch2a(DATA_T I[256][14][14], DATA_T O[256][14][14], DATA_T W[4][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 256; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 14; x++){
			for (y = 0; y < 14; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_26(DATA_T I[256][14][14], DATA_T O[256][14][14])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res4b_branch2b(DATA_T I[256][14][14], DATA_T O[256][14][14], DATA_T W[256][256][3][3], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<256; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 14 + p && y + j < 14 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_bn4b_branch2b(DATA_T I[256][14][14], DATA_T O[256][14][14], DATA_T W[4][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 256; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 14; x++){
			for (y = 0; y < 14; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_27(DATA_T I[256][14][14], DATA_T O[256][14][14])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res4b_branch2c(DATA_T I[256][14][14], DATA_T O[1024][14][14], DATA_T W[1024][256][1][1], DATA_T B[1024]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<1024; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<256; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_bn4b_branch2c(DATA_T I[1024][14][14], DATA_T O[1024][14][14], DATA_T W[4][1024]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 1024; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 14; x++){
			for (y = 0; y < 14; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_add_9(DATA_T I1[1024][14][14], DATA_T I2[1024][14][14], DATA_T O[1024][14][14]) {
	int m, x, y;
	for (m = 0; m< 1024; m++) {
		for (x = 0; x < 14; x++) {
			for(y = 0; y < 14; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_activation_28(DATA_T I[1024][14][14], DATA_T O[1024][14][14])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<1024; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res4c_branch2a(DATA_T I[1024][14][14], DATA_T O[256][14][14], DATA_T W[256][1024][1][1], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<1024; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_bn4c_branch2a(DATA_T I[256][14][14], DATA_T O[256][14][14], DATA_T W[4][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 256; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 14; x++){
			for (y = 0; y < 14; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_29(DATA_T I[256][14][14], DATA_T O[256][14][14])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res4c_branch2b(DATA_T I[256][14][14], DATA_T O[256][14][14], DATA_T W[256][256][3][3], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<256; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 14 + p && y + j < 14 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_bn4c_branch2b(DATA_T I[256][14][14], DATA_T O[256][14][14], DATA_T W[4][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 256; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 14; x++){
			for (y = 0; y < 14; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_30(DATA_T I[256][14][14], DATA_T O[256][14][14])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res4c_branch2c(DATA_T I[256][14][14], DATA_T O[1024][14][14], DATA_T W[1024][256][1][1], DATA_T B[1024]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<1024; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<256; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_bn4c_branch2c(DATA_T I[1024][14][14], DATA_T O[1024][14][14], DATA_T W[4][1024]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 1024; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 14; x++){
			for (y = 0; y < 14; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_add_10(DATA_T I1[1024][14][14], DATA_T I2[1024][14][14], DATA_T O[1024][14][14]) {
	int m, x, y;
	for (m = 0; m< 1024; m++) {
		for (x = 0; x < 14; x++) {
			for(y = 0; y < 14; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_activation_31(DATA_T I[1024][14][14], DATA_T O[1024][14][14])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<1024; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res4d_branch2a(DATA_T I[1024][14][14], DATA_T O[256][14][14], DATA_T W[256][1024][1][1], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<1024; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_bn4d_branch2a(DATA_T I[256][14][14], DATA_T O[256][14][14], DATA_T W[4][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 256; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 14; x++){
			for (y = 0; y < 14; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_32(DATA_T I[256][14][14], DATA_T O[256][14][14])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res4d_branch2b(DATA_T I[256][14][14], DATA_T O[256][14][14], DATA_T W[256][256][3][3], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<256; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 14 + p && y + j < 14 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_bn4d_branch2b(DATA_T I[256][14][14], DATA_T O[256][14][14], DATA_T W[4][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 256; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 14; x++){
			for (y = 0; y < 14; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_33(DATA_T I[256][14][14], DATA_T O[256][14][14])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res4d_branch2c(DATA_T I[256][14][14], DATA_T O[1024][14][14], DATA_T W[1024][256][1][1], DATA_T B[1024]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<1024; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<256; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_bn4d_branch2c(DATA_T I[1024][14][14], DATA_T O[1024][14][14], DATA_T W[4][1024]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 1024; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 14; x++){
			for (y = 0; y < 14; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_add_11(DATA_T I1[1024][14][14], DATA_T I2[1024][14][14], DATA_T O[1024][14][14]) {
	int m, x, y;
	for (m = 0; m< 1024; m++) {
		for (x = 0; x < 14; x++) {
			for(y = 0; y < 14; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_activation_34(DATA_T I[1024][14][14], DATA_T O[1024][14][14])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<1024; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res4e_branch2a(DATA_T I[1024][14][14], DATA_T O[256][14][14], DATA_T W[256][1024][1][1], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<1024; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_bn4e_branch2a(DATA_T I[256][14][14], DATA_T O[256][14][14], DATA_T W[4][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 256; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 14; x++){
			for (y = 0; y < 14; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_35(DATA_T I[256][14][14], DATA_T O[256][14][14])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res4e_branch2b(DATA_T I[256][14][14], DATA_T O[256][14][14], DATA_T W[256][256][3][3], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<256; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 14 + p && y + j < 14 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_bn4e_branch2b(DATA_T I[256][14][14], DATA_T O[256][14][14], DATA_T W[4][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 256; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 14; x++){
			for (y = 0; y < 14; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_36(DATA_T I[256][14][14], DATA_T O[256][14][14])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res4e_branch2c(DATA_T I[256][14][14], DATA_T O[1024][14][14], DATA_T W[1024][256][1][1], DATA_T B[1024]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<1024; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<256; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_bn4e_branch2c(DATA_T I[1024][14][14], DATA_T O[1024][14][14], DATA_T W[4][1024]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 1024; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 14; x++){
			for (y = 0; y < 14; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_add_12(DATA_T I1[1024][14][14], DATA_T I2[1024][14][14], DATA_T O[1024][14][14]) {
	int m, x, y;
	for (m = 0; m< 1024; m++) {
		for (x = 0; x < 14; x++) {
			for(y = 0; y < 14; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_activation_37(DATA_T I[1024][14][14], DATA_T O[1024][14][14])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<1024; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res4f_branch2a(DATA_T I[1024][14][14], DATA_T O[256][14][14], DATA_T W[256][1024][1][1], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<1024; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_bn4f_branch2a(DATA_T I[256][14][14], DATA_T O[256][14][14], DATA_T W[4][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 256; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 14; x++){
			for (y = 0; y < 14; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_38(DATA_T I[256][14][14], DATA_T O[256][14][14])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res4f_branch2b(DATA_T I[256][14][14], DATA_T O[256][14][14], DATA_T W[256][256][3][3], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<256; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 14 + p && y + j < 14 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_bn4f_branch2b(DATA_T I[256][14][14], DATA_T O[256][14][14], DATA_T W[4][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 256; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 14; x++){
			for (y = 0; y < 14; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_39(DATA_T I[256][14][14], DATA_T O[256][14][14])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res4f_branch2c(DATA_T I[256][14][14], DATA_T O[1024][14][14], DATA_T W[1024][256][1][1], DATA_T B[1024]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<1024; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<256; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_bn4f_branch2c(DATA_T I[1024][14][14], DATA_T O[1024][14][14], DATA_T W[4][1024]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 1024; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 14; x++){
			for (y = 0; y < 14; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_add_13(DATA_T I1[1024][14][14], DATA_T I2[1024][14][14], DATA_T O[1024][14][14]) {
	int m, x, y;
	for (m = 0; m< 1024; m++) {
		for (x = 0; x < 14; x++) {
			for(y = 0; y < 14; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_activation_40(DATA_T I[1024][14][14], DATA_T O[1024][14][14])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<1024; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res5a_branch2a(DATA_T I[1024][14][14], DATA_T O[512][7][7], DATA_T W[512][1024][1][1], DATA_T B[512]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<512; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<1024; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_bn5a_branch2a(DATA_T I[512][7][7], DATA_T O[512][7][7], DATA_T W[4][512]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 512; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 7; x++){
			for (y = 0; y < 7; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_41(DATA_T I[512][7][7], DATA_T O[512][7][7])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<512; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res5a_branch2b(DATA_T I[512][7][7], DATA_T O[512][7][7], DATA_T W[512][512][3][3], DATA_T B[512]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<512; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<512; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 7 + p && y + j < 7 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_bn5a_branch2b(DATA_T I[512][7][7], DATA_T O[512][7][7], DATA_T W[4][512]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 512; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 7; x++){
			for (y = 0; y < 7; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_42(DATA_T I[512][7][7], DATA_T O[512][7][7])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<512; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res5a_branch2c(DATA_T I[512][7][7], DATA_T O[2048][7][7], DATA_T W[2048][512][1][1], DATA_T B[2048]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<2048; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<512; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_res5a_branch1(DATA_T I[1024][14][14], DATA_T O[2048][7][7], DATA_T W[2048][1024][1][1], DATA_T B[2048]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<2048; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<1024; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_bn5a_branch2c(DATA_T I[2048][7][7], DATA_T O[2048][7][7], DATA_T W[4][2048]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 2048; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 7; x++){
			for (y = 0; y < 7; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_bn5a_branch1(DATA_T I[2048][7][7], DATA_T O[2048][7][7], DATA_T W[4][2048]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 2048; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 7; x++){
			for (y = 0; y < 7; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_add_14(DATA_T I1[2048][7][7], DATA_T I2[2048][7][7], DATA_T O[2048][7][7]) {
	int m, x, y;
	for (m = 0; m< 2048; m++) {
		for (x = 0; x < 7; x++) {
			for(y = 0; y < 7; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_activation_43(DATA_T I[2048][7][7], DATA_T O[2048][7][7])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<2048; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res5b_branch2a(DATA_T I[2048][7][7], DATA_T O[512][7][7], DATA_T W[512][2048][1][1], DATA_T B[512]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<512; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<2048; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_bn5b_branch2a(DATA_T I[512][7][7], DATA_T O[512][7][7], DATA_T W[4][512]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 512; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 7; x++){
			for (y = 0; y < 7; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_44(DATA_T I[512][7][7], DATA_T O[512][7][7])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<512; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res5b_branch2b(DATA_T I[512][7][7], DATA_T O[512][7][7], DATA_T W[512][512][3][3], DATA_T B[512]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<512; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<512; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 7 + p && y + j < 7 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_bn5b_branch2b(DATA_T I[512][7][7], DATA_T O[512][7][7], DATA_T W[4][512]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 512; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 7; x++){
			for (y = 0; y < 7; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_45(DATA_T I[512][7][7], DATA_T O[512][7][7])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<512; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res5b_branch2c(DATA_T I[512][7][7], DATA_T O[2048][7][7], DATA_T W[2048][512][1][1], DATA_T B[2048]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<2048; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<512; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_bn5b_branch2c(DATA_T I[2048][7][7], DATA_T O[2048][7][7], DATA_T W[4][2048]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 2048; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 7; x++){
			for (y = 0; y < 7; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_add_15(DATA_T I1[2048][7][7], DATA_T I2[2048][7][7], DATA_T O[2048][7][7]) {
	int m, x, y;
	for (m = 0; m< 2048; m++) {
		for (x = 0; x < 7; x++) {
			for(y = 0; y < 7; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_activation_46(DATA_T I[2048][7][7], DATA_T O[2048][7][7])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<2048; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res5c_branch2a(DATA_T I[2048][7][7], DATA_T O[512][7][7], DATA_T W[512][2048][1][1], DATA_T B[512]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<512; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<2048; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_bn5c_branch2a(DATA_T I[512][7][7], DATA_T O[512][7][7], DATA_T W[4][512]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 512; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 7; x++){
			for (y = 0; y < 7; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_47(DATA_T I[512][7][7], DATA_T O[512][7][7])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<512; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res5c_branch2b(DATA_T I[512][7][7], DATA_T O[512][7][7], DATA_T W[512][512][3][3], DATA_T B[512]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<512; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<512; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 7 + p && y + j < 7 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_bn5c_branch2b(DATA_T I[512][7][7], DATA_T O[512][7][7], DATA_T W[4][512]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 512; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 7; x++){
			for (y = 0; y < 7; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_48(DATA_T I[512][7][7], DATA_T O[512][7][7])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<512; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_res5c_branch2c(DATA_T I[512][7][7], DATA_T O[2048][7][7], DATA_T W[2048][512][1][1], DATA_T B[2048]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<2048; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<512; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_bn5c_branch2c(DATA_T I[2048][7][7], DATA_T O[2048][7][7], DATA_T W[4][2048]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 2048; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 7; x++){
			for (y = 0; y < 7; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_add_16(DATA_T I1[2048][7][7], DATA_T I2[2048][7][7], DATA_T O[2048][7][7]) {
	int m, x, y;
	for (m = 0; m< 2048; m++) {
		for (x = 0; x < 7; x++) {
			for(y = 0; y < 7; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_activation_49(DATA_T I[2048][7][7], DATA_T O[2048][7][7])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<2048; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_avg_pool(DATA_T I[2048][7][7], DATA_T O[2048][1][1])
{
	int m, x, y, i, j;
	DATA_T sum;
	int p = (7*(1-1) - 7 + 7)/2;
    DATA_T div;
	for (m = 0; m<2048; m++) {
		for (x = 0; x<1; x++) {
			for (y = 0; y<1; y++) {
                sum = (DATA_T)0;
                if(x!=0 && x!=1-1 && y!=0 && y!=1-1)
                    div = 9;
                else if((x==0 || x==1-1) && (y==0 || y==1-1))
                    div = 4;
                else
                    div = 6;
				for (i = 0; i<7; i++) {
					for (j = 0; j<7; j++) {

						if (x + i < 7 + p && y + j < 7 + p && x + i -p >= 0 && y + j -p >= 0)
						    sum += I[m][x*7 + i -p][y*7 + j -p];
					}
				}
				O[m][x][y] = sum/div;
			}
		}
	}
}
void SW_flatten_1(DATA_T I[2048][1][1], DATA_T O[2048]){
	int i, j, x, y;
	i = 0;
	 for(x=0; x<1;x++)
		for(y=0; y<1; y++)
			for (j=0; j<2048; j++) {
				O[i] = I[j][x][y];
				i++;
			}
}

void SW_fc1000(DATA_T I[2048], DATA_T O[1000], DATA_T W[1000][2048], DATA_T B[1000])
{
    //Dense
	int m, c;
	for(m=0; m<1000; m++){
        O[m] = 0;
		for (c = 0; c < 2048; c++){
            O[m] += W[m][c] * I[c];
        }
        O[m] += B[m];
        if (O[m] < 0) //Relu
            O[m] = 0;
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
	static DATA_T W18[64][256][1][1];
	static DATA_T B18[64];
	static DATA_T W19[4][64];
	static DATA_T W21[64][64][3][3];
	static DATA_T B21[64];
	static DATA_T W22[4][64];
	static DATA_T W24[256][64][1][1];
	static DATA_T B24[256];
	static DATA_T W25[4][256];
	static DATA_T W28[64][256][1][1];
	static DATA_T B28[64];
	static DATA_T W29[4][64];
	static DATA_T W31[64][64][3][3];
	static DATA_T B31[64];
	static DATA_T W32[4][64];
	static DATA_T W34[256][64][1][1];
	static DATA_T B34[256];
	static DATA_T W35[4][256];
	static DATA_T W38[128][256][1][1];
	static DATA_T B38[128];
	static DATA_T W39[4][128];
	static DATA_T W41[128][128][3][3];
	static DATA_T B41[128];
	static DATA_T W42[4][128];
	static DATA_T W44[512][128][1][1];
	static DATA_T B44[512];
	static DATA_T W45[512][256][1][1];
	static DATA_T B45[512];
	static DATA_T W46[4][512];
	static DATA_T W47[4][512];
	static DATA_T W50[128][512][1][1];
	static DATA_T B50[128];
	static DATA_T W51[4][128];
	static DATA_T W53[128][128][3][3];
	static DATA_T B53[128];
	static DATA_T W54[4][128];
	static DATA_T W56[512][128][1][1];
	static DATA_T B56[512];
	static DATA_T W57[4][512];
	static DATA_T W60[128][512][1][1];
	static DATA_T B60[128];
	static DATA_T W61[4][128];
	static DATA_T W63[128][128][3][3];
	static DATA_T B63[128];
	static DATA_T W64[4][128];
	static DATA_T W66[512][128][1][1];
	static DATA_T B66[512];
	static DATA_T W67[4][512];
	static DATA_T W70[128][512][1][1];
	static DATA_T B70[128];
	static DATA_T W71[4][128];
	static DATA_T W73[128][128][3][3];
	static DATA_T B73[128];
	static DATA_T W74[4][128];
	static DATA_T W76[512][128][1][1];
	static DATA_T B76[512];
	static DATA_T W77[4][512];
	static DATA_T W80[256][512][1][1];
	static DATA_T B80[256];
	static DATA_T W81[4][256];
	static DATA_T W83[256][256][3][3];
	static DATA_T B83[256];
	static DATA_T W84[4][256];
	static DATA_T W86[1024][256][1][1];
	static DATA_T B86[1024];
	static DATA_T W87[1024][512][1][1];
	static DATA_T B87[1024];
	static DATA_T W88[4][1024];
	static DATA_T W89[4][1024];
	static DATA_T W92[256][1024][1][1];
	static DATA_T B92[256];
	static DATA_T W93[4][256];
	static DATA_T W95[256][256][3][3];
	static DATA_T B95[256];
	static DATA_T W96[4][256];
	static DATA_T W98[1024][256][1][1];
	static DATA_T B98[1024];
	static DATA_T W99[4][1024];
	static DATA_T W102[256][1024][1][1];
	static DATA_T B102[256];
	static DATA_T W103[4][256];
	static DATA_T W105[256][256][3][3];
	static DATA_T B105[256];
	static DATA_T W106[4][256];
	static DATA_T W108[1024][256][1][1];
	static DATA_T B108[1024];
	static DATA_T W109[4][1024];
	static DATA_T W112[256][1024][1][1];
	static DATA_T B112[256];
	static DATA_T W113[4][256];
	static DATA_T W115[256][256][3][3];
	static DATA_T B115[256];
	static DATA_T W116[4][256];
	static DATA_T W118[1024][256][1][1];
	static DATA_T B118[1024];
	static DATA_T W119[4][1024];
	static DATA_T W122[256][1024][1][1];
	static DATA_T B122[256];
	static DATA_T W123[4][256];
	static DATA_T W125[256][256][3][3];
	static DATA_T B125[256];
	static DATA_T W126[4][256];
	static DATA_T W128[1024][256][1][1];
	static DATA_T B128[1024];
	static DATA_T W129[4][1024];
	static DATA_T W132[256][1024][1][1];
	static DATA_T B132[256];
	static DATA_T W133[4][256];
	static DATA_T W135[256][256][3][3];
	static DATA_T B135[256];
	static DATA_T W136[4][256];
	static DATA_T W138[1024][256][1][1];
	static DATA_T B138[1024];
	static DATA_T W139[4][1024];
	static DATA_T W142[512][1024][1][1];
	static DATA_T B142[512];
	static DATA_T W143[4][512];
	static DATA_T W145[512][512][3][3];
	static DATA_T B145[512];
	static DATA_T W146[4][512];
	static DATA_T W148[2048][512][1][1];
	static DATA_T B148[2048];
	static DATA_T W149[2048][1024][1][1];
	static DATA_T B149[2048];
	static DATA_T W150[4][2048];
	static DATA_T W151[4][2048];
	static DATA_T W154[512][2048][1][1];
	static DATA_T B154[512];
	static DATA_T W155[4][512];
	static DATA_T W157[512][512][3][3];
	static DATA_T B157[512];
	static DATA_T W158[4][512];
	static DATA_T W160[2048][512][1][1];
	static DATA_T B160[2048];
	static DATA_T W161[4][2048];
	static DATA_T W164[512][2048][1][1];
	static DATA_T B164[512];
	static DATA_T W165[4][512];
	static DATA_T W167[512][512][3][3];
	static DATA_T B167[512];
	static DATA_T W168[4][512];
	static DATA_T W170[2048][512][1][1];
	static DATA_T B170[2048];
	static DATA_T W171[4][2048];
	static DATA_T B176[1000];
	static DATA_T W176[1000][2048];
	

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
	static DATA_T O17_SW[256][55][55];
	static DATA_T O18_SW[64][55][55];
	static DATA_T O19_SW[64][55][55];
	static DATA_T O20_SW[64][55][55];
	static DATA_T O21_SW[64][55][55];
	static DATA_T O22_SW[64][55][55];
	static DATA_T O23_SW[64][55][55];
	static DATA_T O24_SW[256][55][55];
	static DATA_T O25_SW[256][55][55];
	static DATA_T O26_SW[256][55][55];
	static DATA_T O27_SW[256][55][55];
	static DATA_T O28_SW[64][55][55];
	static DATA_T O29_SW[64][55][55];
	static DATA_T O30_SW[64][55][55];
	static DATA_T O31_SW[64][55][55];
	static DATA_T O32_SW[64][55][55];
	static DATA_T O33_SW[64][55][55];
	static DATA_T O34_SW[256][55][55];
	static DATA_T O35_SW[256][55][55];
	static DATA_T O36_SW[256][55][55];
	static DATA_T O37_SW[256][55][55];
	static DATA_T O38_SW[128][28][28];
	static DATA_T O39_SW[128][28][28];
	static DATA_T O40_SW[128][28][28];
	static DATA_T O41_SW[128][28][28];
	static DATA_T O42_SW[128][28][28];
	static DATA_T O43_SW[128][28][28];
	static DATA_T O44_SW[512][28][28];
	static DATA_T O45_SW[512][28][28];
	static DATA_T O46_SW[512][28][28];
	static DATA_T O47_SW[512][28][28];
	static DATA_T O48_SW[512][28][28];
	static DATA_T O49_SW[512][28][28];
	static DATA_T O50_SW[128][28][28];
	static DATA_T O51_SW[128][28][28];
	static DATA_T O52_SW[128][28][28];
	static DATA_T O53_SW[128][28][28];
	static DATA_T O54_SW[128][28][28];
	static DATA_T O55_SW[128][28][28];
	static DATA_T O56_SW[512][28][28];
	static DATA_T O57_SW[512][28][28];
	static DATA_T O58_SW[512][28][28];
	static DATA_T O59_SW[512][28][28];
	static DATA_T O60_SW[128][28][28];
	static DATA_T O61_SW[128][28][28];
	static DATA_T O62_SW[128][28][28];
	static DATA_T O63_SW[128][28][28];
	static DATA_T O64_SW[128][28][28];
	static DATA_T O65_SW[128][28][28];
	static DATA_T O66_SW[512][28][28];
	static DATA_T O67_SW[512][28][28];
	static DATA_T O68_SW[512][28][28];
	static DATA_T O69_SW[512][28][28];
	static DATA_T O70_SW[128][28][28];
	static DATA_T O71_SW[128][28][28];
	static DATA_T O72_SW[128][28][28];
	static DATA_T O73_SW[128][28][28];
	static DATA_T O74_SW[128][28][28];
	static DATA_T O75_SW[128][28][28];
	static DATA_T O76_SW[512][28][28];
	static DATA_T O77_SW[512][28][28];
	static DATA_T O78_SW[512][28][28];
	static DATA_T O79_SW[512][28][28];
	static DATA_T O80_SW[256][14][14];
	static DATA_T O81_SW[256][14][14];
	static DATA_T O82_SW[256][14][14];
	static DATA_T O83_SW[256][14][14];
	static DATA_T O84_SW[256][14][14];
	static DATA_T O85_SW[256][14][14];
	static DATA_T O86_SW[1024][14][14];
	static DATA_T O87_SW[1024][14][14];
	static DATA_T O88_SW[1024][14][14];
	static DATA_T O89_SW[1024][14][14];
	static DATA_T O90_SW[1024][14][14];
	static DATA_T O91_SW[1024][14][14];
	static DATA_T O92_SW[256][14][14];
	static DATA_T O93_SW[256][14][14];
	static DATA_T O94_SW[256][14][14];
	static DATA_T O95_SW[256][14][14];
	static DATA_T O96_SW[256][14][14];
	static DATA_T O97_SW[256][14][14];
	static DATA_T O98_SW[1024][14][14];
	static DATA_T O99_SW[1024][14][14];
	static DATA_T O100_SW[1024][14][14];
	static DATA_T O101_SW[1024][14][14];
	static DATA_T O102_SW[256][14][14];
	static DATA_T O103_SW[256][14][14];
	static DATA_T O104_SW[256][14][14];
	static DATA_T O105_SW[256][14][14];
	static DATA_T O106_SW[256][14][14];
	static DATA_T O107_SW[256][14][14];
	static DATA_T O108_SW[1024][14][14];
	static DATA_T O109_SW[1024][14][14];
	static DATA_T O110_SW[1024][14][14];
	static DATA_T O111_SW[1024][14][14];
	static DATA_T O112_SW[256][14][14];
	static DATA_T O113_SW[256][14][14];
	static DATA_T O114_SW[256][14][14];
	static DATA_T O115_SW[256][14][14];
	static DATA_T O116_SW[256][14][14];
	static DATA_T O117_SW[256][14][14];
	static DATA_T O118_SW[1024][14][14];
	static DATA_T O119_SW[1024][14][14];
	static DATA_T O120_SW[1024][14][14];
	static DATA_T O121_SW[1024][14][14];
	static DATA_T O122_SW[256][14][14];
	static DATA_T O123_SW[256][14][14];
	static DATA_T O124_SW[256][14][14];
	static DATA_T O125_SW[256][14][14];
	static DATA_T O126_SW[256][14][14];
	static DATA_T O127_SW[256][14][14];
	static DATA_T O128_SW[1024][14][14];
	static DATA_T O129_SW[1024][14][14];
	static DATA_T O130_SW[1024][14][14];
	static DATA_T O131_SW[1024][14][14];
	static DATA_T O132_SW[256][14][14];
	static DATA_T O133_SW[256][14][14];
	static DATA_T O134_SW[256][14][14];
	static DATA_T O135_SW[256][14][14];
	static DATA_T O136_SW[256][14][14];
	static DATA_T O137_SW[256][14][14];
	static DATA_T O138_SW[1024][14][14];
	static DATA_T O139_SW[1024][14][14];
	static DATA_T O140_SW[1024][14][14];
	static DATA_T O141_SW[1024][14][14];
	static DATA_T O142_SW[512][7][7];
	static DATA_T O143_SW[512][7][7];
	static DATA_T O144_SW[512][7][7];
	static DATA_T O145_SW[512][7][7];
	static DATA_T O146_SW[512][7][7];
	static DATA_T O147_SW[512][7][7];
	static DATA_T O148_SW[2048][7][7];
	static DATA_T O149_SW[2048][7][7];
	static DATA_T O150_SW[2048][7][7];
	static DATA_T O151_SW[2048][7][7];
	static DATA_T O152_SW[2048][7][7];
	static DATA_T O153_SW[2048][7][7];
	static DATA_T O154_SW[512][7][7];
	static DATA_T O155_SW[512][7][7];
	static DATA_T O156_SW[512][7][7];
	static DATA_T O157_SW[512][7][7];
	static DATA_T O158_SW[512][7][7];
	static DATA_T O159_SW[512][7][7];
	static DATA_T O160_SW[2048][7][7];
	static DATA_T O161_SW[2048][7][7];
	static DATA_T O162_SW[2048][7][7];
	static DATA_T O163_SW[2048][7][7];
	static DATA_T O164_SW[512][7][7];
	static DATA_T O165_SW[512][7][7];
	static DATA_T O166_SW[512][7][7];
	static DATA_T O167_SW[512][7][7];
	static DATA_T O168_SW[512][7][7];
	static DATA_T O169_SW[512][7][7];
	static DATA_T O170_SW[2048][7][7];
	static DATA_T O171_SW[2048][7][7];
	static DATA_T O172_SW[2048][7][7];
	static DATA_T O173_SW[2048][7][7];
	static DATA_T O174_SW[2048][1][1];
	static DATA_T O175_SW[2048];
	static DATA_T O176_SW[1000];
	

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
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W18[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B18[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W19[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W21[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B21[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W22[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W24[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B24[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W25[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W28[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B28[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W29[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W31[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B31[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W32[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W34[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B34[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W35[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W38[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B38[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W39[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W41[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B41[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W42[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W44[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B44[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W45[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B45[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 512 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W46[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 512 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W47[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W50[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B50[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W51[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W53[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B53[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W54[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W56[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B56[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 512 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W57[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W60[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B60[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W61[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W63[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B63[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W64[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W66[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B66[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 512 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W67[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W70[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B70[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W71[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W73[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B73[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W74[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W76[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B76[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 512 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W77[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W80[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B80[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W81[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W83[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B83[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W84[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 1024 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W86[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 1024 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B86[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 1024 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W87[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 1024 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B87[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1024 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W88[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1024 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W89[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1024 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W92[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B92[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W93[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W95[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B95[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W96[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 1024 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W98[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 1024 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B98[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1024 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W99[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1024 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W102[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B102[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W103[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W105[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B105[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W106[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 1024 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W108[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 1024 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B108[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1024 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W109[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1024 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W112[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B112[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W113[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W115[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B115[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W116[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 1024 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W118[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 1024 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B118[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1024 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W119[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1024 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W122[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B122[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W123[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W125[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B125[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W126[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 1024 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W128[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 1024 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B128[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1024 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W129[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1024 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W132[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B132[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W133[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W135[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B135[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W136[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 1024 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W138[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 1024 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B138[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1024 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W139[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1024 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W142[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B142[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 512 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W143[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W145[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B145[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 512 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W146[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 2048 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W148[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 2048 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B148[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1024 ; i++) {
			for (j = 0; j < 2048 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W149[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 2048 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B149[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 2048 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W150[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 2048 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W151[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 2048 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W154[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B154[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 512 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W155[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W157[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B157[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 512 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W158[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 2048 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W160[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 2048 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B160[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 2048 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W161[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 2048 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W164[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B164[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 512 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W165[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W167[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B167[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 512 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W168[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 2048 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W170[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 2048 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B170[m] = (DATA_T) trash;
}


	for (x = 0; x < 4; x++) {
    for (y = 0; y < 2048 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W171[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  2048 ; m++) {
	for (k = 0; k < 1000 ; k++) {
		fread(&trash, sizeof(int), 1, w_stream);
        W176[k][m] = (DATA_T) trash;
	}
}

for (m = 0; m < 1000 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B176[m] = (DATA_T) trash;
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
	printf("[C_verifier.cpp]Calculate Activation(Relu)17\n\n");
	SW_activation_4(O16_SW,O17_SW);
	printf("[C_verifier.cpp]Calculate Conv2D18\n\n");
	SW_res2b_branch2a(O17_SW,O18_SW,W18,B18);
	printf("[C_verifier.cpp]Calculate BatchNormalization19\n\n");
	SW_bn2b_branch2a(O18_SW,O19_SW, W19);
	printf("[C_verifier.cpp]Calculate Activation(Relu)20\n\n");
	SW_activation_5(O19_SW,O20_SW);
	printf("[C_verifier.cpp]Calculate Conv2D21\n\n");
	SW_res2b_branch2b(O20_SW,O21_SW,W21,B21);
	printf("[C_verifier.cpp]Calculate BatchNormalization22\n\n");
	SW_bn2b_branch2b(O21_SW,O22_SW, W22);
	printf("[C_verifier.cpp]Calculate Activation(Relu)23\n\n");
	SW_activation_6(O22_SW,O23_SW);
	printf("[C_verifier.cpp]Calculate Conv2D24\n\n");
	SW_res2b_branch2c(O23_SW,O24_SW,W24,B24);
	printf("[C_verifier.cpp]Calculate BatchNormalization25\n\n");
	SW_bn2b_branch2c(O24_SW,O25_SW, W25);
	printf("[C_verifier.cpp]Calculate Add26\n\n");
	SW_add_2(O25_SW,O17_SW,O26_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)27\n\n");
	SW_activation_7(O26_SW,O27_SW);
	printf("[C_verifier.cpp]Calculate Conv2D28\n\n");
	SW_res2c_branch2a(O27_SW,O28_SW,W28,B28);
	printf("[C_verifier.cpp]Calculate BatchNormalization29\n\n");
	SW_bn2c_branch2a(O28_SW,O29_SW, W29);
	printf("[C_verifier.cpp]Calculate Activation(Relu)30\n\n");
	SW_activation_8(O29_SW,O30_SW);
	printf("[C_verifier.cpp]Calculate Conv2D31\n\n");
	SW_res2c_branch2b(O30_SW,O31_SW,W31,B31);
	printf("[C_verifier.cpp]Calculate BatchNormalization32\n\n");
	SW_bn2c_branch2b(O31_SW,O32_SW, W32);
	printf("[C_verifier.cpp]Calculate Activation(Relu)33\n\n");
	SW_activation_9(O32_SW,O33_SW);
	printf("[C_verifier.cpp]Calculate Conv2D34\n\n");
	SW_res2c_branch2c(O33_SW,O34_SW,W34,B34);
	printf("[C_verifier.cpp]Calculate BatchNormalization35\n\n");
	SW_bn2c_branch2c(O34_SW,O35_SW, W35);
	printf("[C_verifier.cpp]Calculate Add36\n\n");
	SW_add_3(O35_SW,O27_SW,O36_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)37\n\n");
	SW_activation_10(O36_SW,O37_SW);
	printf("[C_verifier.cpp]Calculate Conv2D38\n\n");
	SW_res3a_branch2a(O37_SW,O38_SW,W38,B38);
	printf("[C_verifier.cpp]Calculate BatchNormalization39\n\n");
	SW_bn3a_branch2a(O38_SW,O39_SW, W39);
	printf("[C_verifier.cpp]Calculate Activation(Relu)40\n\n");
	SW_activation_11(O39_SW,O40_SW);
	printf("[C_verifier.cpp]Calculate Conv2D41\n\n");
	SW_res3a_branch2b(O40_SW,O41_SW,W41,B41);
	printf("[C_verifier.cpp]Calculate BatchNormalization42\n\n");
	SW_bn3a_branch2b(O41_SW,O42_SW, W42);
	printf("[C_verifier.cpp]Calculate Activation(Relu)43\n\n");
	SW_activation_12(O42_SW,O43_SW);
	printf("[C_verifier.cpp]Calculate Conv2D44\n\n");
	SW_res3a_branch2c(O43_SW,O44_SW,W44,B44);
	printf("[C_verifier.cpp]Calculate Conv2D45\n\n");
	SW_res3a_branch1(O37_SW,O45_SW,W45,B45);
	printf("[C_verifier.cpp]Calculate BatchNormalization46\n\n");
	SW_bn3a_branch2c(O44_SW,O46_SW, W46);
	printf("[C_verifier.cpp]Calculate BatchNormalization47\n\n");
	SW_bn3a_branch1(O45_SW,O47_SW, W47);
	printf("[C_verifier.cpp]Calculate Add48\n\n");
	SW_add_4(O46_SW,O47_SW,O48_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)49\n\n");
	SW_activation_13(O48_SW,O49_SW);
	printf("[C_verifier.cpp]Calculate Conv2D50\n\n");
	SW_res3b_branch2a(O49_SW,O50_SW,W50,B50);
	printf("[C_verifier.cpp]Calculate BatchNormalization51\n\n");
	SW_bn3b_branch2a(O50_SW,O51_SW, W51);
	printf("[C_verifier.cpp]Calculate Activation(Relu)52\n\n");
	SW_activation_14(O51_SW,O52_SW);
	printf("[C_verifier.cpp]Calculate Conv2D53\n\n");
	SW_res3b_branch2b(O52_SW,O53_SW,W53,B53);
	printf("[C_verifier.cpp]Calculate BatchNormalization54\n\n");
	SW_bn3b_branch2b(O53_SW,O54_SW, W54);
	printf("[C_verifier.cpp]Calculate Activation(Relu)55\n\n");
	SW_activation_15(O54_SW,O55_SW);
	printf("[C_verifier.cpp]Calculate Conv2D56\n\n");
	SW_res3b_branch2c(O55_SW,O56_SW,W56,B56);
	printf("[C_verifier.cpp]Calculate BatchNormalization57\n\n");
	SW_bn3b_branch2c(O56_SW,O57_SW, W57);
	printf("[C_verifier.cpp]Calculate Add58\n\n");
	SW_add_5(O57_SW,O49_SW,O58_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)59\n\n");
	SW_activation_16(O58_SW,O59_SW);
	printf("[C_verifier.cpp]Calculate Conv2D60\n\n");
	SW_res3c_branch2a(O59_SW,O60_SW,W60,B60);
	printf("[C_verifier.cpp]Calculate BatchNormalization61\n\n");
	SW_bn3c_branch2a(O60_SW,O61_SW, W61);
	printf("[C_verifier.cpp]Calculate Activation(Relu)62\n\n");
	SW_activation_17(O61_SW,O62_SW);
	printf("[C_verifier.cpp]Calculate Conv2D63\n\n");
	SW_res3c_branch2b(O62_SW,O63_SW,W63,B63);
	printf("[C_verifier.cpp]Calculate BatchNormalization64\n\n");
	SW_bn3c_branch2b(O63_SW,O64_SW, W64);
	printf("[C_verifier.cpp]Calculate Activation(Relu)65\n\n");
	SW_activation_18(O64_SW,O65_SW);
	printf("[C_verifier.cpp]Calculate Conv2D66\n\n");
	SW_res3c_branch2c(O65_SW,O66_SW,W66,B66);
	printf("[C_verifier.cpp]Calculate BatchNormalization67\n\n");
	SW_bn3c_branch2c(O66_SW,O67_SW, W67);
	printf("[C_verifier.cpp]Calculate Add68\n\n");
	SW_add_6(O67_SW,O59_SW,O68_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)69\n\n");
	SW_activation_19(O68_SW,O69_SW);
	printf("[C_verifier.cpp]Calculate Conv2D70\n\n");
	SW_res3d_branch2a(O69_SW,O70_SW,W70,B70);
	printf("[C_verifier.cpp]Calculate BatchNormalization71\n\n");
	SW_bn3d_branch2a(O70_SW,O71_SW, W71);
	printf("[C_verifier.cpp]Calculate Activation(Relu)72\n\n");
	SW_activation_20(O71_SW,O72_SW);
	printf("[C_verifier.cpp]Calculate Conv2D73\n\n");
	SW_res3d_branch2b(O72_SW,O73_SW,W73,B73);
	printf("[C_verifier.cpp]Calculate BatchNormalization74\n\n");
	SW_bn3d_branch2b(O73_SW,O74_SW, W74);
	printf("[C_verifier.cpp]Calculate Activation(Relu)75\n\n");
	SW_activation_21(O74_SW,O75_SW);
	printf("[C_verifier.cpp]Calculate Conv2D76\n\n");
	SW_res3d_branch2c(O75_SW,O76_SW,W76,B76);
	printf("[C_verifier.cpp]Calculate BatchNormalization77\n\n");
	SW_bn3d_branch2c(O76_SW,O77_SW, W77);
	printf("[C_verifier.cpp]Calculate Add78\n\n");
	SW_add_7(O77_SW,O69_SW,O78_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)79\n\n");
	SW_activation_22(O78_SW,O79_SW);
	printf("[C_verifier.cpp]Calculate Conv2D80\n\n");
	SW_res4a_branch2a(O79_SW,O80_SW,W80,B80);
	printf("[C_verifier.cpp]Calculate BatchNormalization81\n\n");
	SW_bn4a_branch2a(O80_SW,O81_SW, W81);
	printf("[C_verifier.cpp]Calculate Activation(Relu)82\n\n");
	SW_activation_23(O81_SW,O82_SW);
	printf("[C_verifier.cpp]Calculate Conv2D83\n\n");
	SW_res4a_branch2b(O82_SW,O83_SW,W83,B83);
	printf("[C_verifier.cpp]Calculate BatchNormalization84\n\n");
	SW_bn4a_branch2b(O83_SW,O84_SW, W84);
	printf("[C_verifier.cpp]Calculate Activation(Relu)85\n\n");
	SW_activation_24(O84_SW,O85_SW);
	printf("[C_verifier.cpp]Calculate Conv2D86\n\n");
	SW_res4a_branch2c(O85_SW,O86_SW,W86,B86);
	printf("[C_verifier.cpp]Calculate Conv2D87\n\n");
	SW_res4a_branch1(O79_SW,O87_SW,W87,B87);
	printf("[C_verifier.cpp]Calculate BatchNormalization88\n\n");
	SW_bn4a_branch2c(O86_SW,O88_SW, W88);
	printf("[C_verifier.cpp]Calculate BatchNormalization89\n\n");
	SW_bn4a_branch1(O87_SW,O89_SW, W89);
	printf("[C_verifier.cpp]Calculate Add90\n\n");
	SW_add_8(O88_SW,O89_SW,O90_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)91\n\n");
	SW_activation_25(O90_SW,O91_SW);
	printf("[C_verifier.cpp]Calculate Conv2D92\n\n");
	SW_res4b_branch2a(O91_SW,O92_SW,W92,B92);
	printf("[C_verifier.cpp]Calculate BatchNormalization93\n\n");
	SW_bn4b_branch2a(O92_SW,O93_SW, W93);
	printf("[C_verifier.cpp]Calculate Activation(Relu)94\n\n");
	SW_activation_26(O93_SW,O94_SW);
	printf("[C_verifier.cpp]Calculate Conv2D95\n\n");
	SW_res4b_branch2b(O94_SW,O95_SW,W95,B95);
	printf("[C_verifier.cpp]Calculate BatchNormalization96\n\n");
	SW_bn4b_branch2b(O95_SW,O96_SW, W96);
	printf("[C_verifier.cpp]Calculate Activation(Relu)97\n\n");
	SW_activation_27(O96_SW,O97_SW);
	printf("[C_verifier.cpp]Calculate Conv2D98\n\n");
	SW_res4b_branch2c(O97_SW,O98_SW,W98,B98);
	printf("[C_verifier.cpp]Calculate BatchNormalization99\n\n");
	SW_bn4b_branch2c(O98_SW,O99_SW, W99);
	printf("[C_verifier.cpp]Calculate Add100\n\n");
	SW_add_9(O99_SW,O91_SW,O100_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)101\n\n");
	SW_activation_28(O100_SW,O101_SW);
	printf("[C_verifier.cpp]Calculate Conv2D102\n\n");
	SW_res4c_branch2a(O101_SW,O102_SW,W102,B102);
	printf("[C_verifier.cpp]Calculate BatchNormalization103\n\n");
	SW_bn4c_branch2a(O102_SW,O103_SW, W103);
	printf("[C_verifier.cpp]Calculate Activation(Relu)104\n\n");
	SW_activation_29(O103_SW,O104_SW);
	printf("[C_verifier.cpp]Calculate Conv2D105\n\n");
	SW_res4c_branch2b(O104_SW,O105_SW,W105,B105);
	printf("[C_verifier.cpp]Calculate BatchNormalization106\n\n");
	SW_bn4c_branch2b(O105_SW,O106_SW, W106);
	printf("[C_verifier.cpp]Calculate Activation(Relu)107\n\n");
	SW_activation_30(O106_SW,O107_SW);
	printf("[C_verifier.cpp]Calculate Conv2D108\n\n");
	SW_res4c_branch2c(O107_SW,O108_SW,W108,B108);
	printf("[C_verifier.cpp]Calculate BatchNormalization109\n\n");
	SW_bn4c_branch2c(O108_SW,O109_SW, W109);
	printf("[C_verifier.cpp]Calculate Add110\n\n");
	SW_add_10(O109_SW,O101_SW,O110_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)111\n\n");
	SW_activation_31(O110_SW,O111_SW);
	printf("[C_verifier.cpp]Calculate Conv2D112\n\n");
	SW_res4d_branch2a(O111_SW,O112_SW,W112,B112);
	printf("[C_verifier.cpp]Calculate BatchNormalization113\n\n");
	SW_bn4d_branch2a(O112_SW,O113_SW, W113);
	printf("[C_verifier.cpp]Calculate Activation(Relu)114\n\n");
	SW_activation_32(O113_SW,O114_SW);
	printf("[C_verifier.cpp]Calculate Conv2D115\n\n");
	SW_res4d_branch2b(O114_SW,O115_SW,W115,B115);
	printf("[C_verifier.cpp]Calculate BatchNormalization116\n\n");
	SW_bn4d_branch2b(O115_SW,O116_SW, W116);
	printf("[C_verifier.cpp]Calculate Activation(Relu)117\n\n");
	SW_activation_33(O116_SW,O117_SW);
	printf("[C_verifier.cpp]Calculate Conv2D118\n\n");
	SW_res4d_branch2c(O117_SW,O118_SW,W118,B118);
	printf("[C_verifier.cpp]Calculate BatchNormalization119\n\n");
	SW_bn4d_branch2c(O118_SW,O119_SW, W119);
	printf("[C_verifier.cpp]Calculate Add120\n\n");
	SW_add_11(O119_SW,O111_SW,O120_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)121\n\n");
	SW_activation_34(O120_SW,O121_SW);
	printf("[C_verifier.cpp]Calculate Conv2D122\n\n");
	SW_res4e_branch2a(O121_SW,O122_SW,W122,B122);
	printf("[C_verifier.cpp]Calculate BatchNormalization123\n\n");
	SW_bn4e_branch2a(O122_SW,O123_SW, W123);
	printf("[C_verifier.cpp]Calculate Activation(Relu)124\n\n");
	SW_activation_35(O123_SW,O124_SW);
	printf("[C_verifier.cpp]Calculate Conv2D125\n\n");
	SW_res4e_branch2b(O124_SW,O125_SW,W125,B125);
	printf("[C_verifier.cpp]Calculate BatchNormalization126\n\n");
	SW_bn4e_branch2b(O125_SW,O126_SW, W126);
	printf("[C_verifier.cpp]Calculate Activation(Relu)127\n\n");
	SW_activation_36(O126_SW,O127_SW);
	printf("[C_verifier.cpp]Calculate Conv2D128\n\n");
	SW_res4e_branch2c(O127_SW,O128_SW,W128,B128);
	printf("[C_verifier.cpp]Calculate BatchNormalization129\n\n");
	SW_bn4e_branch2c(O128_SW,O129_SW, W129);
	printf("[C_verifier.cpp]Calculate Add130\n\n");
	SW_add_12(O129_SW,O121_SW,O130_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)131\n\n");
	SW_activation_37(O130_SW,O131_SW);
	printf("[C_verifier.cpp]Calculate Conv2D132\n\n");
	SW_res4f_branch2a(O131_SW,O132_SW,W132,B132);
	printf("[C_verifier.cpp]Calculate BatchNormalization133\n\n");
	SW_bn4f_branch2a(O132_SW,O133_SW, W133);
	printf("[C_verifier.cpp]Calculate Activation(Relu)134\n\n");
	SW_activation_38(O133_SW,O134_SW);
	printf("[C_verifier.cpp]Calculate Conv2D135\n\n");
	SW_res4f_branch2b(O134_SW,O135_SW,W135,B135);
	printf("[C_verifier.cpp]Calculate BatchNormalization136\n\n");
	SW_bn4f_branch2b(O135_SW,O136_SW, W136);
	printf("[C_verifier.cpp]Calculate Activation(Relu)137\n\n");
	SW_activation_39(O136_SW,O137_SW);
	printf("[C_verifier.cpp]Calculate Conv2D138\n\n");
	SW_res4f_branch2c(O137_SW,O138_SW,W138,B138);
	printf("[C_verifier.cpp]Calculate BatchNormalization139\n\n");
	SW_bn4f_branch2c(O138_SW,O139_SW, W139);
	printf("[C_verifier.cpp]Calculate Add140\n\n");
	SW_add_13(O139_SW,O131_SW,O140_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)141\n\n");
	SW_activation_40(O140_SW,O141_SW);
	printf("[C_verifier.cpp]Calculate Conv2D142\n\n");
	SW_res5a_branch2a(O141_SW,O142_SW,W142,B142);
	printf("[C_verifier.cpp]Calculate BatchNormalization143\n\n");
	SW_bn5a_branch2a(O142_SW,O143_SW, W143);
	printf("[C_verifier.cpp]Calculate Activation(Relu)144\n\n");
	SW_activation_41(O143_SW,O144_SW);
	printf("[C_verifier.cpp]Calculate Conv2D145\n\n");
	SW_res5a_branch2b(O144_SW,O145_SW,W145,B145);
	printf("[C_verifier.cpp]Calculate BatchNormalization146\n\n");
	SW_bn5a_branch2b(O145_SW,O146_SW, W146);
	printf("[C_verifier.cpp]Calculate Activation(Relu)147\n\n");
	SW_activation_42(O146_SW,O147_SW);
	printf("[C_verifier.cpp]Calculate Conv2D148\n\n");
	SW_res5a_branch2c(O147_SW,O148_SW,W148,B148);
	printf("[C_verifier.cpp]Calculate Conv2D149\n\n");
	SW_res5a_branch1(O141_SW,O149_SW,W149,B149);
	printf("[C_verifier.cpp]Calculate BatchNormalization150\n\n");
	SW_bn5a_branch2c(O148_SW,O150_SW, W150);
	printf("[C_verifier.cpp]Calculate BatchNormalization151\n\n");
	SW_bn5a_branch1(O149_SW,O151_SW, W151);
	printf("[C_verifier.cpp]Calculate Add152\n\n");
	SW_add_14(O150_SW,O151_SW,O152_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)153\n\n");
	SW_activation_43(O152_SW,O153_SW);
	printf("[C_verifier.cpp]Calculate Conv2D154\n\n");
	SW_res5b_branch2a(O153_SW,O154_SW,W154,B154);
	printf("[C_verifier.cpp]Calculate BatchNormalization155\n\n");
	SW_bn5b_branch2a(O154_SW,O155_SW, W155);
	printf("[C_verifier.cpp]Calculate Activation(Relu)156\n\n");
	SW_activation_44(O155_SW,O156_SW);
	printf("[C_verifier.cpp]Calculate Conv2D157\n\n");
	SW_res5b_branch2b(O156_SW,O157_SW,W157,B157);
	printf("[C_verifier.cpp]Calculate BatchNormalization158\n\n");
	SW_bn5b_branch2b(O157_SW,O158_SW, W158);
	printf("[C_verifier.cpp]Calculate Activation(Relu)159\n\n");
	SW_activation_45(O158_SW,O159_SW);
	printf("[C_verifier.cpp]Calculate Conv2D160\n\n");
	SW_res5b_branch2c(O159_SW,O160_SW,W160,B160);
	printf("[C_verifier.cpp]Calculate BatchNormalization161\n\n");
	SW_bn5b_branch2c(O160_SW,O161_SW, W161);
	printf("[C_verifier.cpp]Calculate Add162\n\n");
	SW_add_15(O161_SW,O153_SW,O162_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)163\n\n");
	SW_activation_46(O162_SW,O163_SW);
	printf("[C_verifier.cpp]Calculate Conv2D164\n\n");
	SW_res5c_branch2a(O163_SW,O164_SW,W164,B164);
	printf("[C_verifier.cpp]Calculate BatchNormalization165\n\n");
	SW_bn5c_branch2a(O164_SW,O165_SW, W165);
	printf("[C_verifier.cpp]Calculate Activation(Relu)166\n\n");
	SW_activation_47(O165_SW,O166_SW);
	printf("[C_verifier.cpp]Calculate Conv2D167\n\n");
	SW_res5c_branch2b(O166_SW,O167_SW,W167,B167);
	printf("[C_verifier.cpp]Calculate BatchNormalization168\n\n");
	SW_bn5c_branch2b(O167_SW,O168_SW, W168);
	printf("[C_verifier.cpp]Calculate Activation(Relu)169\n\n");
	SW_activation_48(O168_SW,O169_SW);
	printf("[C_verifier.cpp]Calculate Conv2D170\n\n");
	SW_res5c_branch2c(O169_SW,O170_SW,W170,B170);
	printf("[C_verifier.cpp]Calculate BatchNormalization171\n\n");
	SW_bn5c_branch2c(O170_SW,O171_SW, W171);
	printf("[C_verifier.cpp]Calculate Add172\n\n");
	SW_add_16(O171_SW,O163_SW,O172_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)173\n\n");
	SW_activation_49(O172_SW,O173_SW);
	printf("[C_verifier.cpp]Calculate AveragePooling2D174\n\n");
	SW_avg_pool(O173_SW,O174_SW);
	printf("[C_verifier.py]Calculate Flatten175\n\n");
	SW_flatten_1(O174_SW,O175_SW);
	printf("[C_verifier.cpp]Calculate Dense176\n\n");
	SW_fc1000(O175_SW,O176_SW,W176,B176);
	

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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 55 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 55 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O17_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O17_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O18_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O18_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O19_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O19_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O20_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O20_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O21_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O21_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O22_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O22_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O23_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O23_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O24_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O24_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O25_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O25_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O26_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O26_SW[y][k][x]);
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
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O27_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O27_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O28_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O28_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O29_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O29_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O30_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O30_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O31_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O31_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O32_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O32_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O33_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O33_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O34_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O34_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O35_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O35_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O36_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O36_SW[y][k][x]);
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
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O37_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O37_SW[y][k][x]);
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
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O38_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O38_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O39_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O39_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O40_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O40_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O41_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O41_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O42_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O42_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O43_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O43_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O44_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O44_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O45_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O45_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O46_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O46_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O47_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O47_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O48_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O48_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O49_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O49_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O50_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O50_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O51_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O51_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O52_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O52_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O53_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O53_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O54_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O54_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O55_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O55_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O56_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O56_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O57_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O57_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O58_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O58_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O59_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O59_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O60_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O60_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O61_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O61_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O62_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O62_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O63_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O63_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O64_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O64_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O65_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O65_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O66_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O66_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O67_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O67_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O68_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O68_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O69_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O69_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O70_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O70_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O71_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O71_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O72_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O72_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O73_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O73_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O74_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O74_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O75_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O75_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O76_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O76_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O77_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O77_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O78_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O78_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O79_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O79_SW[y][k][x]);
		}
		if(x != 28 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 28 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O80_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O80_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O81_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O81_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O82_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O82_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O83_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O83_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O84_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O84_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O85_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O85_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O86_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O86_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O87_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O87_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O88_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O88_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O89_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O89_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O90_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O90_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O91_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O91_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O92_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O92_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O93_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O93_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O94_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O94_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O95_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O95_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O96_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O96_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O97_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O97_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O98_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O98_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O99_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O99_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O100_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O100_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O101_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O101_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O102_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O102_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O103_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O103_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O104_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O104_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O105_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O105_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O106_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O106_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O107_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O107_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O108_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O108_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O109_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O109_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O110_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O110_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O111_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O111_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O112_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O112_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O113_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O113_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O114_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O114_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O115_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O115_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O116_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O116_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O117_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O117_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O118_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O118_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O119_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O119_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O120_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O120_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O121_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O121_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O122_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O122_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O123_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O123_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O124_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O124_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O125_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O125_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O126_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O126_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O127_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O127_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O128_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O128_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O129_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O129_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O130_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O130_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O131_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O131_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O132_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O132_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O133_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O133_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O134_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O134_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O135_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O135_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O136_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O136_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O137_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O137_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O138_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O138_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O139_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O139_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O140_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O140_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O141_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O141_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O142_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O142_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O143_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O143_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O144_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O144_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O145_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O145_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O146_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O146_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O147_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O147_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2048 ; y++) {
			fprintf(o_stream,"%.6f ",O148_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O148_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2048 ; y++) {
			fprintf(o_stream,"%.6f ",O149_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O149_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2048 ; y++) {
			fprintf(o_stream,"%.6f ",O150_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O150_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2048 ; y++) {
			fprintf(o_stream,"%.6f ",O151_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O151_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2048 ; y++) {
			fprintf(o_stream,"%.6f ",O152_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O152_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2048 ; y++) {
			fprintf(o_stream,"%.6f ",O153_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O153_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O154_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O154_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O155_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O155_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O156_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O156_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O157_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O157_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O158_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O158_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O159_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O159_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2048 ; y++) {
			fprintf(o_stream,"%.6f ",O160_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O160_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2048 ; y++) {
			fprintf(o_stream,"%.6f ",O161_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O161_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2048 ; y++) {
			fprintf(o_stream,"%.6f ",O162_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O162_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2048 ; y++) {
			fprintf(o_stream,"%.6f ",O163_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O163_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O164_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O164_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O165_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O165_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O166_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O166_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O167_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O167_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O168_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O168_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O169_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O169_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2048 ; y++) {
			fprintf(o_stream,"%.6f ",O170_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O170_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2048 ; y++) {
			fprintf(o_stream,"%.6f ",O171_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O171_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2048 ; y++) {
			fprintf(o_stream,"%.6f ",O172_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O172_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2048 ; y++) {
			fprintf(o_stream,"%.6f ",O173_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O173_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","AveragePooling2D : [[");
for (k = 0; k < 1 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 1 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2048 ; y++) {
			fprintf(o_stream,"%.6f ",O174_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O174_SW[y][k][x]);
		}
		if(x != 1 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 1 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Flatten : [[");
for (k = 0; k <  2048 ; k++) {
	fprintf(o_stream,"%.6f ",O175_SW[k]);
	fprintf(c_num,"%.6f ",O175_SW[k]);
}
fprintf(o_stream,"%s","]]\n\n");


fprintf(o_stream,"%s","Dense : [[");
for (k = 0; k <  1000 ; k++) {
	fprintf(o_stream,"%.6f ",O176_SW[k]);
	fprintf(c_num,"%.6f ",O176_SW[k]);
}
fprintf(o_stream,"%s","]]\n\n");





    fclose(w_stream);
    fclose(i_stream);
    fclose(o_stream);
    fclose(c_num);

    return 0;
}
