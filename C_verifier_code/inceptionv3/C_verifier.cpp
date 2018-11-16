#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <math.h>
    
using namespace std;

typedef float DATA_T;

void SW_conv2d_1(DATA_T I[3][224][224], DATA_T O[32][111][111], DATA_T W[32][3][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<32; m++) {
		for (x = 0; x<111; x++) {
			for (y = 0; y<111; y++) {
				ofm = 0;
				for (k = 0; k<3; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i <= 224 && y + j <= 224) {
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

void SW_batch_normalization_1(DATA_T I[32][111][111], DATA_T O[32][111][111], DATA_T W[3][32]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 32; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 111; x++){
			for (y = 0; y < 111; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_50(DATA_T I[32][111][111], DATA_T O[32][111][111])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<32; m++) {
		for (x = 0; x<111; x++) {
			for (y = 0; y<111; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_conv2d_2(DATA_T I[32][111][111], DATA_T O[32][109][109], DATA_T W[32][32][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<32; m++) {
		for (x = 0; x<109; x++) {
			for (y = 0; y<109; y++) {
				ofm = 0;
				for (k = 0; k<32; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i <= 111 && y + j <= 111) {
								ifm = I[k][x*1 + i][y*1 + j];
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

void SW_batch_normalization_2(DATA_T I[32][109][109], DATA_T O[32][109][109], DATA_T W[3][32]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 32; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 109; x++){
			for (y = 0; y < 109; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_51(DATA_T I[32][109][109], DATA_T O[32][109][109])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<32; m++) {
		for (x = 0; x<109; x++) {
			for (y = 0; y<109; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_conv2d_3(DATA_T I[32][109][109], DATA_T O[64][109][109], DATA_T W[64][32][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(109 - 1) - 109 + 3)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<109; x++) {
			for (y = 0; y<109; y++) {
				ofm = 0;
				for (k = 0; k<32; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 109 + p && y + j < 109 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_batch_normalization_3(DATA_T I[64][109][109], DATA_T O[64][109][109], DATA_T W[3][64]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 64; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 109; x++){
			for (y = 0; y < 109; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_52(DATA_T I[64][109][109], DATA_T O[64][109][109])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<64; m++) {
		for (x = 0; x<109; x++) {
			for (y = 0; y<109; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_max_pooling2d_2(DATA_T I[64][109][109], DATA_T O[64][54][54])
{
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<64; m++) {
		for (x = 0; x<54; x++) {
			for (y = 0; y<54; y++) {
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
void SW_conv2d_4(DATA_T I[64][54][54], DATA_T O[80][54][54], DATA_T W[80][64][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<80; m++) {
		for (x = 0; x<54; x++) {
			for (y = 0; y<54; y++) {
				ofm = 0;
				for (k = 0; k<64; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 54 && y + j <= 54) {
								ifm = I[k][x*1 + i][y*1 + j];
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

void SW_batch_normalization_4(DATA_T I[80][54][54], DATA_T O[80][54][54], DATA_T W[3][80]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 80; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 54; x++){
			for (y = 0; y < 54; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_53(DATA_T I[80][54][54], DATA_T O[80][54][54])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<80; m++) {
		for (x = 0; x<54; x++) {
			for (y = 0; y<54; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_conv2d_5(DATA_T I[80][54][54], DATA_T O[192][52][52], DATA_T W[192][80][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<192; m++) {
		for (x = 0; x<52; x++) {
			for (y = 0; y<52; y++) {
				ofm = 0;
				for (k = 0; k<80; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i <= 54 && y + j <= 54) {
								ifm = I[k][x*1 + i][y*1 + j];
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

void SW_batch_normalization_5(DATA_T I[192][52][52], DATA_T O[192][52][52], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 52; x++){
			for (y = 0; y < 52; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_54(DATA_T I[192][52][52], DATA_T O[192][52][52])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<192; m++) {
		for (x = 0; x<52; x++) {
			for (y = 0; y<52; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_max_pooling2d_3(DATA_T I[192][52][52], DATA_T O[192][25][25])
{
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<192; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
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
void SW_conv2d_9(DATA_T I[192][25][25], DATA_T O[64][25][25], DATA_T W[64][192][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<192; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 25 + p && y + j < 25 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_batch_normalization_9(DATA_T I[64][25][25], DATA_T O[64][25][25], DATA_T W[3][64]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 64; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 25; x++){
			for (y = 0; y < 25; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_58(DATA_T I[64][25][25], DATA_T O[64][25][25])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_conv2d_7(DATA_T I[192][25][25], DATA_T O[48][25][25], DATA_T W[48][192][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<48; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<192; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 25 + p && y + j < 25 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_conv2d_10(DATA_T I[64][25][25], DATA_T O[96][25][25], DATA_T W[96][64][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 3)/2;
	for (m = 0; m<96; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<64; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 25 + p && y + j < 25 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_batch_normalization_7(DATA_T I[48][25][25], DATA_T O[48][25][25], DATA_T W[3][48]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 48; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 25; x++){
			for (y = 0; y < 25; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_10(DATA_T I[96][25][25], DATA_T O[96][25][25], DATA_T W[3][96]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 96; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 25; x++){
			for (y = 0; y < 25; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_56(DATA_T I[48][25][25], DATA_T O[48][25][25])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<48; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_activation_59(DATA_T I[96][25][25], DATA_T O[96][25][25])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<96; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_average_pooling2d_1(DATA_T I[192][25][25], DATA_T O[192][25][25])
{
	int m, x, y, i, j;
	DATA_T sum;
	int p = (1*(25-1) - 25 + 3)/2;
    DATA_T div = (DATA_T)(3*3);
	for (m = 0; m<192; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
                sum = (DATA_T)0;
				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {

						if (x + i < 25 + p && y + j < 25 + p && x + i -p >= 0 && y + j -p >= 0)
						    sum += I[m][x*1 + i][y*1 + j];
					}
				}
				O[m][x][y] = sum;
			}
		}
	}
}
void SW_conv2d_6(DATA_T I[192][25][25], DATA_T O[64][25][25], DATA_T W[64][192][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<192; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 25 + p && y + j < 25 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_conv2d_8(DATA_T I[48][25][25], DATA_T O[64][25][25], DATA_T W[64][48][5][5]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 5)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<48; k++) {
					for (i = 0; i<5; i++) {
						for (j = 0; j<5; j++) {
							if (x + i < 25 + p && y + j < 25 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_conv2d_11(DATA_T I[96][25][25], DATA_T O[96][25][25], DATA_T W[96][96][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 3)/2;
	for (m = 0; m<96; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<96; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 25 + p && y + j < 25 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_conv2d_12(DATA_T I[192][25][25], DATA_T O[32][25][25], DATA_T W[32][192][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<192; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 25 + p && y + j < 25 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_batch_normalization_6(DATA_T I[64][25][25], DATA_T O[64][25][25], DATA_T W[3][64]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 64; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 25; x++){
			for (y = 0; y < 25; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_8(DATA_T I[64][25][25], DATA_T O[64][25][25], DATA_T W[3][64]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 64; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 25; x++){
			for (y = 0; y < 25; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_11(DATA_T I[96][25][25], DATA_T O[96][25][25], DATA_T W[3][96]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 96; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 25; x++){
			for (y = 0; y < 25; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_12(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 32; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 25; x++){
			for (y = 0; y < 25; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_55(DATA_T I[64][25][25], DATA_T O[64][25][25])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_activation_57(DATA_T I[64][25][25], DATA_T O[64][25][25])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_activation_60(DATA_T I[96][25][25], DATA_T O[96][25][25])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<96; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_activation_61(DATA_T I[32][25][25], DATA_T O[32][25][25])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_mixed0(DATA_T I1[64][25][25], DATA_T I2[64][25][25], DATA_T I3[96][25][25], DATA_T I4[32][25][25], DATA_T O[256][25][25]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 25; x++) {
		for(y = 0; y < 25; y++) {
			ch=0;
			for(k = ch; k < ch+64; k++){
				O[x][y][k] = I1[x][y][k];
			}
			ch+=64;
			for(k = ch; k < ch+64; k++){
				O[x][y][k] = I2[x][y][k-ch];
			}
			ch+=64;
			for(k = ch; k < ch+96; k++){
				O[x][y][k] = I3[x][y][k-ch];
			}
			ch+=96;
			for(k = ch; k < ch+32; k++){
				O[x][y][k] = I4[x][y][k-ch];
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
	static DATA_T W2[3][32];
	static DATA_T W4[32][32][3][3];
	static DATA_T W5[3][32];
	static DATA_T W7[64][32][3][3];
	static DATA_T W8[3][64];
	static DATA_T W11[80][64][1][1];
	static DATA_T W12[3][80];
	static DATA_T W14[192][80][3][3];
	static DATA_T W15[3][192];
	static DATA_T W18[64][192][1][1];
	static DATA_T W19[3][64];
	static DATA_T W21[48][192][1][1];
	static DATA_T W22[96][64][3][3];
	static DATA_T W23[3][48];
	static DATA_T W24[3][96];
	static DATA_T W28[64][192][1][1];
	static DATA_T W29[64][48][5][5];
	static DATA_T W30[96][96][3][3];
	static DATA_T W31[32][192][1][1];
	static DATA_T W32[3][64];
	static DATA_T W33[3][64];
	static DATA_T W34[3][96];
	static DATA_T W35[3][32];
	

    static DATA_T O0_SW[3][224][224];
	static DATA_T O1_SW[32][111][111];
	static DATA_T O2_SW[32][111][111];
	static DATA_T O3_SW[32][111][111];
	static DATA_T O4_SW[32][109][109];
	static DATA_T O5_SW[32][109][109];
	static DATA_T O6_SW[32][109][109];
	static DATA_T O7_SW[64][109][109];
	static DATA_T O8_SW[64][109][109];
	static DATA_T O9_SW[64][109][109];
	static DATA_T O10_SW[64][54][54];
	static DATA_T O11_SW[80][54][54];
	static DATA_T O12_SW[80][54][54];
	static DATA_T O13_SW[80][54][54];
	static DATA_T O14_SW[192][52][52];
	static DATA_T O15_SW[192][52][52];
	static DATA_T O16_SW[192][52][52];
	static DATA_T O17_SW[192][25][25];
	static DATA_T O18_SW[64][25][25];
	static DATA_T O19_SW[64][25][25];
	static DATA_T O20_SW[64][25][25];
	static DATA_T O21_SW[48][25][25];
	static DATA_T O22_SW[96][25][25];
	static DATA_T O23_SW[48][25][25];
	static DATA_T O24_SW[96][25][25];
	static DATA_T O25_SW[48][25][25];
	static DATA_T O26_SW[96][25][25];
	static DATA_T O27_SW[192][25][25];
	static DATA_T O28_SW[64][25][25];
	static DATA_T O29_SW[64][25][25];
	static DATA_T O30_SW[96][25][25];
	static DATA_T O31_SW[32][25][25];
	static DATA_T O32_SW[64][25][25];
	static DATA_T O33_SW[64][25][25];
	static DATA_T O34_SW[96][25][25];
	static DATA_T O35_SW[32][25][25];
	static DATA_T O36_SW[64][25][25];
	static DATA_T O37_SW[64][25][25];
	static DATA_T O38_SW[96][25][25];
	static DATA_T O39_SW[32][25][25];
	static DATA_T O40_SW[256][25][25];
	

    FILE *w_stream = fopen(argv[1], "rb");
    if (w_stream == NULL) printf("weight file was not opened");
    FILE *i_stream = fopen(argv[2], "rb");
    if (i_stream == NULL) printf("input file was not opened");
    FILE *o_stream = fopen("../../cpp_generator/inceptionv3/Output/C_output.txt", "w");
    if (o_stream == NULL) printf("Output file was not opened");
    FILE *c_num = fopen("../../cpp_generator/inceptionv3/Output/c_output_num.txt", "w");
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

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W2[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W4[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B4[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W5[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W7[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B7[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W8[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 80 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W11[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 80 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B11[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 80 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W12[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 80 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W14[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B14[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W15[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 192 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W18[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B18[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W19[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 192 ; i++) {
			for (j = 0; j < 48 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W21[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 48 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B21[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 96 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W22[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 96 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B22[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 48 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W23[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 96 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W24[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 192 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W28[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B28[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  5 ; m++) {
	for (k = 0; k < 5 ; k++) {
		for (i = 0; i < 48 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W29[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B29[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 96 ; i++) {
			for (j = 0; j < 96 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W30[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 96 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B30[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 192 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W31[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B31[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W32[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W33[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 96 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W34[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W35[x][y] = (DATA_T) trash;
    }
}
	
    printf("[C_verifier.cpp]Finish Initialization");

    printf("[C_verifier.cpp]InputLayer\n\n");
	printf("[C_verifier.cpp]Calculate Conv2D1\n\n");
	SW_conv2d_1(O0_SW,O1_SW,W1);
	printf("[C_verifier.cpp]Calculate BatchNormalization2\n\n");
	SW_batch_normalization_1(O1_SW,O2_SW, W2);
	printf("[C_verifier.cpp]Calculate Activation(Relu)3\n\n");
	SW_activation_50(O2_SW,O3_SW);
	printf("[C_verifier.cpp]Calculate Conv2D4\n\n");
	SW_conv2d_2(O3_SW,O4_SW,W4);
	printf("[C_verifier.cpp]Calculate BatchNormalization5\n\n");
	SW_batch_normalization_2(O4_SW,O5_SW, W5);
	printf("[C_verifier.cpp]Calculate Activation(Relu)6\n\n");
	SW_activation_51(O5_SW,O6_SW);
	printf("[C_verifier.cpp]Calculate Conv2D7\n\n");
	SW_conv2d_3(O6_SW,O7_SW,W7);
	printf("[C_verifier.cpp]Calculate BatchNormalization8\n\n");
	SW_batch_normalization_3(O7_SW,O8_SW, W8);
	printf("[C_verifier.cpp]Calculate Activation(Relu)9\n\n");
	SW_activation_52(O8_SW,O9_SW);
	printf("[C_verifier.cpp]Calculate MaxPooling2D10\n\n");
	SW_max_pooling2d_2(O9_SW,O10_SW);
	printf("[C_verifier.cpp]Calculate Conv2D11\n\n");
	SW_conv2d_4(O10_SW,O11_SW,W11);
	printf("[C_verifier.cpp]Calculate BatchNormalization12\n\n");
	SW_batch_normalization_4(O11_SW,O12_SW, W12);
	printf("[C_verifier.cpp]Calculate Activation(Relu)13\n\n");
	SW_activation_53(O12_SW,O13_SW);
	printf("[C_verifier.cpp]Calculate Conv2D14\n\n");
	SW_conv2d_5(O13_SW,O14_SW,W14);
	printf("[C_verifier.cpp]Calculate BatchNormalization15\n\n");
	SW_batch_normalization_5(O14_SW,O15_SW, W15);
	printf("[C_verifier.cpp]Calculate Activation(Relu)16\n\n");
	SW_activation_54(O15_SW,O16_SW);
	printf("[C_verifier.cpp]Calculate MaxPooling2D17\n\n");
	SW_max_pooling2d_3(O16_SW,O17_SW);
	printf("[C_verifier.cpp]Calculate Conv2D18\n\n");
	SW_conv2d_9(O17_SW,O18_SW,W18);
	printf("[C_verifier.cpp]Calculate BatchNormalization19\n\n");
	SW_batch_normalization_9(O18_SW,O19_SW, W19);
	printf("[C_verifier.cpp]Calculate Activation(Relu)20\n\n");
	SW_activation_58(O19_SW,O20_SW);
	printf("[C_verifier.cpp]Calculate Conv2D21\n\n");
	SW_conv2d_7(O17_SW,O21_SW,W21);
	printf("[C_verifier.cpp]Calculate Conv2D22\n\n");
	SW_conv2d_10(O20_SW,O22_SW,W22);
	printf("[C_verifier.cpp]Calculate BatchNormalization23\n\n");
	SW_batch_normalization_7(O21_SW,O23_SW, W23);
	printf("[C_verifier.cpp]Calculate BatchNormalization24\n\n");
	SW_batch_normalization_10(O22_SW,O24_SW, W24);
	printf("[C_verifier.cpp]Calculate Activation(Relu)25\n\n");
	SW_activation_56(O23_SW,O25_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)26\n\n");
	SW_activation_59(O24_SW,O26_SW);
	printf("[C_verifier.cpp]Calculate AveragePooling2D27\n\n");
	SW_average_pooling2d_1(O17_SW,O27_SW);
	printf("[C_verifier.cpp]Calculate Conv2D28\n\n");
	SW_conv2d_6(O17_SW,O28_SW,W28);
	printf("[C_verifier.cpp]Calculate Conv2D29\n\n");
	SW_conv2d_8(O25_SW,O29_SW,W29);
	printf("[C_verifier.cpp]Calculate Conv2D30\n\n");
	SW_conv2d_11(O26_SW,O30_SW,W30);
	printf("[C_verifier.cpp]Calculate Conv2D31\n\n");
	SW_conv2d_12(O27_SW,O31_SW,W31);
	printf("[C_verifier.cpp]Calculate BatchNormalization32\n\n");
	SW_batch_normalization_6(O28_SW,O32_SW, W32);
	printf("[C_verifier.cpp]Calculate BatchNormalization33\n\n");
	SW_batch_normalization_8(O29_SW,O33_SW, W33);
	printf("[C_verifier.cpp]Calculate BatchNormalization34\n\n");
	SW_batch_normalization_11(O30_SW,O34_SW, W34);
	printf("[C_verifier.cpp]Calculate BatchNormalization35\n\n");
	SW_batch_normalization_12(O31_SW,O35_SW, W35);
	printf("[C_verifier.cpp]Calculate Activation(Relu)36\n\n");
	SW_activation_55(O32_SW,O36_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)37\n\n");
	SW_activation_57(O33_SW,O37_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)38\n\n");
	SW_activation_60(O34_SW,O38_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)39\n\n");
	SW_activation_61(O35_SW,O39_SW);
	printf("[C_verifier.cpp]Calculate Concatenate40\n\n");
	SW_mixed0(O36_SW, O37_SW, O38_SW, O39_SW,O40_SW);
	

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
for (k = 0; k < 111 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 111 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O1_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O1_SW[y][k][x]);
		}
		if(x != 111 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 111 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 111 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 111 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O2_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O2_SW[y][k][x]);
		}
		if(x != 111 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 111 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 111 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 111 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O3_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O3_SW[y][k][x]);
		}
		if(x != 111 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 111 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 109 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 109 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O4_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O4_SW[y][k][x]);
		}
		if(x != 109 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 109 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 109 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 109 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O5_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O5_SW[y][k][x]);
		}
		if(x != 109 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 109 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 109 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 109 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O6_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O6_SW[y][k][x]);
		}
		if(x != 109 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 109 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 109 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 109 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O7_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O7_SW[y][k][x]);
		}
		if(x != 109 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 109 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 109 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 109 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O8_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O8_SW[y][k][x]);
		}
		if(x != 109 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 109 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 109 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 109 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O9_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O9_SW[y][k][x]);
		}
		if(x != 109 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 109 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","MaxPooling2D : [[");
for (k = 0; k < 54 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 54 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O10_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O10_SW[y][k][x]);
		}
		if(x != 54 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 54 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 54 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 54 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 80 ; y++) {
			fprintf(o_stream,"%.6f ",O11_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O11_SW[y][k][x]);
		}
		if(x != 54 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 54 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 54 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 54 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 80 ; y++) {
			fprintf(o_stream,"%.6f ",O12_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O12_SW[y][k][x]);
		}
		if(x != 54 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 54 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 54 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 54 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 80 ; y++) {
			fprintf(o_stream,"%.6f ",O13_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O13_SW[y][k][x]);
		}
		if(x != 54 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 54 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 52 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 52 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O14_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O14_SW[y][k][x]);
		}
		if(x != 52 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 52 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 52 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 52 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O15_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O15_SW[y][k][x]);
		}
		if(x != 52 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 52 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 52 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 52 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O16_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O16_SW[y][k][x]);
		}
		if(x != 52 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 52 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","MaxPooling2D : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O17_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O17_SW[y][k][x]);
		}
		if(x != 25 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 25 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O18_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O18_SW[y][k][x]);
		}
		if(x != 25 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 25 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O19_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O19_SW[y][k][x]);
		}
		if(x != 25 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 25 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O20_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O20_SW[y][k][x]);
		}
		if(x != 25 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 25 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 48 ; y++) {
			fprintf(o_stream,"%.6f ",O21_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O21_SW[y][k][x]);
		}
		if(x != 25 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 25 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 96 ; y++) {
			fprintf(o_stream,"%.6f ",O22_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O22_SW[y][k][x]);
		}
		if(x != 25 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 25 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 48 ; y++) {
			fprintf(o_stream,"%.6f ",O23_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O23_SW[y][k][x]);
		}
		if(x != 25 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 25 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 96 ; y++) {
			fprintf(o_stream,"%.6f ",O24_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O24_SW[y][k][x]);
		}
		if(x != 25 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 25 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 48 ; y++) {
			fprintf(o_stream,"%.6f ",O25_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O25_SW[y][k][x]);
		}
		if(x != 25 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 25 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 96 ; y++) {
			fprintf(o_stream,"%.6f ",O26_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O26_SW[y][k][x]);
		}
		if(x != 25 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 25 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","AveragePooling2D : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O27_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O27_SW[y][k][x]);
		}
		if(x != 25 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 25 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O28_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O28_SW[y][k][x]);
		}
		if(x != 25 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 25 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O29_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O29_SW[y][k][x]);
		}
		if(x != 25 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 25 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 96 ; y++) {
			fprintf(o_stream,"%.6f ",O30_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O30_SW[y][k][x]);
		}
		if(x != 25 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 25 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O31_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O31_SW[y][k][x]);
		}
		if(x != 25 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 25 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O32_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O32_SW[y][k][x]);
		}
		if(x != 25 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 25 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O33_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O33_SW[y][k][x]);
		}
		if(x != 25 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 25 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 96 ; y++) {
			fprintf(o_stream,"%.6f ",O34_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O34_SW[y][k][x]);
		}
		if(x != 25 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 25 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O35_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O35_SW[y][k][x]);
		}
		if(x != 25 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 25 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O36_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O36_SW[y][k][x]);
		}
		if(x != 25 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 25 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O37_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O37_SW[y][k][x]);
		}
		if(x != 25 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 25 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 96 ; y++) {
			fprintf(o_stream,"%.6f ",O38_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O38_SW[y][k][x]);
		}
		if(x != 25 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 25 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O39_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O39_SW[y][k][x]);
		}
		if(x != 25 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 25 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O40_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O40_SW[y][k][x]);
		}
		if(x != 25 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 25 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");





    fclose(w_stream);
    fclose(i_stream);
    fclose(o_stream);
    fclose(c_num);

    return 0;
}
