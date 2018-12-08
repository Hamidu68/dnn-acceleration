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
void SW_activation_1(DATA_T I[32][111][111], DATA_T O[32][111][111]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<111; x++) {
			for (y = 0; y<111; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
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
void SW_activation_2(DATA_T I[32][109][109], DATA_T O[32][109][109]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<109; x++) {
			for (y = 0; y<109; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
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
							if (x*1 + i < 109 + p && y*1 + j < 109 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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
void SW_activation_3(DATA_T I[64][109][109], DATA_T O[64][109][109]) {
	int m, x, y, i, j, k;
	for (m = 0; m<64; m++) {
		for (x = 0; x<109; x++) {
			for (y = 0; y<109; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_max_pooling2d_1(DATA_T I[64][109][109], DATA_T O[64][54][54])
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
void SW_activation_4(DATA_T I[80][54][54], DATA_T O[80][54][54]) {
	int m, x, y, i, j, k;
	for (m = 0; m<80; m++) {
		for (x = 0; x<54; x++) {
			for (y = 0; y<54; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
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
void SW_activation_5(DATA_T I[192][52][52], DATA_T O[192][52][52]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<52; x++) {
			for (y = 0; y<52; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_max_pooling2d_2(DATA_T I[192][52][52], DATA_T O[192][25][25])
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
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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
void SW_activation_9(DATA_T I[64][25][25], DATA_T O[64][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
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
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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
void SW_activation_7(DATA_T I[48][25][25], DATA_T O[48][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<48; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_10(DATA_T I[96][25][25], DATA_T O[96][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<96; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_average_pooling2d_1(DATA_T I[192][25][25], DATA_T O[192][25][25])
{
	int m, x, y, i, j;
	DATA_T sum;
	int p = (1*(25-1) - 25 + 3)/2;
    DATA_T div;
	for (m = 0; m<192; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
                sum = (DATA_T)0;
                if(x!=0 && x!=25-1 && y!=0 && y!=25-1)
                    div = 9;
                else if((x==0 || x==25-1) && (y==0 || y==25-1))
                    div = 4;
                else
                    div = 6;
				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {

						if (x + i < 25 + p && y + j < 25 + p && x + i -p >= 0 && y + j -p >= 0)
						    sum += I[m][x*1 + i -p][y*1 + j -p];
					}
				}
				O[m][x][y] = sum/div;
			}
		}
	}
}
void SW_conv2d_6(DATA_T I[192][25][25], DATA_T O[96][25][25], DATA_T W[96][192][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<96; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<192; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_12(DATA_T I[192][25][25], DATA_T O[64][25][25], DATA_T W[64][192][1][1]) {
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
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_6(DATA_T I[96][25][25], DATA_T O[96][25][25], DATA_T W[3][96]) {
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
void SW_batch_normalization_12(DATA_T I[64][25][25], DATA_T O[64][25][25], DATA_T W[3][64]) {
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
void SW_activation_6(DATA_T I[96][25][25], DATA_T O[96][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<96; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_8(DATA_T I[64][25][25], DATA_T O[64][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_11(DATA_T I[96][25][25], DATA_T O[96][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<96; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_12(DATA_T I[64][25][25], DATA_T O[64][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_mixed_5b(DATA_T I1[96][25][25], DATA_T I2[64][25][25], DATA_T I3[96][25][25], DATA_T I4[64][25][25], DATA_T O[320][25][25]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 25; x++) {
		for(y = 0; y < 25; y++) {
			ch=0;
			for(k = ch; k < ch+96; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=96;
			for(k = ch; k < ch+64; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
			ch+=64;
			for(k = ch; k < ch+96; k++){
				O[k][x][y] = I3[k-ch][x][y];
			}
			ch+=96;
			for(k = ch; k < ch+64; k++){
				O[k][x][y] = I4[k-ch][x][y];
			}
		}
	}

}
void SW_conv2d_16(DATA_T I[320][25][25], DATA_T O[32][25][25], DATA_T W[32][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_16(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_activation_16(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_14(DATA_T I[320][25][25], DATA_T O[32][25][25], DATA_T W[32][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_17(DATA_T I[32][25][25], DATA_T O[48][25][25], DATA_T W[48][32][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 3)/2;
	for (m = 0; m<48; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<32; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_14(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_batch_normalization_17(DATA_T I[48][25][25], DATA_T O[48][25][25], DATA_T W[3][48]) {
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
void SW_activation_14(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_17(DATA_T I[48][25][25], DATA_T O[48][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<48; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_13(DATA_T I[320][25][25], DATA_T O[32][25][25], DATA_T W[32][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_15(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[32][32][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<32; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_18(DATA_T I[48][25][25], DATA_T O[64][25][25], DATA_T W[64][48][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 3)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<48; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_13(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_batch_normalization_15(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_batch_normalization_18(DATA_T I[64][25][25], DATA_T O[64][25][25], DATA_T W[3][64]) {
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
void SW_activation_13(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_15(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_18(DATA_T I[64][25][25], DATA_T O[64][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block35_1_mixed(DATA_T I1[32][25][25], DATA_T I2[32][25][25], DATA_T I3[64][25][25], DATA_T O[128][25][25]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 25; x++) {
		for(y = 0; y < 25; y++) {
			ch=0;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=32;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
			ch+=32;
			for(k = ch; k < ch+64; k++){
				O[k][x][y] = I3[k-ch][x][y];
			}
		}
	}

}
void SW_block35_1_conv(DATA_T I[128][25][25], DATA_T O[320][25][25], DATA_T W[320][128][1][1], DATA_T B[320]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<320; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
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
				ofm += B[m];
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block35_1(DATA_T I1[320][25][25], DATA_T I2[320][25][25], DATA_T O[320][25][25]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 25; x++) {
		for(y = 0; y < 25; y++) {
			ch=0;
			for(k = ch; k < ch+320; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=320;
			for(k = ch; k < ch+320; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block35_1_ac(DATA_T I[320][25][25], DATA_T O[320][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<320; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_22(DATA_T I[320][25][25], DATA_T O[32][25][25], DATA_T W[32][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_22(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_activation_22(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_20(DATA_T I[320][25][25], DATA_T O[32][25][25], DATA_T W[32][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_23(DATA_T I[32][25][25], DATA_T O[48][25][25], DATA_T W[48][32][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 3)/2;
	for (m = 0; m<48; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<32; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_20(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_batch_normalization_23(DATA_T I[48][25][25], DATA_T O[48][25][25], DATA_T W[3][48]) {
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
void SW_activation_20(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_23(DATA_T I[48][25][25], DATA_T O[48][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<48; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_19(DATA_T I[320][25][25], DATA_T O[32][25][25], DATA_T W[32][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_21(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[32][32][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<32; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_24(DATA_T I[48][25][25], DATA_T O[64][25][25], DATA_T W[64][48][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 3)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<48; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_19(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_batch_normalization_21(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_batch_normalization_24(DATA_T I[64][25][25], DATA_T O[64][25][25], DATA_T W[3][64]) {
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
void SW_activation_19(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_21(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_24(DATA_T I[64][25][25], DATA_T O[64][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block35_2_mixed(DATA_T I1[32][25][25], DATA_T I2[32][25][25], DATA_T I3[64][25][25], DATA_T O[128][25][25]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 25; x++) {
		for(y = 0; y < 25; y++) {
			ch=0;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=32;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
			ch+=32;
			for(k = ch; k < ch+64; k++){
				O[k][x][y] = I3[k-ch][x][y];
			}
		}
	}

}
void SW_block35_2_conv(DATA_T I[128][25][25], DATA_T O[320][25][25], DATA_T W[320][128][1][1], DATA_T B[320]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<320; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
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
				ofm += B[m];
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block35_2(DATA_T I1[320][25][25], DATA_T I2[320][25][25], DATA_T O[320][25][25]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 25; x++) {
		for(y = 0; y < 25; y++) {
			ch=0;
			for(k = ch; k < ch+320; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=320;
			for(k = ch; k < ch+320; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block35_2_ac(DATA_T I[320][25][25], DATA_T O[320][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<320; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_28(DATA_T I[320][25][25], DATA_T O[32][25][25], DATA_T W[32][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_28(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_activation_28(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_26(DATA_T I[320][25][25], DATA_T O[32][25][25], DATA_T W[32][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_29(DATA_T I[32][25][25], DATA_T O[48][25][25], DATA_T W[48][32][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 3)/2;
	for (m = 0; m<48; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<32; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_26(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_batch_normalization_29(DATA_T I[48][25][25], DATA_T O[48][25][25], DATA_T W[3][48]) {
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
void SW_activation_26(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_29(DATA_T I[48][25][25], DATA_T O[48][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<48; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_25(DATA_T I[320][25][25], DATA_T O[32][25][25], DATA_T W[32][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_27(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[32][32][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<32; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_30(DATA_T I[48][25][25], DATA_T O[64][25][25], DATA_T W[64][48][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 3)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<48; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_25(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_batch_normalization_27(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_batch_normalization_30(DATA_T I[64][25][25], DATA_T O[64][25][25], DATA_T W[3][64]) {
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
void SW_activation_25(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_27(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_30(DATA_T I[64][25][25], DATA_T O[64][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block35_3_mixed(DATA_T I1[32][25][25], DATA_T I2[32][25][25], DATA_T I3[64][25][25], DATA_T O[128][25][25]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 25; x++) {
		for(y = 0; y < 25; y++) {
			ch=0;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=32;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
			ch+=32;
			for(k = ch; k < ch+64; k++){
				O[k][x][y] = I3[k-ch][x][y];
			}
		}
	}

}
void SW_block35_3_conv(DATA_T I[128][25][25], DATA_T O[320][25][25], DATA_T W[320][128][1][1], DATA_T B[320]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<320; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
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
				ofm += B[m];
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block35_3(DATA_T I1[320][25][25], DATA_T I2[320][25][25], DATA_T O[320][25][25]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 25; x++) {
		for(y = 0; y < 25; y++) {
			ch=0;
			for(k = ch; k < ch+320; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=320;
			for(k = ch; k < ch+320; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block35_3_ac(DATA_T I[320][25][25], DATA_T O[320][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<320; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_34(DATA_T I[320][25][25], DATA_T O[32][25][25], DATA_T W[32][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_34(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_activation_34(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_32(DATA_T I[320][25][25], DATA_T O[32][25][25], DATA_T W[32][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_35(DATA_T I[32][25][25], DATA_T O[48][25][25], DATA_T W[48][32][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 3)/2;
	for (m = 0; m<48; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<32; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_32(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_batch_normalization_35(DATA_T I[48][25][25], DATA_T O[48][25][25], DATA_T W[3][48]) {
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
void SW_activation_32(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_35(DATA_T I[48][25][25], DATA_T O[48][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<48; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_31(DATA_T I[320][25][25], DATA_T O[32][25][25], DATA_T W[32][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_33(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[32][32][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<32; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_36(DATA_T I[48][25][25], DATA_T O[64][25][25], DATA_T W[64][48][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 3)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<48; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_31(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_batch_normalization_33(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_batch_normalization_36(DATA_T I[64][25][25], DATA_T O[64][25][25], DATA_T W[3][64]) {
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
void SW_activation_31(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_33(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_36(DATA_T I[64][25][25], DATA_T O[64][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block35_4_mixed(DATA_T I1[32][25][25], DATA_T I2[32][25][25], DATA_T I3[64][25][25], DATA_T O[128][25][25]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 25; x++) {
		for(y = 0; y < 25; y++) {
			ch=0;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=32;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
			ch+=32;
			for(k = ch; k < ch+64; k++){
				O[k][x][y] = I3[k-ch][x][y];
			}
		}
	}

}
void SW_block35_4_conv(DATA_T I[128][25][25], DATA_T O[320][25][25], DATA_T W[320][128][1][1], DATA_T B[320]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<320; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
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
				ofm += B[m];
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block35_4(DATA_T I1[320][25][25], DATA_T I2[320][25][25], DATA_T O[320][25][25]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 25; x++) {
		for(y = 0; y < 25; y++) {
			ch=0;
			for(k = ch; k < ch+320; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=320;
			for(k = ch; k < ch+320; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block35_4_ac(DATA_T I[320][25][25], DATA_T O[320][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<320; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_40(DATA_T I[320][25][25], DATA_T O[32][25][25], DATA_T W[32][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_40(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_activation_40(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_38(DATA_T I[320][25][25], DATA_T O[32][25][25], DATA_T W[32][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_41(DATA_T I[32][25][25], DATA_T O[48][25][25], DATA_T W[48][32][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 3)/2;
	for (m = 0; m<48; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<32; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_38(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_batch_normalization_41(DATA_T I[48][25][25], DATA_T O[48][25][25], DATA_T W[3][48]) {
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
void SW_activation_38(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_41(DATA_T I[48][25][25], DATA_T O[48][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<48; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_37(DATA_T I[320][25][25], DATA_T O[32][25][25], DATA_T W[32][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_39(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[32][32][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<32; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_42(DATA_T I[48][25][25], DATA_T O[64][25][25], DATA_T W[64][48][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 3)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<48; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_37(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_batch_normalization_39(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_batch_normalization_42(DATA_T I[64][25][25], DATA_T O[64][25][25], DATA_T W[3][64]) {
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
void SW_activation_37(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_39(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_42(DATA_T I[64][25][25], DATA_T O[64][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block35_5_mixed(DATA_T I1[32][25][25], DATA_T I2[32][25][25], DATA_T I3[64][25][25], DATA_T O[128][25][25]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 25; x++) {
		for(y = 0; y < 25; y++) {
			ch=0;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=32;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
			ch+=32;
			for(k = ch; k < ch+64; k++){
				O[k][x][y] = I3[k-ch][x][y];
			}
		}
	}

}
void SW_block35_5_conv(DATA_T I[128][25][25], DATA_T O[320][25][25], DATA_T W[320][128][1][1], DATA_T B[320]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<320; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
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
				ofm += B[m];
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block35_5(DATA_T I1[320][25][25], DATA_T I2[320][25][25], DATA_T O[320][25][25]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 25; x++) {
		for(y = 0; y < 25; y++) {
			ch=0;
			for(k = ch; k < ch+320; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=320;
			for(k = ch; k < ch+320; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block35_5_ac(DATA_T I[320][25][25], DATA_T O[320][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<320; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_46(DATA_T I[320][25][25], DATA_T O[32][25][25], DATA_T W[32][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_46(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_activation_46(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_44(DATA_T I[320][25][25], DATA_T O[32][25][25], DATA_T W[32][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_47(DATA_T I[32][25][25], DATA_T O[48][25][25], DATA_T W[48][32][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 3)/2;
	for (m = 0; m<48; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<32; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_44(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_batch_normalization_47(DATA_T I[48][25][25], DATA_T O[48][25][25], DATA_T W[3][48]) {
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
void SW_activation_44(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_47(DATA_T I[48][25][25], DATA_T O[48][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<48; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_43(DATA_T I[320][25][25], DATA_T O[32][25][25], DATA_T W[32][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_45(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[32][32][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<32; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_48(DATA_T I[48][25][25], DATA_T O[64][25][25], DATA_T W[64][48][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 3)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<48; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_43(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_batch_normalization_45(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_batch_normalization_48(DATA_T I[64][25][25], DATA_T O[64][25][25], DATA_T W[3][64]) {
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
void SW_activation_43(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_45(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_48(DATA_T I[64][25][25], DATA_T O[64][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block35_6_mixed(DATA_T I1[32][25][25], DATA_T I2[32][25][25], DATA_T I3[64][25][25], DATA_T O[128][25][25]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 25; x++) {
		for(y = 0; y < 25; y++) {
			ch=0;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=32;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
			ch+=32;
			for(k = ch; k < ch+64; k++){
				O[k][x][y] = I3[k-ch][x][y];
			}
		}
	}

}
void SW_block35_6_conv(DATA_T I[128][25][25], DATA_T O[320][25][25], DATA_T W[320][128][1][1], DATA_T B[320]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<320; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
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
				ofm += B[m];
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block35_6(DATA_T I1[320][25][25], DATA_T I2[320][25][25], DATA_T O[320][25][25]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 25; x++) {
		for(y = 0; y < 25; y++) {
			ch=0;
			for(k = ch; k < ch+320; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=320;
			for(k = ch; k < ch+320; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block35_6_ac(DATA_T I[320][25][25], DATA_T O[320][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<320; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_52(DATA_T I[320][25][25], DATA_T O[32][25][25], DATA_T W[32][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_52(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_activation_52(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_50(DATA_T I[320][25][25], DATA_T O[32][25][25], DATA_T W[32][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_53(DATA_T I[32][25][25], DATA_T O[48][25][25], DATA_T W[48][32][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 3)/2;
	for (m = 0; m<48; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<32; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_50(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_batch_normalization_53(DATA_T I[48][25][25], DATA_T O[48][25][25], DATA_T W[3][48]) {
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
void SW_activation_50(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_53(DATA_T I[48][25][25], DATA_T O[48][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<48; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_49(DATA_T I[320][25][25], DATA_T O[32][25][25], DATA_T W[32][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_51(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[32][32][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<32; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_54(DATA_T I[48][25][25], DATA_T O[64][25][25], DATA_T W[64][48][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 3)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<48; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_49(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_batch_normalization_51(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_batch_normalization_54(DATA_T I[64][25][25], DATA_T O[64][25][25], DATA_T W[3][64]) {
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
void SW_activation_49(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_51(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_54(DATA_T I[64][25][25], DATA_T O[64][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block35_7_mixed(DATA_T I1[32][25][25], DATA_T I2[32][25][25], DATA_T I3[64][25][25], DATA_T O[128][25][25]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 25; x++) {
		for(y = 0; y < 25; y++) {
			ch=0;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=32;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
			ch+=32;
			for(k = ch; k < ch+64; k++){
				O[k][x][y] = I3[k-ch][x][y];
			}
		}
	}

}
void SW_block35_7_conv(DATA_T I[128][25][25], DATA_T O[320][25][25], DATA_T W[320][128][1][1], DATA_T B[320]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<320; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
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
				ofm += B[m];
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block35_7(DATA_T I1[320][25][25], DATA_T I2[320][25][25], DATA_T O[320][25][25]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 25; x++) {
		for(y = 0; y < 25; y++) {
			ch=0;
			for(k = ch; k < ch+320; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=320;
			for(k = ch; k < ch+320; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block35_7_ac(DATA_T I[320][25][25], DATA_T O[320][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<320; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_58(DATA_T I[320][25][25], DATA_T O[32][25][25], DATA_T W[32][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_58(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_activation_58(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_56(DATA_T I[320][25][25], DATA_T O[32][25][25], DATA_T W[32][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_59(DATA_T I[32][25][25], DATA_T O[48][25][25], DATA_T W[48][32][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 3)/2;
	for (m = 0; m<48; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<32; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_56(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_batch_normalization_59(DATA_T I[48][25][25], DATA_T O[48][25][25], DATA_T W[3][48]) {
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
void SW_activation_56(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_59(DATA_T I[48][25][25], DATA_T O[48][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<48; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_55(DATA_T I[320][25][25], DATA_T O[32][25][25], DATA_T W[32][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_57(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[32][32][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<32; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_60(DATA_T I[48][25][25], DATA_T O[64][25][25], DATA_T W[64][48][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 3)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<48; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_55(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_batch_normalization_57(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_batch_normalization_60(DATA_T I[64][25][25], DATA_T O[64][25][25], DATA_T W[3][64]) {
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
void SW_activation_55(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_57(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_60(DATA_T I[64][25][25], DATA_T O[64][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block35_8_mixed(DATA_T I1[32][25][25], DATA_T I2[32][25][25], DATA_T I3[64][25][25], DATA_T O[128][25][25]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 25; x++) {
		for(y = 0; y < 25; y++) {
			ch=0;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=32;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
			ch+=32;
			for(k = ch; k < ch+64; k++){
				O[k][x][y] = I3[k-ch][x][y];
			}
		}
	}

}
void SW_block35_8_conv(DATA_T I[128][25][25], DATA_T O[320][25][25], DATA_T W[320][128][1][1], DATA_T B[320]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<320; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
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
				ofm += B[m];
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block35_8(DATA_T I1[320][25][25], DATA_T I2[320][25][25], DATA_T O[320][25][25]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 25; x++) {
		for(y = 0; y < 25; y++) {
			ch=0;
			for(k = ch; k < ch+320; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=320;
			for(k = ch; k < ch+320; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block35_8_ac(DATA_T I[320][25][25], DATA_T O[320][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<320; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_64(DATA_T I[320][25][25], DATA_T O[32][25][25], DATA_T W[32][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_64(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_activation_64(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_62(DATA_T I[320][25][25], DATA_T O[32][25][25], DATA_T W[32][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_65(DATA_T I[32][25][25], DATA_T O[48][25][25], DATA_T W[48][32][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 3)/2;
	for (m = 0; m<48; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<32; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_62(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_batch_normalization_65(DATA_T I[48][25][25], DATA_T O[48][25][25], DATA_T W[3][48]) {
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
void SW_activation_62(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_65(DATA_T I[48][25][25], DATA_T O[48][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<48; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_61(DATA_T I[320][25][25], DATA_T O[32][25][25], DATA_T W[32][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_63(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[32][32][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<32; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_66(DATA_T I[48][25][25], DATA_T O[64][25][25], DATA_T W[64][48][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 3)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<48; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_61(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_batch_normalization_63(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_batch_normalization_66(DATA_T I[64][25][25], DATA_T O[64][25][25], DATA_T W[3][64]) {
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
void SW_activation_61(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_63(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_66(DATA_T I[64][25][25], DATA_T O[64][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block35_9_mixed(DATA_T I1[32][25][25], DATA_T I2[32][25][25], DATA_T I3[64][25][25], DATA_T O[128][25][25]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 25; x++) {
		for(y = 0; y < 25; y++) {
			ch=0;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=32;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
			ch+=32;
			for(k = ch; k < ch+64; k++){
				O[k][x][y] = I3[k-ch][x][y];
			}
		}
	}

}
void SW_block35_9_conv(DATA_T I[128][25][25], DATA_T O[320][25][25], DATA_T W[320][128][1][1], DATA_T B[320]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<320; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
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
				ofm += B[m];
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block35_9(DATA_T I1[320][25][25], DATA_T I2[320][25][25], DATA_T O[320][25][25]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 25; x++) {
		for(y = 0; y < 25; y++) {
			ch=0;
			for(k = ch; k < ch+320; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=320;
			for(k = ch; k < ch+320; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block35_9_ac(DATA_T I[320][25][25], DATA_T O[320][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<320; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_70(DATA_T I[320][25][25], DATA_T O[32][25][25], DATA_T W[32][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_70(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_activation_70(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_68(DATA_T I[320][25][25], DATA_T O[32][25][25], DATA_T W[32][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_71(DATA_T I[32][25][25], DATA_T O[48][25][25], DATA_T W[48][32][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 3)/2;
	for (m = 0; m<48; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<32; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_68(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_batch_normalization_71(DATA_T I[48][25][25], DATA_T O[48][25][25], DATA_T W[3][48]) {
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
void SW_activation_68(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_71(DATA_T I[48][25][25], DATA_T O[48][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<48; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_67(DATA_T I[320][25][25], DATA_T O[32][25][25], DATA_T W[32][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_69(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[32][32][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<32; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_72(DATA_T I[48][25][25], DATA_T O[64][25][25], DATA_T W[64][48][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 3)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<48; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_67(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_batch_normalization_69(DATA_T I[32][25][25], DATA_T O[32][25][25], DATA_T W[3][32]) {
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
void SW_batch_normalization_72(DATA_T I[64][25][25], DATA_T O[64][25][25], DATA_T W[3][64]) {
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
void SW_activation_67(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_69(DATA_T I[32][25][25], DATA_T O[32][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<32; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_72(DATA_T I[64][25][25], DATA_T O[64][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block35_10_mixed(DATA_T I1[32][25][25], DATA_T I2[32][25][25], DATA_T I3[64][25][25], DATA_T O[128][25][25]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 25; x++) {
		for(y = 0; y < 25; y++) {
			ch=0;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=32;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
			ch+=32;
			for(k = ch; k < ch+64; k++){
				O[k][x][y] = I3[k-ch][x][y];
			}
		}
	}

}
void SW_block35_10_conv(DATA_T I[128][25][25], DATA_T O[320][25][25], DATA_T W[320][128][1][1], DATA_T B[320]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<320; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
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
				ofm += B[m];
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block35_10(DATA_T I1[320][25][25], DATA_T I2[320][25][25], DATA_T O[320][25][25]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 25; x++) {
		for(y = 0; y < 25; y++) {
			ch=0;
			for(k = ch; k < ch+320; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=320;
			for(k = ch; k < ch+320; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block35_10_ac(DATA_T I[320][25][25], DATA_T O[320][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<320; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_74(DATA_T I[320][25][25], DATA_T O[256][25][25], DATA_T W[256][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_74(DATA_T I[256][25][25], DATA_T O[256][25][25], DATA_T W[3][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 256; m++){
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
void SW_activation_74(DATA_T I[256][25][25], DATA_T O[256][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<256; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_75(DATA_T I[256][25][25], DATA_T O[256][25][25], DATA_T W[256][256][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 3)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<256; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 25 + p && y*1 + j < 25 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_75(DATA_T I[256][25][25], DATA_T O[256][25][25], DATA_T W[3][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 256; m++){
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
void SW_activation_75(DATA_T I[256][25][25], DATA_T O[256][25][25]) {
	int m, x, y, i, j, k;
	for (m = 0; m<256; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_73(DATA_T I[320][25][25], DATA_T O[384][12][12], DATA_T W[384][320][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<384; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i <= 25 && y + j <= 25) {
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

void SW_conv2d_76(DATA_T I[256][25][25], DATA_T O[384][12][12], DATA_T W[384][256][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<384; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<256; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i <= 25 && y + j <= 25) {
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

void SW_batch_normalization_73(DATA_T I[384][12][12], DATA_T O[384][12][12], DATA_T W[3][384]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 384; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_76(DATA_T I[384][12][12], DATA_T O[384][12][12], DATA_T W[3][384]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 384; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_73(DATA_T I[384][12][12], DATA_T O[384][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<384; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_76(DATA_T I[384][12][12], DATA_T O[384][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<384; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_max_pooling2d_3(DATA_T I[320][25][25], DATA_T O[320][12][12])
{
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<320; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
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
void SW_mixed_6a(DATA_T I1[384][12][12], DATA_T I2[384][12][12], DATA_T I3[320][12][12], DATA_T O[1088][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+384; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=384;
			for(k = ch; k < ch+384; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
			ch+=384;
			for(k = ch; k < ch+320; k++){
				O[k][x][y] = I3[k-ch][x][y];
			}
		}
	}

}
void SW_conv2d_78(DATA_T I[1088][12][12], DATA_T O[128][12][12], DATA_T W[128][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_78(DATA_T I[128][12][12], DATA_T O[128][12][12], DATA_T W[3][128]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 128; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_78(DATA_T I[128][12][12], DATA_T O[128][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_79(DATA_T I[128][12][12], DATA_T O[160][12][12], DATA_T W[160][128][1][7]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<7; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_79(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[3][160]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 160; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_79(DATA_T I[160][12][12], DATA_T O[160][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_77(DATA_T I[1088][12][12], DATA_T O[192][12][12], DATA_T W[192][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_80(DATA_T I[160][12][12], DATA_T O[192][12][12], DATA_T W[192][160][7][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 7)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<160; k++) {
					for (i = 0; i<7; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_77(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_80(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_77(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_80(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block17_1_mixed(DATA_T I1[192][12][12], DATA_T I2[192][12][12], DATA_T O[384][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_block17_1_conv(DATA_T I[384][12][12], DATA_T O[1088][12][12], DATA_T W[1088][384][1][1], DATA_T B[1088]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 12 + p && y + j < 12 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block17_1(DATA_T I1[1088][12][12], DATA_T I2[1088][12][12], DATA_T O[1088][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+1088; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=1088;
			for(k = ch; k < ch+1088; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block17_1_ac(DATA_T I[1088][12][12], DATA_T O[1088][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_82(DATA_T I[1088][12][12], DATA_T O[128][12][12], DATA_T W[128][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_82(DATA_T I[128][12][12], DATA_T O[128][12][12], DATA_T W[3][128]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 128; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_82(DATA_T I[128][12][12], DATA_T O[128][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_83(DATA_T I[128][12][12], DATA_T O[160][12][12], DATA_T W[160][128][1][7]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<7; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_83(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[3][160]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 160; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_83(DATA_T I[160][12][12], DATA_T O[160][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_81(DATA_T I[1088][12][12], DATA_T O[192][12][12], DATA_T W[192][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_84(DATA_T I[160][12][12], DATA_T O[192][12][12], DATA_T W[192][160][7][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 7)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<160; k++) {
					for (i = 0; i<7; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_81(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_84(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_81(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_84(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block17_2_mixed(DATA_T I1[192][12][12], DATA_T I2[192][12][12], DATA_T O[384][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_block17_2_conv(DATA_T I[384][12][12], DATA_T O[1088][12][12], DATA_T W[1088][384][1][1], DATA_T B[1088]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 12 + p && y + j < 12 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block17_2(DATA_T I1[1088][12][12], DATA_T I2[1088][12][12], DATA_T O[1088][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+1088; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=1088;
			for(k = ch; k < ch+1088; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block17_2_ac(DATA_T I[1088][12][12], DATA_T O[1088][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_86(DATA_T I[1088][12][12], DATA_T O[128][12][12], DATA_T W[128][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_86(DATA_T I[128][12][12], DATA_T O[128][12][12], DATA_T W[3][128]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 128; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_86(DATA_T I[128][12][12], DATA_T O[128][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_87(DATA_T I[128][12][12], DATA_T O[160][12][12], DATA_T W[160][128][1][7]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<7; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_87(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[3][160]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 160; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_87(DATA_T I[160][12][12], DATA_T O[160][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_85(DATA_T I[1088][12][12], DATA_T O[192][12][12], DATA_T W[192][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_88(DATA_T I[160][12][12], DATA_T O[192][12][12], DATA_T W[192][160][7][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 7)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<160; k++) {
					for (i = 0; i<7; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_85(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_88(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_85(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_88(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block17_3_mixed(DATA_T I1[192][12][12], DATA_T I2[192][12][12], DATA_T O[384][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_block17_3_conv(DATA_T I[384][12][12], DATA_T O[1088][12][12], DATA_T W[1088][384][1][1], DATA_T B[1088]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 12 + p && y + j < 12 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block17_3(DATA_T I1[1088][12][12], DATA_T I2[1088][12][12], DATA_T O[1088][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+1088; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=1088;
			for(k = ch; k < ch+1088; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block17_3_ac(DATA_T I[1088][12][12], DATA_T O[1088][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_90(DATA_T I[1088][12][12], DATA_T O[128][12][12], DATA_T W[128][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_90(DATA_T I[128][12][12], DATA_T O[128][12][12], DATA_T W[3][128]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 128; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_90(DATA_T I[128][12][12], DATA_T O[128][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_91(DATA_T I[128][12][12], DATA_T O[160][12][12], DATA_T W[160][128][1][7]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<7; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_91(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[3][160]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 160; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_91(DATA_T I[160][12][12], DATA_T O[160][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_89(DATA_T I[1088][12][12], DATA_T O[192][12][12], DATA_T W[192][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_92(DATA_T I[160][12][12], DATA_T O[192][12][12], DATA_T W[192][160][7][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 7)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<160; k++) {
					for (i = 0; i<7; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_89(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_92(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_89(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_92(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block17_4_mixed(DATA_T I1[192][12][12], DATA_T I2[192][12][12], DATA_T O[384][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_block17_4_conv(DATA_T I[384][12][12], DATA_T O[1088][12][12], DATA_T W[1088][384][1][1], DATA_T B[1088]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 12 + p && y + j < 12 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block17_4(DATA_T I1[1088][12][12], DATA_T I2[1088][12][12], DATA_T O[1088][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+1088; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=1088;
			for(k = ch; k < ch+1088; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block17_4_ac(DATA_T I[1088][12][12], DATA_T O[1088][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_94(DATA_T I[1088][12][12], DATA_T O[128][12][12], DATA_T W[128][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_94(DATA_T I[128][12][12], DATA_T O[128][12][12], DATA_T W[3][128]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 128; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_94(DATA_T I[128][12][12], DATA_T O[128][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_95(DATA_T I[128][12][12], DATA_T O[160][12][12], DATA_T W[160][128][1][7]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<7; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_95(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[3][160]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 160; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_95(DATA_T I[160][12][12], DATA_T O[160][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_93(DATA_T I[1088][12][12], DATA_T O[192][12][12], DATA_T W[192][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_96(DATA_T I[160][12][12], DATA_T O[192][12][12], DATA_T W[192][160][7][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 7)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<160; k++) {
					for (i = 0; i<7; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_93(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_96(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_93(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_96(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block17_5_mixed(DATA_T I1[192][12][12], DATA_T I2[192][12][12], DATA_T O[384][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_block17_5_conv(DATA_T I[384][12][12], DATA_T O[1088][12][12], DATA_T W[1088][384][1][1], DATA_T B[1088]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 12 + p && y + j < 12 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block17_5(DATA_T I1[1088][12][12], DATA_T I2[1088][12][12], DATA_T O[1088][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+1088; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=1088;
			for(k = ch; k < ch+1088; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block17_5_ac(DATA_T I[1088][12][12], DATA_T O[1088][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_98(DATA_T I[1088][12][12], DATA_T O[128][12][12], DATA_T W[128][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_98(DATA_T I[128][12][12], DATA_T O[128][12][12], DATA_T W[3][128]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 128; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_98(DATA_T I[128][12][12], DATA_T O[128][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_99(DATA_T I[128][12][12], DATA_T O[160][12][12], DATA_T W[160][128][1][7]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<7; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_99(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[3][160]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 160; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_99(DATA_T I[160][12][12], DATA_T O[160][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_97(DATA_T I[1088][12][12], DATA_T O[192][12][12], DATA_T W[192][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_100(DATA_T I[160][12][12], DATA_T O[192][12][12], DATA_T W[192][160][7][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 7)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<160; k++) {
					for (i = 0; i<7; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_97(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_100(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_97(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_100(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block17_6_mixed(DATA_T I1[192][12][12], DATA_T I2[192][12][12], DATA_T O[384][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_block17_6_conv(DATA_T I[384][12][12], DATA_T O[1088][12][12], DATA_T W[1088][384][1][1], DATA_T B[1088]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 12 + p && y + j < 12 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block17_6(DATA_T I1[1088][12][12], DATA_T I2[1088][12][12], DATA_T O[1088][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+1088; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=1088;
			for(k = ch; k < ch+1088; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block17_6_ac(DATA_T I[1088][12][12], DATA_T O[1088][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_102(DATA_T I[1088][12][12], DATA_T O[128][12][12], DATA_T W[128][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_102(DATA_T I[128][12][12], DATA_T O[128][12][12], DATA_T W[3][128]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 128; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_102(DATA_T I[128][12][12], DATA_T O[128][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_103(DATA_T I[128][12][12], DATA_T O[160][12][12], DATA_T W[160][128][1][7]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<7; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_103(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[3][160]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 160; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_103(DATA_T I[160][12][12], DATA_T O[160][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_101(DATA_T I[1088][12][12], DATA_T O[192][12][12], DATA_T W[192][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_104(DATA_T I[160][12][12], DATA_T O[192][12][12], DATA_T W[192][160][7][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 7)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<160; k++) {
					for (i = 0; i<7; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_101(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_104(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_101(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_104(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block17_7_mixed(DATA_T I1[192][12][12], DATA_T I2[192][12][12], DATA_T O[384][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_block17_7_conv(DATA_T I[384][12][12], DATA_T O[1088][12][12], DATA_T W[1088][384][1][1], DATA_T B[1088]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 12 + p && y + j < 12 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block17_7(DATA_T I1[1088][12][12], DATA_T I2[1088][12][12], DATA_T O[1088][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+1088; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=1088;
			for(k = ch; k < ch+1088; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block17_7_ac(DATA_T I[1088][12][12], DATA_T O[1088][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_106(DATA_T I[1088][12][12], DATA_T O[128][12][12], DATA_T W[128][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_106(DATA_T I[128][12][12], DATA_T O[128][12][12], DATA_T W[3][128]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 128; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_106(DATA_T I[128][12][12], DATA_T O[128][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_107(DATA_T I[128][12][12], DATA_T O[160][12][12], DATA_T W[160][128][1][7]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<7; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_107(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[3][160]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 160; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_107(DATA_T I[160][12][12], DATA_T O[160][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_105(DATA_T I[1088][12][12], DATA_T O[192][12][12], DATA_T W[192][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_108(DATA_T I[160][12][12], DATA_T O[192][12][12], DATA_T W[192][160][7][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 7)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<160; k++) {
					for (i = 0; i<7; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_105(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_108(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_105(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_108(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block17_8_mixed(DATA_T I1[192][12][12], DATA_T I2[192][12][12], DATA_T O[384][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_block17_8_conv(DATA_T I[384][12][12], DATA_T O[1088][12][12], DATA_T W[1088][384][1][1], DATA_T B[1088]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 12 + p && y + j < 12 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block17_8(DATA_T I1[1088][12][12], DATA_T I2[1088][12][12], DATA_T O[1088][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+1088; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=1088;
			for(k = ch; k < ch+1088; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block17_8_ac(DATA_T I[1088][12][12], DATA_T O[1088][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_110(DATA_T I[1088][12][12], DATA_T O[128][12][12], DATA_T W[128][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_110(DATA_T I[128][12][12], DATA_T O[128][12][12], DATA_T W[3][128]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 128; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_110(DATA_T I[128][12][12], DATA_T O[128][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_111(DATA_T I[128][12][12], DATA_T O[160][12][12], DATA_T W[160][128][1][7]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<7; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_111(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[3][160]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 160; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_111(DATA_T I[160][12][12], DATA_T O[160][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_109(DATA_T I[1088][12][12], DATA_T O[192][12][12], DATA_T W[192][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_112(DATA_T I[160][12][12], DATA_T O[192][12][12], DATA_T W[192][160][7][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 7)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<160; k++) {
					for (i = 0; i<7; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_109(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_112(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_109(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_112(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block17_9_mixed(DATA_T I1[192][12][12], DATA_T I2[192][12][12], DATA_T O[384][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_block17_9_conv(DATA_T I[384][12][12], DATA_T O[1088][12][12], DATA_T W[1088][384][1][1], DATA_T B[1088]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 12 + p && y + j < 12 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block17_9(DATA_T I1[1088][12][12], DATA_T I2[1088][12][12], DATA_T O[1088][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+1088; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=1088;
			for(k = ch; k < ch+1088; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block17_9_ac(DATA_T I[1088][12][12], DATA_T O[1088][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_114(DATA_T I[1088][12][12], DATA_T O[128][12][12], DATA_T W[128][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_114(DATA_T I[128][12][12], DATA_T O[128][12][12], DATA_T W[3][128]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 128; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_114(DATA_T I[128][12][12], DATA_T O[128][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_115(DATA_T I[128][12][12], DATA_T O[160][12][12], DATA_T W[160][128][1][7]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<7; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_115(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[3][160]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 160; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_115(DATA_T I[160][12][12], DATA_T O[160][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_113(DATA_T I[1088][12][12], DATA_T O[192][12][12], DATA_T W[192][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_116(DATA_T I[160][12][12], DATA_T O[192][12][12], DATA_T W[192][160][7][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 7)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<160; k++) {
					for (i = 0; i<7; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_113(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_116(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_113(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_116(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block17_10_mixed(DATA_T I1[192][12][12], DATA_T I2[192][12][12], DATA_T O[384][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_block17_10_conv(DATA_T I[384][12][12], DATA_T O[1088][12][12], DATA_T W[1088][384][1][1], DATA_T B[1088]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 12 + p && y + j < 12 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block17_10(DATA_T I1[1088][12][12], DATA_T I2[1088][12][12], DATA_T O[1088][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+1088; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=1088;
			for(k = ch; k < ch+1088; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block17_10_ac(DATA_T I[1088][12][12], DATA_T O[1088][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_118(DATA_T I[1088][12][12], DATA_T O[128][12][12], DATA_T W[128][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_118(DATA_T I[128][12][12], DATA_T O[128][12][12], DATA_T W[3][128]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 128; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_118(DATA_T I[128][12][12], DATA_T O[128][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_119(DATA_T I[128][12][12], DATA_T O[160][12][12], DATA_T W[160][128][1][7]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<7; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_119(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[3][160]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 160; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_119(DATA_T I[160][12][12], DATA_T O[160][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_117(DATA_T I[1088][12][12], DATA_T O[192][12][12], DATA_T W[192][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_120(DATA_T I[160][12][12], DATA_T O[192][12][12], DATA_T W[192][160][7][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 7)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<160; k++) {
					for (i = 0; i<7; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_117(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_120(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_117(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_120(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block17_11_mixed(DATA_T I1[192][12][12], DATA_T I2[192][12][12], DATA_T O[384][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_block17_11_conv(DATA_T I[384][12][12], DATA_T O[1088][12][12], DATA_T W[1088][384][1][1], DATA_T B[1088]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 12 + p && y + j < 12 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block17_11(DATA_T I1[1088][12][12], DATA_T I2[1088][12][12], DATA_T O[1088][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+1088; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=1088;
			for(k = ch; k < ch+1088; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block17_11_ac(DATA_T I[1088][12][12], DATA_T O[1088][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_122(DATA_T I[1088][12][12], DATA_T O[128][12][12], DATA_T W[128][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_122(DATA_T I[128][12][12], DATA_T O[128][12][12], DATA_T W[3][128]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 128; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_122(DATA_T I[128][12][12], DATA_T O[128][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_123(DATA_T I[128][12][12], DATA_T O[160][12][12], DATA_T W[160][128][1][7]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<7; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_123(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[3][160]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 160; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_123(DATA_T I[160][12][12], DATA_T O[160][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_121(DATA_T I[1088][12][12], DATA_T O[192][12][12], DATA_T W[192][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_124(DATA_T I[160][12][12], DATA_T O[192][12][12], DATA_T W[192][160][7][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 7)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<160; k++) {
					for (i = 0; i<7; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_121(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_124(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_121(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_124(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block17_12_mixed(DATA_T I1[192][12][12], DATA_T I2[192][12][12], DATA_T O[384][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_block17_12_conv(DATA_T I[384][12][12], DATA_T O[1088][12][12], DATA_T W[1088][384][1][1], DATA_T B[1088]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 12 + p && y + j < 12 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block17_12(DATA_T I1[1088][12][12], DATA_T I2[1088][12][12], DATA_T O[1088][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+1088; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=1088;
			for(k = ch; k < ch+1088; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block17_12_ac(DATA_T I[1088][12][12], DATA_T O[1088][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_126(DATA_T I[1088][12][12], DATA_T O[128][12][12], DATA_T W[128][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_126(DATA_T I[128][12][12], DATA_T O[128][12][12], DATA_T W[3][128]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 128; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_126(DATA_T I[128][12][12], DATA_T O[128][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_127(DATA_T I[128][12][12], DATA_T O[160][12][12], DATA_T W[160][128][1][7]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<7; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_127(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[3][160]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 160; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_127(DATA_T I[160][12][12], DATA_T O[160][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_125(DATA_T I[1088][12][12], DATA_T O[192][12][12], DATA_T W[192][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_128(DATA_T I[160][12][12], DATA_T O[192][12][12], DATA_T W[192][160][7][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 7)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<160; k++) {
					for (i = 0; i<7; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_125(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_128(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_125(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_128(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block17_13_mixed(DATA_T I1[192][12][12], DATA_T I2[192][12][12], DATA_T O[384][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_block17_13_conv(DATA_T I[384][12][12], DATA_T O[1088][12][12], DATA_T W[1088][384][1][1], DATA_T B[1088]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 12 + p && y + j < 12 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block17_13(DATA_T I1[1088][12][12], DATA_T I2[1088][12][12], DATA_T O[1088][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+1088; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=1088;
			for(k = ch; k < ch+1088; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block17_13_ac(DATA_T I[1088][12][12], DATA_T O[1088][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_130(DATA_T I[1088][12][12], DATA_T O[128][12][12], DATA_T W[128][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_130(DATA_T I[128][12][12], DATA_T O[128][12][12], DATA_T W[3][128]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 128; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_130(DATA_T I[128][12][12], DATA_T O[128][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_131(DATA_T I[128][12][12], DATA_T O[160][12][12], DATA_T W[160][128][1][7]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<7; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_131(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[3][160]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 160; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_131(DATA_T I[160][12][12], DATA_T O[160][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_129(DATA_T I[1088][12][12], DATA_T O[192][12][12], DATA_T W[192][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_132(DATA_T I[160][12][12], DATA_T O[192][12][12], DATA_T W[192][160][7][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 7)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<160; k++) {
					for (i = 0; i<7; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_129(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_132(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_129(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_132(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block17_14_mixed(DATA_T I1[192][12][12], DATA_T I2[192][12][12], DATA_T O[384][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_block17_14_conv(DATA_T I[384][12][12], DATA_T O[1088][12][12], DATA_T W[1088][384][1][1], DATA_T B[1088]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 12 + p && y + j < 12 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block17_14(DATA_T I1[1088][12][12], DATA_T I2[1088][12][12], DATA_T O[1088][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+1088; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=1088;
			for(k = ch; k < ch+1088; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block17_14_ac(DATA_T I[1088][12][12], DATA_T O[1088][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_134(DATA_T I[1088][12][12], DATA_T O[128][12][12], DATA_T W[128][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_134(DATA_T I[128][12][12], DATA_T O[128][12][12], DATA_T W[3][128]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 128; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_134(DATA_T I[128][12][12], DATA_T O[128][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_135(DATA_T I[128][12][12], DATA_T O[160][12][12], DATA_T W[160][128][1][7]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<7; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_135(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[3][160]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 160; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_135(DATA_T I[160][12][12], DATA_T O[160][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_133(DATA_T I[1088][12][12], DATA_T O[192][12][12], DATA_T W[192][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_136(DATA_T I[160][12][12], DATA_T O[192][12][12], DATA_T W[192][160][7][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 7)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<160; k++) {
					for (i = 0; i<7; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_133(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_136(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_133(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_136(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block17_15_mixed(DATA_T I1[192][12][12], DATA_T I2[192][12][12], DATA_T O[384][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_block17_15_conv(DATA_T I[384][12][12], DATA_T O[1088][12][12], DATA_T W[1088][384][1][1], DATA_T B[1088]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 12 + p && y + j < 12 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block17_15(DATA_T I1[1088][12][12], DATA_T I2[1088][12][12], DATA_T O[1088][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+1088; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=1088;
			for(k = ch; k < ch+1088; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block17_15_ac(DATA_T I[1088][12][12], DATA_T O[1088][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_138(DATA_T I[1088][12][12], DATA_T O[128][12][12], DATA_T W[128][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_138(DATA_T I[128][12][12], DATA_T O[128][12][12], DATA_T W[3][128]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 128; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_138(DATA_T I[128][12][12], DATA_T O[128][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_139(DATA_T I[128][12][12], DATA_T O[160][12][12], DATA_T W[160][128][1][7]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<7; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_139(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[3][160]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 160; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_139(DATA_T I[160][12][12], DATA_T O[160][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_137(DATA_T I[1088][12][12], DATA_T O[192][12][12], DATA_T W[192][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_140(DATA_T I[160][12][12], DATA_T O[192][12][12], DATA_T W[192][160][7][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 7)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<160; k++) {
					for (i = 0; i<7; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_137(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_140(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_137(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_140(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block17_16_mixed(DATA_T I1[192][12][12], DATA_T I2[192][12][12], DATA_T O[384][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_block17_16_conv(DATA_T I[384][12][12], DATA_T O[1088][12][12], DATA_T W[1088][384][1][1], DATA_T B[1088]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 12 + p && y + j < 12 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block17_16(DATA_T I1[1088][12][12], DATA_T I2[1088][12][12], DATA_T O[1088][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+1088; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=1088;
			for(k = ch; k < ch+1088; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block17_16_ac(DATA_T I[1088][12][12], DATA_T O[1088][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_142(DATA_T I[1088][12][12], DATA_T O[128][12][12], DATA_T W[128][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_142(DATA_T I[128][12][12], DATA_T O[128][12][12], DATA_T W[3][128]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 128; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_142(DATA_T I[128][12][12], DATA_T O[128][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_143(DATA_T I[128][12][12], DATA_T O[160][12][12], DATA_T W[160][128][1][7]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<7; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_143(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[3][160]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 160; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_143(DATA_T I[160][12][12], DATA_T O[160][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_141(DATA_T I[1088][12][12], DATA_T O[192][12][12], DATA_T W[192][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_144(DATA_T I[160][12][12], DATA_T O[192][12][12], DATA_T W[192][160][7][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 7)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<160; k++) {
					for (i = 0; i<7; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_141(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_144(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_141(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_144(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block17_17_mixed(DATA_T I1[192][12][12], DATA_T I2[192][12][12], DATA_T O[384][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_block17_17_conv(DATA_T I[384][12][12], DATA_T O[1088][12][12], DATA_T W[1088][384][1][1], DATA_T B[1088]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 12 + p && y + j < 12 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block17_17(DATA_T I1[1088][12][12], DATA_T I2[1088][12][12], DATA_T O[1088][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+1088; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=1088;
			for(k = ch; k < ch+1088; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block17_17_ac(DATA_T I[1088][12][12], DATA_T O[1088][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_146(DATA_T I[1088][12][12], DATA_T O[128][12][12], DATA_T W[128][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_146(DATA_T I[128][12][12], DATA_T O[128][12][12], DATA_T W[3][128]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 128; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_146(DATA_T I[128][12][12], DATA_T O[128][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_147(DATA_T I[128][12][12], DATA_T O[160][12][12], DATA_T W[160][128][1][7]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<7; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_147(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[3][160]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 160; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_147(DATA_T I[160][12][12], DATA_T O[160][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_145(DATA_T I[1088][12][12], DATA_T O[192][12][12], DATA_T W[192][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_148(DATA_T I[160][12][12], DATA_T O[192][12][12], DATA_T W[192][160][7][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 7)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<160; k++) {
					for (i = 0; i<7; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_145(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_148(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_145(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_148(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block17_18_mixed(DATA_T I1[192][12][12], DATA_T I2[192][12][12], DATA_T O[384][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_block17_18_conv(DATA_T I[384][12][12], DATA_T O[1088][12][12], DATA_T W[1088][384][1][1], DATA_T B[1088]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 12 + p && y + j < 12 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block17_18(DATA_T I1[1088][12][12], DATA_T I2[1088][12][12], DATA_T O[1088][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+1088; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=1088;
			for(k = ch; k < ch+1088; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block17_18_ac(DATA_T I[1088][12][12], DATA_T O[1088][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_150(DATA_T I[1088][12][12], DATA_T O[128][12][12], DATA_T W[128][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_150(DATA_T I[128][12][12], DATA_T O[128][12][12], DATA_T W[3][128]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 128; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_150(DATA_T I[128][12][12], DATA_T O[128][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_151(DATA_T I[128][12][12], DATA_T O[160][12][12], DATA_T W[160][128][1][7]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<7; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_151(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[3][160]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 160; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_151(DATA_T I[160][12][12], DATA_T O[160][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_149(DATA_T I[1088][12][12], DATA_T O[192][12][12], DATA_T W[192][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_152(DATA_T I[160][12][12], DATA_T O[192][12][12], DATA_T W[192][160][7][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 7)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<160; k++) {
					for (i = 0; i<7; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_149(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_152(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_149(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_152(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block17_19_mixed(DATA_T I1[192][12][12], DATA_T I2[192][12][12], DATA_T O[384][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_block17_19_conv(DATA_T I[384][12][12], DATA_T O[1088][12][12], DATA_T W[1088][384][1][1], DATA_T B[1088]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 12 + p && y + j < 12 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block17_19(DATA_T I1[1088][12][12], DATA_T I2[1088][12][12], DATA_T O[1088][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+1088; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=1088;
			for(k = ch; k < ch+1088; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block17_19_ac(DATA_T I[1088][12][12], DATA_T O[1088][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_154(DATA_T I[1088][12][12], DATA_T O[128][12][12], DATA_T W[128][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_154(DATA_T I[128][12][12], DATA_T O[128][12][12], DATA_T W[3][128]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 128; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_154(DATA_T I[128][12][12], DATA_T O[128][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_155(DATA_T I[128][12][12], DATA_T O[160][12][12], DATA_T W[160][128][1][7]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<7; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_155(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[3][160]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 160; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_155(DATA_T I[160][12][12], DATA_T O[160][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_153(DATA_T I[1088][12][12], DATA_T O[192][12][12], DATA_T W[192][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_156(DATA_T I[160][12][12], DATA_T O[192][12][12], DATA_T W[192][160][7][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 7)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<160; k++) {
					for (i = 0; i<7; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_153(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_156(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_153(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_156(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block17_20_mixed(DATA_T I1[192][12][12], DATA_T I2[192][12][12], DATA_T O[384][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_block17_20_conv(DATA_T I[384][12][12], DATA_T O[1088][12][12], DATA_T W[1088][384][1][1], DATA_T B[1088]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 12 + p && y + j < 12 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block17_20(DATA_T I1[1088][12][12], DATA_T I2[1088][12][12], DATA_T O[1088][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+1088; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=1088;
			for(k = ch; k < ch+1088; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block17_20_ac(DATA_T I[1088][12][12], DATA_T O[1088][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_161(DATA_T I[1088][12][12], DATA_T O[256][12][12], DATA_T W[256][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_161(DATA_T I[256][12][12], DATA_T O[256][12][12], DATA_T W[3][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 256; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_161(DATA_T I[256][12][12], DATA_T O[256][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<256; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_157(DATA_T I[1088][12][12], DATA_T O[256][12][12], DATA_T W[256][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_159(DATA_T I[1088][12][12], DATA_T O[256][12][12], DATA_T W[256][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_162(DATA_T I[256][12][12], DATA_T O[288][12][12], DATA_T W[288][256][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 3)/2;
	for (m = 0; m<288; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<256; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 12 + p && y*1 + j < 12 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_157(DATA_T I[256][12][12], DATA_T O[256][12][12], DATA_T W[3][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 256; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_159(DATA_T I[256][12][12], DATA_T O[256][12][12], DATA_T W[3][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 256; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_162(DATA_T I[288][12][12], DATA_T O[288][12][12], DATA_T W[3][288]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 288; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 12; x++){
			for (y = 0; y < 12; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_157(DATA_T I[256][12][12], DATA_T O[256][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<256; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_159(DATA_T I[256][12][12], DATA_T O[256][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<256; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_162(DATA_T I[288][12][12], DATA_T O[288][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<288; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_158(DATA_T I[256][12][12], DATA_T O[384][5][5], DATA_T W[384][256][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<384; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<256; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i <= 12 && y + j <= 12) {
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

void SW_conv2d_160(DATA_T I[256][12][12], DATA_T O[288][5][5], DATA_T W[288][256][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<288; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<256; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i <= 12 && y + j <= 12) {
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

void SW_conv2d_163(DATA_T I[288][12][12], DATA_T O[320][5][5], DATA_T W[320][288][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<320; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<288; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i <= 12 && y + j <= 12) {
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

void SW_batch_normalization_158(DATA_T I[384][5][5], DATA_T O[384][5][5], DATA_T W[3][384]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 384; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_160(DATA_T I[288][5][5], DATA_T O[288][5][5], DATA_T W[3][288]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 288; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_163(DATA_T I[320][5][5], DATA_T O[320][5][5], DATA_T W[3][320]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 320; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_158(DATA_T I[384][5][5], DATA_T O[384][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<384; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_160(DATA_T I[288][5][5], DATA_T O[288][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<288; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_163(DATA_T I[320][5][5], DATA_T O[320][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<320; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_max_pooling2d_4(DATA_T I[1088][12][12], DATA_T O[1088][5][5])
{
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
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
void SW_mixed_7a(DATA_T I1[384][5][5], DATA_T I2[288][5][5], DATA_T I3[320][5][5], DATA_T I4[1088][5][5], DATA_T O[2080][5][5]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 5; x++) {
		for(y = 0; y < 5; y++) {
			ch=0;
			for(k = ch; k < ch+384; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=384;
			for(k = ch; k < ch+288; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
			ch+=288;
			for(k = ch; k < ch+320; k++){
				O[k][x][y] = I3[k-ch][x][y];
			}
			ch+=320;
			for(k = ch; k < ch+1088; k++){
				O[k][x][y] = I4[k-ch][x][y];
			}
		}
	}

}
void SW_conv2d_165(DATA_T I[2080][5][5], DATA_T O[192][5][5], DATA_T W[192][2080][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<2080; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_165(DATA_T I[192][5][5], DATA_T O[192][5][5], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_165(DATA_T I[192][5][5], DATA_T O[192][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_166(DATA_T I[192][5][5], DATA_T O[224][5][5], DATA_T W[224][192][1][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<224; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<192; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_166(DATA_T I[224][5][5], DATA_T O[224][5][5], DATA_T W[3][224]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 224; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_166(DATA_T I[224][5][5], DATA_T O[224][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<224; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_164(DATA_T I[2080][5][5], DATA_T O[192][5][5], DATA_T W[192][2080][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<2080; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_167(DATA_T I[224][5][5], DATA_T O[256][5][5], DATA_T W[256][224][3][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 3)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<224; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_164(DATA_T I[192][5][5], DATA_T O[192][5][5], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_167(DATA_T I[256][5][5], DATA_T O[256][5][5], DATA_T W[3][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 256; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_164(DATA_T I[192][5][5], DATA_T O[192][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_167(DATA_T I[256][5][5], DATA_T O[256][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<256; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block8_1_mixed(DATA_T I1[192][5][5], DATA_T I2[256][5][5], DATA_T O[448][5][5]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 5; x++) {
		for(y = 0; y < 5; y++) {
			ch=0;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+256; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_block8_1_conv(DATA_T I[448][5][5], DATA_T O[2080][5][5], DATA_T W[2080][448][1][1], DATA_T B[2080]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<2080; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<448; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 5 + p && y + j < 5 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block8_1(DATA_T I1[2080][5][5], DATA_T I2[2080][5][5], DATA_T O[2080][5][5]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 5; x++) {
		for(y = 0; y < 5; y++) {
			ch=0;
			for(k = ch; k < ch+2080; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=2080;
			for(k = ch; k < ch+2080; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block8_1_ac(DATA_T I[2080][5][5], DATA_T O[2080][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<2080; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_169(DATA_T I[2080][5][5], DATA_T O[192][5][5], DATA_T W[192][2080][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<2080; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_169(DATA_T I[192][5][5], DATA_T O[192][5][5], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_169(DATA_T I[192][5][5], DATA_T O[192][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_170(DATA_T I[192][5][5], DATA_T O[224][5][5], DATA_T W[224][192][1][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<224; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<192; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_170(DATA_T I[224][5][5], DATA_T O[224][5][5], DATA_T W[3][224]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 224; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_170(DATA_T I[224][5][5], DATA_T O[224][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<224; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_168(DATA_T I[2080][5][5], DATA_T O[192][5][5], DATA_T W[192][2080][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<2080; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_171(DATA_T I[224][5][5], DATA_T O[256][5][5], DATA_T W[256][224][3][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 3)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<224; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_168(DATA_T I[192][5][5], DATA_T O[192][5][5], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_171(DATA_T I[256][5][5], DATA_T O[256][5][5], DATA_T W[3][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 256; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_168(DATA_T I[192][5][5], DATA_T O[192][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_171(DATA_T I[256][5][5], DATA_T O[256][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<256; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block8_2_mixed(DATA_T I1[192][5][5], DATA_T I2[256][5][5], DATA_T O[448][5][5]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 5; x++) {
		for(y = 0; y < 5; y++) {
			ch=0;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+256; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_block8_2_conv(DATA_T I[448][5][5], DATA_T O[2080][5][5], DATA_T W[2080][448][1][1], DATA_T B[2080]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<2080; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<448; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 5 + p && y + j < 5 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block8_2(DATA_T I1[2080][5][5], DATA_T I2[2080][5][5], DATA_T O[2080][5][5]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 5; x++) {
		for(y = 0; y < 5; y++) {
			ch=0;
			for(k = ch; k < ch+2080; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=2080;
			for(k = ch; k < ch+2080; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block8_2_ac(DATA_T I[2080][5][5], DATA_T O[2080][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<2080; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_173(DATA_T I[2080][5][5], DATA_T O[192][5][5], DATA_T W[192][2080][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<2080; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_173(DATA_T I[192][5][5], DATA_T O[192][5][5], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_173(DATA_T I[192][5][5], DATA_T O[192][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_174(DATA_T I[192][5][5], DATA_T O[224][5][5], DATA_T W[224][192][1][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<224; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<192; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_174(DATA_T I[224][5][5], DATA_T O[224][5][5], DATA_T W[3][224]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 224; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_174(DATA_T I[224][5][5], DATA_T O[224][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<224; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_172(DATA_T I[2080][5][5], DATA_T O[192][5][5], DATA_T W[192][2080][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<2080; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_175(DATA_T I[224][5][5], DATA_T O[256][5][5], DATA_T W[256][224][3][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 3)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<224; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_172(DATA_T I[192][5][5], DATA_T O[192][5][5], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_175(DATA_T I[256][5][5], DATA_T O[256][5][5], DATA_T W[3][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 256; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_172(DATA_T I[192][5][5], DATA_T O[192][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_175(DATA_T I[256][5][5], DATA_T O[256][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<256; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block8_3_mixed(DATA_T I1[192][5][5], DATA_T I2[256][5][5], DATA_T O[448][5][5]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 5; x++) {
		for(y = 0; y < 5; y++) {
			ch=0;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+256; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_block8_3_conv(DATA_T I[448][5][5], DATA_T O[2080][5][5], DATA_T W[2080][448][1][1], DATA_T B[2080]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<2080; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<448; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 5 + p && y + j < 5 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block8_3(DATA_T I1[2080][5][5], DATA_T I2[2080][5][5], DATA_T O[2080][5][5]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 5; x++) {
		for(y = 0; y < 5; y++) {
			ch=0;
			for(k = ch; k < ch+2080; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=2080;
			for(k = ch; k < ch+2080; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block8_3_ac(DATA_T I[2080][5][5], DATA_T O[2080][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<2080; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_177(DATA_T I[2080][5][5], DATA_T O[192][5][5], DATA_T W[192][2080][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<2080; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_177(DATA_T I[192][5][5], DATA_T O[192][5][5], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_177(DATA_T I[192][5][5], DATA_T O[192][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_178(DATA_T I[192][5][5], DATA_T O[224][5][5], DATA_T W[224][192][1][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<224; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<192; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_178(DATA_T I[224][5][5], DATA_T O[224][5][5], DATA_T W[3][224]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 224; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_178(DATA_T I[224][5][5], DATA_T O[224][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<224; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_176(DATA_T I[2080][5][5], DATA_T O[192][5][5], DATA_T W[192][2080][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<2080; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_179(DATA_T I[224][5][5], DATA_T O[256][5][5], DATA_T W[256][224][3][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 3)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<224; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_176(DATA_T I[192][5][5], DATA_T O[192][5][5], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_179(DATA_T I[256][5][5], DATA_T O[256][5][5], DATA_T W[3][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 256; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_176(DATA_T I[192][5][5], DATA_T O[192][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_179(DATA_T I[256][5][5], DATA_T O[256][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<256; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block8_4_mixed(DATA_T I1[192][5][5], DATA_T I2[256][5][5], DATA_T O[448][5][5]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 5; x++) {
		for(y = 0; y < 5; y++) {
			ch=0;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+256; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_block8_4_conv(DATA_T I[448][5][5], DATA_T O[2080][5][5], DATA_T W[2080][448][1][1], DATA_T B[2080]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<2080; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<448; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 5 + p && y + j < 5 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block8_4(DATA_T I1[2080][5][5], DATA_T I2[2080][5][5], DATA_T O[2080][5][5]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 5; x++) {
		for(y = 0; y < 5; y++) {
			ch=0;
			for(k = ch; k < ch+2080; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=2080;
			for(k = ch; k < ch+2080; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block8_4_ac(DATA_T I[2080][5][5], DATA_T O[2080][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<2080; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_181(DATA_T I[2080][5][5], DATA_T O[192][5][5], DATA_T W[192][2080][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<2080; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_181(DATA_T I[192][5][5], DATA_T O[192][5][5], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_181(DATA_T I[192][5][5], DATA_T O[192][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_182(DATA_T I[192][5][5], DATA_T O[224][5][5], DATA_T W[224][192][1][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<224; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<192; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_182(DATA_T I[224][5][5], DATA_T O[224][5][5], DATA_T W[3][224]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 224; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_182(DATA_T I[224][5][5], DATA_T O[224][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<224; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_180(DATA_T I[2080][5][5], DATA_T O[192][5][5], DATA_T W[192][2080][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<2080; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_183(DATA_T I[224][5][5], DATA_T O[256][5][5], DATA_T W[256][224][3][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 3)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<224; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_180(DATA_T I[192][5][5], DATA_T O[192][5][5], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_183(DATA_T I[256][5][5], DATA_T O[256][5][5], DATA_T W[3][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 256; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_180(DATA_T I[192][5][5], DATA_T O[192][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_183(DATA_T I[256][5][5], DATA_T O[256][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<256; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block8_5_mixed(DATA_T I1[192][5][5], DATA_T I2[256][5][5], DATA_T O[448][5][5]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 5; x++) {
		for(y = 0; y < 5; y++) {
			ch=0;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+256; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_block8_5_conv(DATA_T I[448][5][5], DATA_T O[2080][5][5], DATA_T W[2080][448][1][1], DATA_T B[2080]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<2080; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<448; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 5 + p && y + j < 5 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block8_5(DATA_T I1[2080][5][5], DATA_T I2[2080][5][5], DATA_T O[2080][5][5]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 5; x++) {
		for(y = 0; y < 5; y++) {
			ch=0;
			for(k = ch; k < ch+2080; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=2080;
			for(k = ch; k < ch+2080; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block8_5_ac(DATA_T I[2080][5][5], DATA_T O[2080][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<2080; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_185(DATA_T I[2080][5][5], DATA_T O[192][5][5], DATA_T W[192][2080][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<2080; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_185(DATA_T I[192][5][5], DATA_T O[192][5][5], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_185(DATA_T I[192][5][5], DATA_T O[192][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_186(DATA_T I[192][5][5], DATA_T O[224][5][5], DATA_T W[224][192][1][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<224; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<192; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_186(DATA_T I[224][5][5], DATA_T O[224][5][5], DATA_T W[3][224]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 224; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_186(DATA_T I[224][5][5], DATA_T O[224][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<224; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_184(DATA_T I[2080][5][5], DATA_T O[192][5][5], DATA_T W[192][2080][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<2080; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_187(DATA_T I[224][5][5], DATA_T O[256][5][5], DATA_T W[256][224][3][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 3)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<224; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_184(DATA_T I[192][5][5], DATA_T O[192][5][5], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_187(DATA_T I[256][5][5], DATA_T O[256][5][5], DATA_T W[3][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 256; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_184(DATA_T I[192][5][5], DATA_T O[192][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_187(DATA_T I[256][5][5], DATA_T O[256][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<256; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block8_6_mixed(DATA_T I1[192][5][5], DATA_T I2[256][5][5], DATA_T O[448][5][5]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 5; x++) {
		for(y = 0; y < 5; y++) {
			ch=0;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+256; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_block8_6_conv(DATA_T I[448][5][5], DATA_T O[2080][5][5], DATA_T W[2080][448][1][1], DATA_T B[2080]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<2080; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<448; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 5 + p && y + j < 5 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block8_6(DATA_T I1[2080][5][5], DATA_T I2[2080][5][5], DATA_T O[2080][5][5]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 5; x++) {
		for(y = 0; y < 5; y++) {
			ch=0;
			for(k = ch; k < ch+2080; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=2080;
			for(k = ch; k < ch+2080; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block8_6_ac(DATA_T I[2080][5][5], DATA_T O[2080][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<2080; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_189(DATA_T I[2080][5][5], DATA_T O[192][5][5], DATA_T W[192][2080][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<2080; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_189(DATA_T I[192][5][5], DATA_T O[192][5][5], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_189(DATA_T I[192][5][5], DATA_T O[192][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_190(DATA_T I[192][5][5], DATA_T O[224][5][5], DATA_T W[224][192][1][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<224; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<192; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_190(DATA_T I[224][5][5], DATA_T O[224][5][5], DATA_T W[3][224]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 224; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_190(DATA_T I[224][5][5], DATA_T O[224][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<224; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_188(DATA_T I[2080][5][5], DATA_T O[192][5][5], DATA_T W[192][2080][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<2080; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_191(DATA_T I[224][5][5], DATA_T O[256][5][5], DATA_T W[256][224][3][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 3)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<224; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_188(DATA_T I[192][5][5], DATA_T O[192][5][5], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_191(DATA_T I[256][5][5], DATA_T O[256][5][5], DATA_T W[3][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 256; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_188(DATA_T I[192][5][5], DATA_T O[192][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_191(DATA_T I[256][5][5], DATA_T O[256][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<256; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block8_7_mixed(DATA_T I1[192][5][5], DATA_T I2[256][5][5], DATA_T O[448][5][5]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 5; x++) {
		for(y = 0; y < 5; y++) {
			ch=0;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+256; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_block8_7_conv(DATA_T I[448][5][5], DATA_T O[2080][5][5], DATA_T W[2080][448][1][1], DATA_T B[2080]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<2080; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<448; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 5 + p && y + j < 5 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block8_7(DATA_T I1[2080][5][5], DATA_T I2[2080][5][5], DATA_T O[2080][5][5]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 5; x++) {
		for(y = 0; y < 5; y++) {
			ch=0;
			for(k = ch; k < ch+2080; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=2080;
			for(k = ch; k < ch+2080; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block8_7_ac(DATA_T I[2080][5][5], DATA_T O[2080][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<2080; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_193(DATA_T I[2080][5][5], DATA_T O[192][5][5], DATA_T W[192][2080][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<2080; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_193(DATA_T I[192][5][5], DATA_T O[192][5][5], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_193(DATA_T I[192][5][5], DATA_T O[192][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_194(DATA_T I[192][5][5], DATA_T O[224][5][5], DATA_T W[224][192][1][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<224; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<192; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_194(DATA_T I[224][5][5], DATA_T O[224][5][5], DATA_T W[3][224]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 224; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_194(DATA_T I[224][5][5], DATA_T O[224][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<224; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_192(DATA_T I[2080][5][5], DATA_T O[192][5][5], DATA_T W[192][2080][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<2080; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_195(DATA_T I[224][5][5], DATA_T O[256][5][5], DATA_T W[256][224][3][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 3)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<224; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_192(DATA_T I[192][5][5], DATA_T O[192][5][5], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_195(DATA_T I[256][5][5], DATA_T O[256][5][5], DATA_T W[3][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 256; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_192(DATA_T I[192][5][5], DATA_T O[192][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_195(DATA_T I[256][5][5], DATA_T O[256][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<256; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block8_8_mixed(DATA_T I1[192][5][5], DATA_T I2[256][5][5], DATA_T O[448][5][5]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 5; x++) {
		for(y = 0; y < 5; y++) {
			ch=0;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+256; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_block8_8_conv(DATA_T I[448][5][5], DATA_T O[2080][5][5], DATA_T W[2080][448][1][1], DATA_T B[2080]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<2080; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<448; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 5 + p && y + j < 5 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block8_8(DATA_T I1[2080][5][5], DATA_T I2[2080][5][5], DATA_T O[2080][5][5]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 5; x++) {
		for(y = 0; y < 5; y++) {
			ch=0;
			for(k = ch; k < ch+2080; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=2080;
			for(k = ch; k < ch+2080; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block8_8_ac(DATA_T I[2080][5][5], DATA_T O[2080][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<2080; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_197(DATA_T I[2080][5][5], DATA_T O[192][5][5], DATA_T W[192][2080][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<2080; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_197(DATA_T I[192][5][5], DATA_T O[192][5][5], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_197(DATA_T I[192][5][5], DATA_T O[192][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_198(DATA_T I[192][5][5], DATA_T O[224][5][5], DATA_T W[224][192][1][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<224; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<192; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_198(DATA_T I[224][5][5], DATA_T O[224][5][5], DATA_T W[3][224]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 224; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_198(DATA_T I[224][5][5], DATA_T O[224][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<224; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_196(DATA_T I[2080][5][5], DATA_T O[192][5][5], DATA_T W[192][2080][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<2080; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_199(DATA_T I[224][5][5], DATA_T O[256][5][5], DATA_T W[256][224][3][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 3)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<224; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_196(DATA_T I[192][5][5], DATA_T O[192][5][5], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_199(DATA_T I[256][5][5], DATA_T O[256][5][5], DATA_T W[3][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 256; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_196(DATA_T I[192][5][5], DATA_T O[192][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_199(DATA_T I[256][5][5], DATA_T O[256][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<256; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block8_9_mixed(DATA_T I1[192][5][5], DATA_T I2[256][5][5], DATA_T O[448][5][5]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 5; x++) {
		for(y = 0; y < 5; y++) {
			ch=0;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+256; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_block8_9_conv(DATA_T I[448][5][5], DATA_T O[2080][5][5], DATA_T W[2080][448][1][1], DATA_T B[2080]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<2080; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<448; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 5 + p && y + j < 5 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block8_9(DATA_T I1[2080][5][5], DATA_T I2[2080][5][5], DATA_T O[2080][5][5]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 5; x++) {
		for(y = 0; y < 5; y++) {
			ch=0;
			for(k = ch; k < ch+2080; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=2080;
			for(k = ch; k < ch+2080; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_block8_9_ac(DATA_T I[2080][5][5], DATA_T O[2080][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<2080; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_201(DATA_T I[2080][5][5], DATA_T O[192][5][5], DATA_T W[192][2080][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<2080; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_201(DATA_T I[192][5][5], DATA_T O[192][5][5], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_201(DATA_T I[192][5][5], DATA_T O[192][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_202(DATA_T I[192][5][5], DATA_T O[224][5][5], DATA_T W[224][192][1][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<224; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<192; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_202(DATA_T I[224][5][5], DATA_T O[224][5][5], DATA_T W[3][224]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 224; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_202(DATA_T I[224][5][5], DATA_T O[224][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<224; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_200(DATA_T I[2080][5][5], DATA_T O[192][5][5], DATA_T W[192][2080][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<2080; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_203(DATA_T I[224][5][5], DATA_T O[256][5][5], DATA_T W[256][224][3][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 3)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<224; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_batch_normalization_200(DATA_T I[192][5][5], DATA_T O[192][5][5], DATA_T W[3][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_batch_normalization_203(DATA_T I[256][5][5], DATA_T O[256][5][5], DATA_T W[3][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 256; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_activation_200(DATA_T I[192][5][5], DATA_T O[192][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_activation_203(DATA_T I[256][5][5], DATA_T O[256][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<256; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block8_10_mixed(DATA_T I1[192][5][5], DATA_T I2[256][5][5], DATA_T O[448][5][5]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 5; x++) {
		for(y = 0; y < 5; y++) {
			ch=0;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+256; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_block8_10_conv(DATA_T I[448][5][5], DATA_T O[2080][5][5], DATA_T W[2080][448][1][1], DATA_T B[2080]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<2080; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<448; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 5 + p && y + j < 5 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block8_10(DATA_T I1[2080][5][5], DATA_T I2[2080][5][5], DATA_T O[2080][5][5]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 5; x++) {
		for(y = 0; y < 5; y++) {
			ch=0;
			for(k = ch; k < ch+2080; k++){
				if(I1[k][x][y]<0)
					O[k][x][y] = -I1[k][x][y];
				else
					O[k][x][y] = I1[k][x][y];
			}
			ch+=2080;
			for(k = ch; k < ch+2080; k++){
				if(I2[k][x][y]<0)
					O[k][x][y] = -I2[k][x][y];
				else
					O[k][x][y] = I2[k][x][y];
			}
		}
	}

}
void SW_conv_7b(DATA_T I[2080][5][5], DATA_T O[1536][5][5], DATA_T W[1536][2080][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<1536; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<2080; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 5 + p && y*1 + j < 5 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv_7b_bn(DATA_T I[1536][5][5], DATA_T O[1536][5][5], DATA_T W[3][1536]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 1536; m++){
        DATA_T beta = W[0][m];
        DATA_T mean = W[1][m];
        DATA_T var = W[2][m];

		for (x = 0; x < 5; x++){
			for (y = 0; y < 5; y++){
				O[m][x][y] = ((I[m][x][y]-mean)/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_conv_7b_ac(DATA_T I[1536][5][5], DATA_T O[1536][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1536; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_avg_pool(DATA_T I[1536][5][5], DATA_T O[1536]) {
	int m, x, y;
	double avg;
	int div = 5 * 5;
	for (m = 0; m < 1536; m++){
		avg = 0;
		for (x = 0; x < 5; x++) {
			for (y = 0; y < 5; y++) {
				avg += I[m][x][y];
			}
		}
		O[m] = avg/div;
	}
}
void SW_predictions(DATA_T I[1536], DATA_T O[1000], DATA_T W[1000][1536], DATA_T B[1000])
{
    //Dense
	int m, c;
	for(m=0; m<1000; m++){
        O[m] = 0;
		for (c = 0; c < 1536; c++){
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
	static DATA_T W28[96][192][1][1];
	static DATA_T W29[64][48][5][5];
	static DATA_T W30[96][96][3][3];
	static DATA_T W31[64][192][1][1];
	static DATA_T W32[3][96];
	static DATA_T W33[3][64];
	static DATA_T W34[3][96];
	static DATA_T W35[3][64];
	static DATA_T W41[32][320][1][1];
	static DATA_T W42[3][32];
	static DATA_T W44[32][320][1][1];
	static DATA_T W45[48][32][3][3];
	static DATA_T W46[3][32];
	static DATA_T W47[3][48];
	static DATA_T W50[32][320][1][1];
	static DATA_T W51[32][32][3][3];
	static DATA_T W52[64][48][3][3];
	static DATA_T W53[3][32];
	static DATA_T W54[3][32];
	static DATA_T W55[3][64];
	static DATA_T W60[320][128][1][1];
	static DATA_T B60[320];
	static DATA_T W63[32][320][1][1];
	static DATA_T W64[3][32];
	static DATA_T W66[32][320][1][1];
	static DATA_T W67[48][32][3][3];
	static DATA_T W68[3][32];
	static DATA_T W69[3][48];
	static DATA_T W72[32][320][1][1];
	static DATA_T W73[32][32][3][3];
	static DATA_T W74[64][48][3][3];
	static DATA_T W75[3][32];
	static DATA_T W76[3][32];
	static DATA_T W77[3][64];
	static DATA_T W82[320][128][1][1];
	static DATA_T B82[320];
	static DATA_T W85[32][320][1][1];
	static DATA_T W86[3][32];
	static DATA_T W88[32][320][1][1];
	static DATA_T W89[48][32][3][3];
	static DATA_T W90[3][32];
	static DATA_T W91[3][48];
	static DATA_T W94[32][320][1][1];
	static DATA_T W95[32][32][3][3];
	static DATA_T W96[64][48][3][3];
	static DATA_T W97[3][32];
	static DATA_T W98[3][32];
	static DATA_T W99[3][64];
	static DATA_T W104[320][128][1][1];
	static DATA_T B104[320];
	static DATA_T W107[32][320][1][1];
	static DATA_T W108[3][32];
	static DATA_T W110[32][320][1][1];
	static DATA_T W111[48][32][3][3];
	static DATA_T W112[3][32];
	static DATA_T W113[3][48];
	static DATA_T W116[32][320][1][1];
	static DATA_T W117[32][32][3][3];
	static DATA_T W118[64][48][3][3];
	static DATA_T W119[3][32];
	static DATA_T W120[3][32];
	static DATA_T W121[3][64];
	static DATA_T W126[320][128][1][1];
	static DATA_T B126[320];
	static DATA_T W129[32][320][1][1];
	static DATA_T W130[3][32];
	static DATA_T W132[32][320][1][1];
	static DATA_T W133[48][32][3][3];
	static DATA_T W134[3][32];
	static DATA_T W135[3][48];
	static DATA_T W138[32][320][1][1];
	static DATA_T W139[32][32][3][3];
	static DATA_T W140[64][48][3][3];
	static DATA_T W141[3][32];
	static DATA_T W142[3][32];
	static DATA_T W143[3][64];
	static DATA_T W148[320][128][1][1];
	static DATA_T B148[320];
	static DATA_T W151[32][320][1][1];
	static DATA_T W152[3][32];
	static DATA_T W154[32][320][1][1];
	static DATA_T W155[48][32][3][3];
	static DATA_T W156[3][32];
	static DATA_T W157[3][48];
	static DATA_T W160[32][320][1][1];
	static DATA_T W161[32][32][3][3];
	static DATA_T W162[64][48][3][3];
	static DATA_T W163[3][32];
	static DATA_T W164[3][32];
	static DATA_T W165[3][64];
	static DATA_T W170[320][128][1][1];
	static DATA_T B170[320];
	static DATA_T W173[32][320][1][1];
	static DATA_T W174[3][32];
	static DATA_T W176[32][320][1][1];
	static DATA_T W177[48][32][3][3];
	static DATA_T W178[3][32];
	static DATA_T W179[3][48];
	static DATA_T W182[32][320][1][1];
	static DATA_T W183[32][32][3][3];
	static DATA_T W184[64][48][3][3];
	static DATA_T W185[3][32];
	static DATA_T W186[3][32];
	static DATA_T W187[3][64];
	static DATA_T W192[320][128][1][1];
	static DATA_T B192[320];
	static DATA_T W195[32][320][1][1];
	static DATA_T W196[3][32];
	static DATA_T W198[32][320][1][1];
	static DATA_T W199[48][32][3][3];
	static DATA_T W200[3][32];
	static DATA_T W201[3][48];
	static DATA_T W204[32][320][1][1];
	static DATA_T W205[32][32][3][3];
	static DATA_T W206[64][48][3][3];
	static DATA_T W207[3][32];
	static DATA_T W208[3][32];
	static DATA_T W209[3][64];
	static DATA_T W214[320][128][1][1];
	static DATA_T B214[320];
	static DATA_T W217[32][320][1][1];
	static DATA_T W218[3][32];
	static DATA_T W220[32][320][1][1];
	static DATA_T W221[48][32][3][3];
	static DATA_T W222[3][32];
	static DATA_T W223[3][48];
	static DATA_T W226[32][320][1][1];
	static DATA_T W227[32][32][3][3];
	static DATA_T W228[64][48][3][3];
	static DATA_T W229[3][32];
	static DATA_T W230[3][32];
	static DATA_T W231[3][64];
	static DATA_T W236[320][128][1][1];
	static DATA_T B236[320];
	static DATA_T W239[32][320][1][1];
	static DATA_T W240[3][32];
	static DATA_T W242[32][320][1][1];
	static DATA_T W243[48][32][3][3];
	static DATA_T W244[3][32];
	static DATA_T W245[3][48];
	static DATA_T W248[32][320][1][1];
	static DATA_T W249[32][32][3][3];
	static DATA_T W250[64][48][3][3];
	static DATA_T W251[3][32];
	static DATA_T W252[3][32];
	static DATA_T W253[3][64];
	static DATA_T W258[320][128][1][1];
	static DATA_T B258[320];
	static DATA_T W261[256][320][1][1];
	static DATA_T W262[3][256];
	static DATA_T W264[256][256][3][3];
	static DATA_T W265[3][256];
	static DATA_T W267[384][320][3][3];
	static DATA_T W268[384][256][3][3];
	static DATA_T W269[3][384];
	static DATA_T W270[3][384];
	static DATA_T W275[128][1088][1][1];
	static DATA_T W276[3][128];
	static DATA_T W278[160][128][1][7];
	static DATA_T W279[3][160];
	static DATA_T W281[192][1088][1][1];
	static DATA_T W282[192][160][7][1];
	static DATA_T W283[3][192];
	static DATA_T W284[3][192];
	static DATA_T W288[1088][384][1][1];
	static DATA_T B288[1088];
	static DATA_T W291[128][1088][1][1];
	static DATA_T W292[3][128];
	static DATA_T W294[160][128][1][7];
	static DATA_T W295[3][160];
	static DATA_T W297[192][1088][1][1];
	static DATA_T W298[192][160][7][1];
	static DATA_T W299[3][192];
	static DATA_T W300[3][192];
	static DATA_T W304[1088][384][1][1];
	static DATA_T B304[1088];
	static DATA_T W307[128][1088][1][1];
	static DATA_T W308[3][128];
	static DATA_T W310[160][128][1][7];
	static DATA_T W311[3][160];
	static DATA_T W313[192][1088][1][1];
	static DATA_T W314[192][160][7][1];
	static DATA_T W315[3][192];
	static DATA_T W316[3][192];
	static DATA_T W320[1088][384][1][1];
	static DATA_T B320[1088];
	static DATA_T W323[128][1088][1][1];
	static DATA_T W324[3][128];
	static DATA_T W326[160][128][1][7];
	static DATA_T W327[3][160];
	static DATA_T W329[192][1088][1][1];
	static DATA_T W330[192][160][7][1];
	static DATA_T W331[3][192];
	static DATA_T W332[3][192];
	static DATA_T W336[1088][384][1][1];
	static DATA_T B336[1088];
	static DATA_T W339[128][1088][1][1];
	static DATA_T W340[3][128];
	static DATA_T W342[160][128][1][7];
	static DATA_T W343[3][160];
	static DATA_T W345[192][1088][1][1];
	static DATA_T W346[192][160][7][1];
	static DATA_T W347[3][192];
	static DATA_T W348[3][192];
	static DATA_T W352[1088][384][1][1];
	static DATA_T B352[1088];
	static DATA_T W355[128][1088][1][1];
	static DATA_T W356[3][128];
	static DATA_T W358[160][128][1][7];
	static DATA_T W359[3][160];
	static DATA_T W361[192][1088][1][1];
	static DATA_T W362[192][160][7][1];
	static DATA_T W363[3][192];
	static DATA_T W364[3][192];
	static DATA_T W368[1088][384][1][1];
	static DATA_T B368[1088];
	static DATA_T W371[128][1088][1][1];
	static DATA_T W372[3][128];
	static DATA_T W374[160][128][1][7];
	static DATA_T W375[3][160];
	static DATA_T W377[192][1088][1][1];
	static DATA_T W378[192][160][7][1];
	static DATA_T W379[3][192];
	static DATA_T W380[3][192];
	static DATA_T W384[1088][384][1][1];
	static DATA_T B384[1088];
	static DATA_T W387[128][1088][1][1];
	static DATA_T W388[3][128];
	static DATA_T W390[160][128][1][7];
	static DATA_T W391[3][160];
	static DATA_T W393[192][1088][1][1];
	static DATA_T W394[192][160][7][1];
	static DATA_T W395[3][192];
	static DATA_T W396[3][192];
	static DATA_T W400[1088][384][1][1];
	static DATA_T B400[1088];
	static DATA_T W403[128][1088][1][1];
	static DATA_T W404[3][128];
	static DATA_T W406[160][128][1][7];
	static DATA_T W407[3][160];
	static DATA_T W409[192][1088][1][1];
	static DATA_T W410[192][160][7][1];
	static DATA_T W411[3][192];
	static DATA_T W412[3][192];
	static DATA_T W416[1088][384][1][1];
	static DATA_T B416[1088];
	static DATA_T W419[128][1088][1][1];
	static DATA_T W420[3][128];
	static DATA_T W422[160][128][1][7];
	static DATA_T W423[3][160];
	static DATA_T W425[192][1088][1][1];
	static DATA_T W426[192][160][7][1];
	static DATA_T W427[3][192];
	static DATA_T W428[3][192];
	static DATA_T W432[1088][384][1][1];
	static DATA_T B432[1088];
	static DATA_T W435[128][1088][1][1];
	static DATA_T W436[3][128];
	static DATA_T W438[160][128][1][7];
	static DATA_T W439[3][160];
	static DATA_T W441[192][1088][1][1];
	static DATA_T W442[192][160][7][1];
	static DATA_T W443[3][192];
	static DATA_T W444[3][192];
	static DATA_T W448[1088][384][1][1];
	static DATA_T B448[1088];
	static DATA_T W451[128][1088][1][1];
	static DATA_T W452[3][128];
	static DATA_T W454[160][128][1][7];
	static DATA_T W455[3][160];
	static DATA_T W457[192][1088][1][1];
	static DATA_T W458[192][160][7][1];
	static DATA_T W459[3][192];
	static DATA_T W460[3][192];
	static DATA_T W464[1088][384][1][1];
	static DATA_T B464[1088];
	static DATA_T W467[128][1088][1][1];
	static DATA_T W468[3][128];
	static DATA_T W470[160][128][1][7];
	static DATA_T W471[3][160];
	static DATA_T W473[192][1088][1][1];
	static DATA_T W474[192][160][7][1];
	static DATA_T W475[3][192];
	static DATA_T W476[3][192];
	static DATA_T W480[1088][384][1][1];
	static DATA_T B480[1088];
	static DATA_T W483[128][1088][1][1];
	static DATA_T W484[3][128];
	static DATA_T W486[160][128][1][7];
	static DATA_T W487[3][160];
	static DATA_T W489[192][1088][1][1];
	static DATA_T W490[192][160][7][1];
	static DATA_T W491[3][192];
	static DATA_T W492[3][192];
	static DATA_T W496[1088][384][1][1];
	static DATA_T B496[1088];
	static DATA_T W499[128][1088][1][1];
	static DATA_T W500[3][128];
	static DATA_T W502[160][128][1][7];
	static DATA_T W503[3][160];
	static DATA_T W505[192][1088][1][1];
	static DATA_T W506[192][160][7][1];
	static DATA_T W507[3][192];
	static DATA_T W508[3][192];
	static DATA_T W512[1088][384][1][1];
	static DATA_T B512[1088];
	static DATA_T W515[128][1088][1][1];
	static DATA_T W516[3][128];
	static DATA_T W518[160][128][1][7];
	static DATA_T W519[3][160];
	static DATA_T W521[192][1088][1][1];
	static DATA_T W522[192][160][7][1];
	static DATA_T W523[3][192];
	static DATA_T W524[3][192];
	static DATA_T W528[1088][384][1][1];
	static DATA_T B528[1088];
	static DATA_T W531[128][1088][1][1];
	static DATA_T W532[3][128];
	static DATA_T W534[160][128][1][7];
	static DATA_T W535[3][160];
	static DATA_T W537[192][1088][1][1];
	static DATA_T W538[192][160][7][1];
	static DATA_T W539[3][192];
	static DATA_T W540[3][192];
	static DATA_T W544[1088][384][1][1];
	static DATA_T B544[1088];
	static DATA_T W547[128][1088][1][1];
	static DATA_T W548[3][128];
	static DATA_T W550[160][128][1][7];
	static DATA_T W551[3][160];
	static DATA_T W553[192][1088][1][1];
	static DATA_T W554[192][160][7][1];
	static DATA_T W555[3][192];
	static DATA_T W556[3][192];
	static DATA_T W560[1088][384][1][1];
	static DATA_T B560[1088];
	static DATA_T W563[128][1088][1][1];
	static DATA_T W564[3][128];
	static DATA_T W566[160][128][1][7];
	static DATA_T W567[3][160];
	static DATA_T W569[192][1088][1][1];
	static DATA_T W570[192][160][7][1];
	static DATA_T W571[3][192];
	static DATA_T W572[3][192];
	static DATA_T W576[1088][384][1][1];
	static DATA_T B576[1088];
	static DATA_T W579[128][1088][1][1];
	static DATA_T W580[3][128];
	static DATA_T W582[160][128][1][7];
	static DATA_T W583[3][160];
	static DATA_T W585[192][1088][1][1];
	static DATA_T W586[192][160][7][1];
	static DATA_T W587[3][192];
	static DATA_T W588[3][192];
	static DATA_T W592[1088][384][1][1];
	static DATA_T B592[1088];
	static DATA_T W595[256][1088][1][1];
	static DATA_T W596[3][256];
	static DATA_T W598[256][1088][1][1];
	static DATA_T W599[256][1088][1][1];
	static DATA_T W600[288][256][3][3];
	static DATA_T W601[3][256];
	static DATA_T W602[3][256];
	static DATA_T W603[3][288];
	static DATA_T W607[384][256][3][3];
	static DATA_T W608[288][256][3][3];
	static DATA_T W609[320][288][3][3];
	static DATA_T W610[3][384];
	static DATA_T W611[3][288];
	static DATA_T W612[3][320];
	static DATA_T W618[192][2080][1][1];
	static DATA_T W619[3][192];
	static DATA_T W621[224][192][1][3];
	static DATA_T W622[3][224];
	static DATA_T W624[192][2080][1][1];
	static DATA_T W625[256][224][3][1];
	static DATA_T W626[3][192];
	static DATA_T W627[3][256];
	static DATA_T W631[2080][448][1][1];
	static DATA_T B631[2080];
	static DATA_T W634[192][2080][1][1];
	static DATA_T W635[3][192];
	static DATA_T W637[224][192][1][3];
	static DATA_T W638[3][224];
	static DATA_T W640[192][2080][1][1];
	static DATA_T W641[256][224][3][1];
	static DATA_T W642[3][192];
	static DATA_T W643[3][256];
	static DATA_T W647[2080][448][1][1];
	static DATA_T B647[2080];
	static DATA_T W650[192][2080][1][1];
	static DATA_T W651[3][192];
	static DATA_T W653[224][192][1][3];
	static DATA_T W654[3][224];
	static DATA_T W656[192][2080][1][1];
	static DATA_T W657[256][224][3][1];
	static DATA_T W658[3][192];
	static DATA_T W659[3][256];
	static DATA_T W663[2080][448][1][1];
	static DATA_T B663[2080];
	static DATA_T W666[192][2080][1][1];
	static DATA_T W667[3][192];
	static DATA_T W669[224][192][1][3];
	static DATA_T W670[3][224];
	static DATA_T W672[192][2080][1][1];
	static DATA_T W673[256][224][3][1];
	static DATA_T W674[3][192];
	static DATA_T W675[3][256];
	static DATA_T W679[2080][448][1][1];
	static DATA_T B679[2080];
	static DATA_T W682[192][2080][1][1];
	static DATA_T W683[3][192];
	static DATA_T W685[224][192][1][3];
	static DATA_T W686[3][224];
	static DATA_T W688[192][2080][1][1];
	static DATA_T W689[256][224][3][1];
	static DATA_T W690[3][192];
	static DATA_T W691[3][256];
	static DATA_T W695[2080][448][1][1];
	static DATA_T B695[2080];
	static DATA_T W698[192][2080][1][1];
	static DATA_T W699[3][192];
	static DATA_T W701[224][192][1][3];
	static DATA_T W702[3][224];
	static DATA_T W704[192][2080][1][1];
	static DATA_T W705[256][224][3][1];
	static DATA_T W706[3][192];
	static DATA_T W707[3][256];
	static DATA_T W711[2080][448][1][1];
	static DATA_T B711[2080];
	static DATA_T W714[192][2080][1][1];
	static DATA_T W715[3][192];
	static DATA_T W717[224][192][1][3];
	static DATA_T W718[3][224];
	static DATA_T W720[192][2080][1][1];
	static DATA_T W721[256][224][3][1];
	static DATA_T W722[3][192];
	static DATA_T W723[3][256];
	static DATA_T W727[2080][448][1][1];
	static DATA_T B727[2080];
	static DATA_T W730[192][2080][1][1];
	static DATA_T W731[3][192];
	static DATA_T W733[224][192][1][3];
	static DATA_T W734[3][224];
	static DATA_T W736[192][2080][1][1];
	static DATA_T W737[256][224][3][1];
	static DATA_T W738[3][192];
	static DATA_T W739[3][256];
	static DATA_T W743[2080][448][1][1];
	static DATA_T B743[2080];
	static DATA_T W746[192][2080][1][1];
	static DATA_T W747[3][192];
	static DATA_T W749[224][192][1][3];
	static DATA_T W750[3][224];
	static DATA_T W752[192][2080][1][1];
	static DATA_T W753[256][224][3][1];
	static DATA_T W754[3][192];
	static DATA_T W755[3][256];
	static DATA_T W759[2080][448][1][1];
	static DATA_T B759[2080];
	static DATA_T W762[192][2080][1][1];
	static DATA_T W763[3][192];
	static DATA_T W765[224][192][1][3];
	static DATA_T W766[3][224];
	static DATA_T W768[192][2080][1][1];
	static DATA_T W769[256][224][3][1];
	static DATA_T W770[3][192];
	static DATA_T W771[3][256];
	static DATA_T W775[2080][448][1][1];
	static DATA_T B775[2080];
	static DATA_T W777[1536][2080][1][1];
	static DATA_T W778[3][1536];
	static DATA_T B781[1000];
	static DATA_T W781[1000][1536];
	

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
	static DATA_T O28_SW[96][25][25];
	static DATA_T O29_SW[64][25][25];
	static DATA_T O30_SW[96][25][25];
	static DATA_T O31_SW[64][25][25];
	static DATA_T O32_SW[96][25][25];
	static DATA_T O33_SW[64][25][25];
	static DATA_T O34_SW[96][25][25];
	static DATA_T O35_SW[64][25][25];
	static DATA_T O36_SW[96][25][25];
	static DATA_T O37_SW[64][25][25];
	static DATA_T O38_SW[96][25][25];
	static DATA_T O39_SW[64][25][25];
	static DATA_T O40_SW[320][25][25];
	static DATA_T O41_SW[32][25][25];
	static DATA_T O42_SW[32][25][25];
	static DATA_T O43_SW[32][25][25];
	static DATA_T O44_SW[32][25][25];
	static DATA_T O45_SW[48][25][25];
	static DATA_T O46_SW[32][25][25];
	static DATA_T O47_SW[48][25][25];
	static DATA_T O48_SW[32][25][25];
	static DATA_T O49_SW[48][25][25];
	static DATA_T O50_SW[32][25][25];
	static DATA_T O51_SW[32][25][25];
	static DATA_T O52_SW[64][25][25];
	static DATA_T O53_SW[32][25][25];
	static DATA_T O54_SW[32][25][25];
	static DATA_T O55_SW[64][25][25];
	static DATA_T O56_SW[32][25][25];
	static DATA_T O57_SW[32][25][25];
	static DATA_T O58_SW[64][25][25];
	static DATA_T O59_SW[128][25][25];
	static DATA_T O60_SW[320][25][25];
	static DATA_T O61_SW[320][25][25];
	static DATA_T O62_SW[320][25][25];
	static DATA_T O63_SW[32][25][25];
	static DATA_T O64_SW[32][25][25];
	static DATA_T O65_SW[32][25][25];
	static DATA_T O66_SW[32][25][25];
	static DATA_T O67_SW[48][25][25];
	static DATA_T O68_SW[32][25][25];
	static DATA_T O69_SW[48][25][25];
	static DATA_T O70_SW[32][25][25];
	static DATA_T O71_SW[48][25][25];
	static DATA_T O72_SW[32][25][25];
	static DATA_T O73_SW[32][25][25];
	static DATA_T O74_SW[64][25][25];
	static DATA_T O75_SW[32][25][25];
	static DATA_T O76_SW[32][25][25];
	static DATA_T O77_SW[64][25][25];
	static DATA_T O78_SW[32][25][25];
	static DATA_T O79_SW[32][25][25];
	static DATA_T O80_SW[64][25][25];
	static DATA_T O81_SW[128][25][25];
	static DATA_T O82_SW[320][25][25];
	static DATA_T O83_SW[320][25][25];
	static DATA_T O84_SW[320][25][25];
	static DATA_T O85_SW[32][25][25];
	static DATA_T O86_SW[32][25][25];
	static DATA_T O87_SW[32][25][25];
	static DATA_T O88_SW[32][25][25];
	static DATA_T O89_SW[48][25][25];
	static DATA_T O90_SW[32][25][25];
	static DATA_T O91_SW[48][25][25];
	static DATA_T O92_SW[32][25][25];
	static DATA_T O93_SW[48][25][25];
	static DATA_T O94_SW[32][25][25];
	static DATA_T O95_SW[32][25][25];
	static DATA_T O96_SW[64][25][25];
	static DATA_T O97_SW[32][25][25];
	static DATA_T O98_SW[32][25][25];
	static DATA_T O99_SW[64][25][25];
	static DATA_T O100_SW[32][25][25];
	static DATA_T O101_SW[32][25][25];
	static DATA_T O102_SW[64][25][25];
	static DATA_T O103_SW[128][25][25];
	static DATA_T O104_SW[320][25][25];
	static DATA_T O105_SW[320][25][25];
	static DATA_T O106_SW[320][25][25];
	static DATA_T O107_SW[32][25][25];
	static DATA_T O108_SW[32][25][25];
	static DATA_T O109_SW[32][25][25];
	static DATA_T O110_SW[32][25][25];
	static DATA_T O111_SW[48][25][25];
	static DATA_T O112_SW[32][25][25];
	static DATA_T O113_SW[48][25][25];
	static DATA_T O114_SW[32][25][25];
	static DATA_T O115_SW[48][25][25];
	static DATA_T O116_SW[32][25][25];
	static DATA_T O117_SW[32][25][25];
	static DATA_T O118_SW[64][25][25];
	static DATA_T O119_SW[32][25][25];
	static DATA_T O120_SW[32][25][25];
	static DATA_T O121_SW[64][25][25];
	static DATA_T O122_SW[32][25][25];
	static DATA_T O123_SW[32][25][25];
	static DATA_T O124_SW[64][25][25];
	static DATA_T O125_SW[128][25][25];
	static DATA_T O126_SW[320][25][25];
	static DATA_T O127_SW[320][25][25];
	static DATA_T O128_SW[320][25][25];
	static DATA_T O129_SW[32][25][25];
	static DATA_T O130_SW[32][25][25];
	static DATA_T O131_SW[32][25][25];
	static DATA_T O132_SW[32][25][25];
	static DATA_T O133_SW[48][25][25];
	static DATA_T O134_SW[32][25][25];
	static DATA_T O135_SW[48][25][25];
	static DATA_T O136_SW[32][25][25];
	static DATA_T O137_SW[48][25][25];
	static DATA_T O138_SW[32][25][25];
	static DATA_T O139_SW[32][25][25];
	static DATA_T O140_SW[64][25][25];
	static DATA_T O141_SW[32][25][25];
	static DATA_T O142_SW[32][25][25];
	static DATA_T O143_SW[64][25][25];
	static DATA_T O144_SW[32][25][25];
	static DATA_T O145_SW[32][25][25];
	static DATA_T O146_SW[64][25][25];
	static DATA_T O147_SW[128][25][25];
	static DATA_T O148_SW[320][25][25];
	static DATA_T O149_SW[320][25][25];
	static DATA_T O150_SW[320][25][25];
	static DATA_T O151_SW[32][25][25];
	static DATA_T O152_SW[32][25][25];
	static DATA_T O153_SW[32][25][25];
	static DATA_T O154_SW[32][25][25];
	static DATA_T O155_SW[48][25][25];
	static DATA_T O156_SW[32][25][25];
	static DATA_T O157_SW[48][25][25];
	static DATA_T O158_SW[32][25][25];
	static DATA_T O159_SW[48][25][25];
	static DATA_T O160_SW[32][25][25];
	static DATA_T O161_SW[32][25][25];
	static DATA_T O162_SW[64][25][25];
	static DATA_T O163_SW[32][25][25];
	static DATA_T O164_SW[32][25][25];
	static DATA_T O165_SW[64][25][25];
	static DATA_T O166_SW[32][25][25];
	static DATA_T O167_SW[32][25][25];
	static DATA_T O168_SW[64][25][25];
	static DATA_T O169_SW[128][25][25];
	static DATA_T O170_SW[320][25][25];
	static DATA_T O171_SW[320][25][25];
	static DATA_T O172_SW[320][25][25];
	static DATA_T O173_SW[32][25][25];
	static DATA_T O174_SW[32][25][25];
	static DATA_T O175_SW[32][25][25];
	static DATA_T O176_SW[32][25][25];
	static DATA_T O177_SW[48][25][25];
	static DATA_T O178_SW[32][25][25];
	static DATA_T O179_SW[48][25][25];
	static DATA_T O180_SW[32][25][25];
	static DATA_T O181_SW[48][25][25];
	static DATA_T O182_SW[32][25][25];
	static DATA_T O183_SW[32][25][25];
	static DATA_T O184_SW[64][25][25];
	static DATA_T O185_SW[32][25][25];
	static DATA_T O186_SW[32][25][25];
	static DATA_T O187_SW[64][25][25];
	static DATA_T O188_SW[32][25][25];
	static DATA_T O189_SW[32][25][25];
	static DATA_T O190_SW[64][25][25];
	static DATA_T O191_SW[128][25][25];
	static DATA_T O192_SW[320][25][25];
	static DATA_T O193_SW[320][25][25];
	static DATA_T O194_SW[320][25][25];
	static DATA_T O195_SW[32][25][25];
	static DATA_T O196_SW[32][25][25];
	static DATA_T O197_SW[32][25][25];
	static DATA_T O198_SW[32][25][25];
	static DATA_T O199_SW[48][25][25];
	static DATA_T O200_SW[32][25][25];
	static DATA_T O201_SW[48][25][25];
	static DATA_T O202_SW[32][25][25];
	static DATA_T O203_SW[48][25][25];
	static DATA_T O204_SW[32][25][25];
	static DATA_T O205_SW[32][25][25];
	static DATA_T O206_SW[64][25][25];
	static DATA_T O207_SW[32][25][25];
	static DATA_T O208_SW[32][25][25];
	static DATA_T O209_SW[64][25][25];
	static DATA_T O210_SW[32][25][25];
	static DATA_T O211_SW[32][25][25];
	static DATA_T O212_SW[64][25][25];
	static DATA_T O213_SW[128][25][25];
	static DATA_T O214_SW[320][25][25];
	static DATA_T O215_SW[320][25][25];
	static DATA_T O216_SW[320][25][25];
	static DATA_T O217_SW[32][25][25];
	static DATA_T O218_SW[32][25][25];
	static DATA_T O219_SW[32][25][25];
	static DATA_T O220_SW[32][25][25];
	static DATA_T O221_SW[48][25][25];
	static DATA_T O222_SW[32][25][25];
	static DATA_T O223_SW[48][25][25];
	static DATA_T O224_SW[32][25][25];
	static DATA_T O225_SW[48][25][25];
	static DATA_T O226_SW[32][25][25];
	static DATA_T O227_SW[32][25][25];
	static DATA_T O228_SW[64][25][25];
	static DATA_T O229_SW[32][25][25];
	static DATA_T O230_SW[32][25][25];
	static DATA_T O231_SW[64][25][25];
	static DATA_T O232_SW[32][25][25];
	static DATA_T O233_SW[32][25][25];
	static DATA_T O234_SW[64][25][25];
	static DATA_T O235_SW[128][25][25];
	static DATA_T O236_SW[320][25][25];
	static DATA_T O237_SW[320][25][25];
	static DATA_T O238_SW[320][25][25];
	static DATA_T O239_SW[32][25][25];
	static DATA_T O240_SW[32][25][25];
	static DATA_T O241_SW[32][25][25];
	static DATA_T O242_SW[32][25][25];
	static DATA_T O243_SW[48][25][25];
	static DATA_T O244_SW[32][25][25];
	static DATA_T O245_SW[48][25][25];
	static DATA_T O246_SW[32][25][25];
	static DATA_T O247_SW[48][25][25];
	static DATA_T O248_SW[32][25][25];
	static DATA_T O249_SW[32][25][25];
	static DATA_T O250_SW[64][25][25];
	static DATA_T O251_SW[32][25][25];
	static DATA_T O252_SW[32][25][25];
	static DATA_T O253_SW[64][25][25];
	static DATA_T O254_SW[32][25][25];
	static DATA_T O255_SW[32][25][25];
	static DATA_T O256_SW[64][25][25];
	static DATA_T O257_SW[128][25][25];
	static DATA_T O258_SW[320][25][25];
	static DATA_T O259_SW[320][25][25];
	static DATA_T O260_SW[320][25][25];
	static DATA_T O261_SW[256][25][25];
	static DATA_T O262_SW[256][25][25];
	static DATA_T O263_SW[256][25][25];
	static DATA_T O264_SW[256][25][25];
	static DATA_T O265_SW[256][25][25];
	static DATA_T O266_SW[256][25][25];
	static DATA_T O267_SW[384][12][12];
	static DATA_T O268_SW[384][12][12];
	static DATA_T O269_SW[384][12][12];
	static DATA_T O270_SW[384][12][12];
	static DATA_T O271_SW[384][12][12];
	static DATA_T O272_SW[384][12][12];
	static DATA_T O273_SW[320][12][12];
	static DATA_T O274_SW[1088][12][12];
	static DATA_T O275_SW[128][12][12];
	static DATA_T O276_SW[128][12][12];
	static DATA_T O277_SW[128][12][12];
	static DATA_T O278_SW[160][12][12];
	static DATA_T O279_SW[160][12][12];
	static DATA_T O280_SW[160][12][12];
	static DATA_T O281_SW[192][12][12];
	static DATA_T O282_SW[192][12][12];
	static DATA_T O283_SW[192][12][12];
	static DATA_T O284_SW[192][12][12];
	static DATA_T O285_SW[192][12][12];
	static DATA_T O286_SW[192][12][12];
	static DATA_T O287_SW[384][12][12];
	static DATA_T O288_SW[1088][12][12];
	static DATA_T O289_SW[1088][12][12];
	static DATA_T O290_SW[1088][12][12];
	static DATA_T O291_SW[128][12][12];
	static DATA_T O292_SW[128][12][12];
	static DATA_T O293_SW[128][12][12];
	static DATA_T O294_SW[160][12][12];
	static DATA_T O295_SW[160][12][12];
	static DATA_T O296_SW[160][12][12];
	static DATA_T O297_SW[192][12][12];
	static DATA_T O298_SW[192][12][12];
	static DATA_T O299_SW[192][12][12];
	static DATA_T O300_SW[192][12][12];
	static DATA_T O301_SW[192][12][12];
	static DATA_T O302_SW[192][12][12];
	static DATA_T O303_SW[384][12][12];
	static DATA_T O304_SW[1088][12][12];
	static DATA_T O305_SW[1088][12][12];
	static DATA_T O306_SW[1088][12][12];
	static DATA_T O307_SW[128][12][12];
	static DATA_T O308_SW[128][12][12];
	static DATA_T O309_SW[128][12][12];
	static DATA_T O310_SW[160][12][12];
	static DATA_T O311_SW[160][12][12];
	static DATA_T O312_SW[160][12][12];
	static DATA_T O313_SW[192][12][12];
	static DATA_T O314_SW[192][12][12];
	static DATA_T O315_SW[192][12][12];
	static DATA_T O316_SW[192][12][12];
	static DATA_T O317_SW[192][12][12];
	static DATA_T O318_SW[192][12][12];
	static DATA_T O319_SW[384][12][12];
	static DATA_T O320_SW[1088][12][12];
	static DATA_T O321_SW[1088][12][12];
	static DATA_T O322_SW[1088][12][12];
	static DATA_T O323_SW[128][12][12];
	static DATA_T O324_SW[128][12][12];
	static DATA_T O325_SW[128][12][12];
	static DATA_T O326_SW[160][12][12];
	static DATA_T O327_SW[160][12][12];
	static DATA_T O328_SW[160][12][12];
	static DATA_T O329_SW[192][12][12];
	static DATA_T O330_SW[192][12][12];
	static DATA_T O331_SW[192][12][12];
	static DATA_T O332_SW[192][12][12];
	static DATA_T O333_SW[192][12][12];
	static DATA_T O334_SW[192][12][12];
	static DATA_T O335_SW[384][12][12];
	static DATA_T O336_SW[1088][12][12];
	static DATA_T O337_SW[1088][12][12];
	static DATA_T O338_SW[1088][12][12];
	static DATA_T O339_SW[128][12][12];
	static DATA_T O340_SW[128][12][12];
	static DATA_T O341_SW[128][12][12];
	static DATA_T O342_SW[160][12][12];
	static DATA_T O343_SW[160][12][12];
	static DATA_T O344_SW[160][12][12];
	static DATA_T O345_SW[192][12][12];
	static DATA_T O346_SW[192][12][12];
	static DATA_T O347_SW[192][12][12];
	static DATA_T O348_SW[192][12][12];
	static DATA_T O349_SW[192][12][12];
	static DATA_T O350_SW[192][12][12];
	static DATA_T O351_SW[384][12][12];
	static DATA_T O352_SW[1088][12][12];
	static DATA_T O353_SW[1088][12][12];
	static DATA_T O354_SW[1088][12][12];
	static DATA_T O355_SW[128][12][12];
	static DATA_T O356_SW[128][12][12];
	static DATA_T O357_SW[128][12][12];
	static DATA_T O358_SW[160][12][12];
	static DATA_T O359_SW[160][12][12];
	static DATA_T O360_SW[160][12][12];
	static DATA_T O361_SW[192][12][12];
	static DATA_T O362_SW[192][12][12];
	static DATA_T O363_SW[192][12][12];
	static DATA_T O364_SW[192][12][12];
	static DATA_T O365_SW[192][12][12];
	static DATA_T O366_SW[192][12][12];
	static DATA_T O367_SW[384][12][12];
	static DATA_T O368_SW[1088][12][12];
	static DATA_T O369_SW[1088][12][12];
	static DATA_T O370_SW[1088][12][12];
	static DATA_T O371_SW[128][12][12];
	static DATA_T O372_SW[128][12][12];
	static DATA_T O373_SW[128][12][12];
	static DATA_T O374_SW[160][12][12];
	static DATA_T O375_SW[160][12][12];
	static DATA_T O376_SW[160][12][12];
	static DATA_T O377_SW[192][12][12];
	static DATA_T O378_SW[192][12][12];
	static DATA_T O379_SW[192][12][12];
	static DATA_T O380_SW[192][12][12];
	static DATA_T O381_SW[192][12][12];
	static DATA_T O382_SW[192][12][12];
	static DATA_T O383_SW[384][12][12];
	static DATA_T O384_SW[1088][12][12];
	static DATA_T O385_SW[1088][12][12];
	static DATA_T O386_SW[1088][12][12];
	static DATA_T O387_SW[128][12][12];
	static DATA_T O388_SW[128][12][12];
	static DATA_T O389_SW[128][12][12];
	static DATA_T O390_SW[160][12][12];
	static DATA_T O391_SW[160][12][12];
	static DATA_T O392_SW[160][12][12];
	static DATA_T O393_SW[192][12][12];
	static DATA_T O394_SW[192][12][12];
	static DATA_T O395_SW[192][12][12];
	static DATA_T O396_SW[192][12][12];
	static DATA_T O397_SW[192][12][12];
	static DATA_T O398_SW[192][12][12];
	static DATA_T O399_SW[384][12][12];
	static DATA_T O400_SW[1088][12][12];
	static DATA_T O401_SW[1088][12][12];
	static DATA_T O402_SW[1088][12][12];
	static DATA_T O403_SW[128][12][12];
	static DATA_T O404_SW[128][12][12];
	static DATA_T O405_SW[128][12][12];
	static DATA_T O406_SW[160][12][12];
	static DATA_T O407_SW[160][12][12];
	static DATA_T O408_SW[160][12][12];
	static DATA_T O409_SW[192][12][12];
	static DATA_T O410_SW[192][12][12];
	static DATA_T O411_SW[192][12][12];
	static DATA_T O412_SW[192][12][12];
	static DATA_T O413_SW[192][12][12];
	static DATA_T O414_SW[192][12][12];
	static DATA_T O415_SW[384][12][12];
	static DATA_T O416_SW[1088][12][12];
	static DATA_T O417_SW[1088][12][12];
	static DATA_T O418_SW[1088][12][12];
	static DATA_T O419_SW[128][12][12];
	static DATA_T O420_SW[128][12][12];
	static DATA_T O421_SW[128][12][12];
	static DATA_T O422_SW[160][12][12];
	static DATA_T O423_SW[160][12][12];
	static DATA_T O424_SW[160][12][12];
	static DATA_T O425_SW[192][12][12];
	static DATA_T O426_SW[192][12][12];
	static DATA_T O427_SW[192][12][12];
	static DATA_T O428_SW[192][12][12];
	static DATA_T O429_SW[192][12][12];
	static DATA_T O430_SW[192][12][12];
	static DATA_T O431_SW[384][12][12];
	static DATA_T O432_SW[1088][12][12];
	static DATA_T O433_SW[1088][12][12];
	static DATA_T O434_SW[1088][12][12];
	static DATA_T O435_SW[128][12][12];
	static DATA_T O436_SW[128][12][12];
	static DATA_T O437_SW[128][12][12];
	static DATA_T O438_SW[160][12][12];
	static DATA_T O439_SW[160][12][12];
	static DATA_T O440_SW[160][12][12];
	static DATA_T O441_SW[192][12][12];
	static DATA_T O442_SW[192][12][12];
	static DATA_T O443_SW[192][12][12];
	static DATA_T O444_SW[192][12][12];
	static DATA_T O445_SW[192][12][12];
	static DATA_T O446_SW[192][12][12];
	static DATA_T O447_SW[384][12][12];
	static DATA_T O448_SW[1088][12][12];
	static DATA_T O449_SW[1088][12][12];
	static DATA_T O450_SW[1088][12][12];
	static DATA_T O451_SW[128][12][12];
	static DATA_T O452_SW[128][12][12];
	static DATA_T O453_SW[128][12][12];
	static DATA_T O454_SW[160][12][12];
	static DATA_T O455_SW[160][12][12];
	static DATA_T O456_SW[160][12][12];
	static DATA_T O457_SW[192][12][12];
	static DATA_T O458_SW[192][12][12];
	static DATA_T O459_SW[192][12][12];
	static DATA_T O460_SW[192][12][12];
	static DATA_T O461_SW[192][12][12];
	static DATA_T O462_SW[192][12][12];
	static DATA_T O463_SW[384][12][12];
	static DATA_T O464_SW[1088][12][12];
	static DATA_T O465_SW[1088][12][12];
	static DATA_T O466_SW[1088][12][12];
	static DATA_T O467_SW[128][12][12];
	static DATA_T O468_SW[128][12][12];
	static DATA_T O469_SW[128][12][12];
	static DATA_T O470_SW[160][12][12];
	static DATA_T O471_SW[160][12][12];
	static DATA_T O472_SW[160][12][12];
	static DATA_T O473_SW[192][12][12];
	static DATA_T O474_SW[192][12][12];
	static DATA_T O475_SW[192][12][12];
	static DATA_T O476_SW[192][12][12];
	static DATA_T O477_SW[192][12][12];
	static DATA_T O478_SW[192][12][12];
	static DATA_T O479_SW[384][12][12];
	static DATA_T O480_SW[1088][12][12];
	static DATA_T O481_SW[1088][12][12];
	static DATA_T O482_SW[1088][12][12];
	static DATA_T O483_SW[128][12][12];
	static DATA_T O484_SW[128][12][12];
	static DATA_T O485_SW[128][12][12];
	static DATA_T O486_SW[160][12][12];
	static DATA_T O487_SW[160][12][12];
	static DATA_T O488_SW[160][12][12];
	static DATA_T O489_SW[192][12][12];
	static DATA_T O490_SW[192][12][12];
	static DATA_T O491_SW[192][12][12];
	static DATA_T O492_SW[192][12][12];
	static DATA_T O493_SW[192][12][12];
	static DATA_T O494_SW[192][12][12];
	static DATA_T O495_SW[384][12][12];
	static DATA_T O496_SW[1088][12][12];
	static DATA_T O497_SW[1088][12][12];
	static DATA_T O498_SW[1088][12][12];
	static DATA_T O499_SW[128][12][12];
	static DATA_T O500_SW[128][12][12];
	static DATA_T O501_SW[128][12][12];
	static DATA_T O502_SW[160][12][12];
	static DATA_T O503_SW[160][12][12];
	static DATA_T O504_SW[160][12][12];
	static DATA_T O505_SW[192][12][12];
	static DATA_T O506_SW[192][12][12];
	static DATA_T O507_SW[192][12][12];
	static DATA_T O508_SW[192][12][12];
	static DATA_T O509_SW[192][12][12];
	static DATA_T O510_SW[192][12][12];
	static DATA_T O511_SW[384][12][12];
	static DATA_T O512_SW[1088][12][12];
	static DATA_T O513_SW[1088][12][12];
	static DATA_T O514_SW[1088][12][12];
	static DATA_T O515_SW[128][12][12];
	static DATA_T O516_SW[128][12][12];
	static DATA_T O517_SW[128][12][12];
	static DATA_T O518_SW[160][12][12];
	static DATA_T O519_SW[160][12][12];
	static DATA_T O520_SW[160][12][12];
	static DATA_T O521_SW[192][12][12];
	static DATA_T O522_SW[192][12][12];
	static DATA_T O523_SW[192][12][12];
	static DATA_T O524_SW[192][12][12];
	static DATA_T O525_SW[192][12][12];
	static DATA_T O526_SW[192][12][12];
	static DATA_T O527_SW[384][12][12];
	static DATA_T O528_SW[1088][12][12];
	static DATA_T O529_SW[1088][12][12];
	static DATA_T O530_SW[1088][12][12];
	static DATA_T O531_SW[128][12][12];
	static DATA_T O532_SW[128][12][12];
	static DATA_T O533_SW[128][12][12];
	static DATA_T O534_SW[160][12][12];
	static DATA_T O535_SW[160][12][12];
	static DATA_T O536_SW[160][12][12];
	static DATA_T O537_SW[192][12][12];
	static DATA_T O538_SW[192][12][12];
	static DATA_T O539_SW[192][12][12];
	static DATA_T O540_SW[192][12][12];
	static DATA_T O541_SW[192][12][12];
	static DATA_T O542_SW[192][12][12];
	static DATA_T O543_SW[384][12][12];
	static DATA_T O544_SW[1088][12][12];
	static DATA_T O545_SW[1088][12][12];
	static DATA_T O546_SW[1088][12][12];
	static DATA_T O547_SW[128][12][12];
	static DATA_T O548_SW[128][12][12];
	static DATA_T O549_SW[128][12][12];
	static DATA_T O550_SW[160][12][12];
	static DATA_T O551_SW[160][12][12];
	static DATA_T O552_SW[160][12][12];
	static DATA_T O553_SW[192][12][12];
	static DATA_T O554_SW[192][12][12];
	static DATA_T O555_SW[192][12][12];
	static DATA_T O556_SW[192][12][12];
	static DATA_T O557_SW[192][12][12];
	static DATA_T O558_SW[192][12][12];
	static DATA_T O559_SW[384][12][12];
	static DATA_T O560_SW[1088][12][12];
	static DATA_T O561_SW[1088][12][12];
	static DATA_T O562_SW[1088][12][12];
	static DATA_T O563_SW[128][12][12];
	static DATA_T O564_SW[128][12][12];
	static DATA_T O565_SW[128][12][12];
	static DATA_T O566_SW[160][12][12];
	static DATA_T O567_SW[160][12][12];
	static DATA_T O568_SW[160][12][12];
	static DATA_T O569_SW[192][12][12];
	static DATA_T O570_SW[192][12][12];
	static DATA_T O571_SW[192][12][12];
	static DATA_T O572_SW[192][12][12];
	static DATA_T O573_SW[192][12][12];
	static DATA_T O574_SW[192][12][12];
	static DATA_T O575_SW[384][12][12];
	static DATA_T O576_SW[1088][12][12];
	static DATA_T O577_SW[1088][12][12];
	static DATA_T O578_SW[1088][12][12];
	static DATA_T O579_SW[128][12][12];
	static DATA_T O580_SW[128][12][12];
	static DATA_T O581_SW[128][12][12];
	static DATA_T O582_SW[160][12][12];
	static DATA_T O583_SW[160][12][12];
	static DATA_T O584_SW[160][12][12];
	static DATA_T O585_SW[192][12][12];
	static DATA_T O586_SW[192][12][12];
	static DATA_T O587_SW[192][12][12];
	static DATA_T O588_SW[192][12][12];
	static DATA_T O589_SW[192][12][12];
	static DATA_T O590_SW[192][12][12];
	static DATA_T O591_SW[384][12][12];
	static DATA_T O592_SW[1088][12][12];
	static DATA_T O593_SW[1088][12][12];
	static DATA_T O594_SW[1088][12][12];
	static DATA_T O595_SW[256][12][12];
	static DATA_T O596_SW[256][12][12];
	static DATA_T O597_SW[256][12][12];
	static DATA_T O598_SW[256][12][12];
	static DATA_T O599_SW[256][12][12];
	static DATA_T O600_SW[288][12][12];
	static DATA_T O601_SW[256][12][12];
	static DATA_T O602_SW[256][12][12];
	static DATA_T O603_SW[288][12][12];
	static DATA_T O604_SW[256][12][12];
	static DATA_T O605_SW[256][12][12];
	static DATA_T O606_SW[288][12][12];
	static DATA_T O607_SW[384][5][5];
	static DATA_T O608_SW[288][5][5];
	static DATA_T O609_SW[320][5][5];
	static DATA_T O610_SW[384][5][5];
	static DATA_T O611_SW[288][5][5];
	static DATA_T O612_SW[320][5][5];
	static DATA_T O613_SW[384][5][5];
	static DATA_T O614_SW[288][5][5];
	static DATA_T O615_SW[320][5][5];
	static DATA_T O616_SW[1088][5][5];
	static DATA_T O617_SW[2080][5][5];
	static DATA_T O618_SW[192][5][5];
	static DATA_T O619_SW[192][5][5];
	static DATA_T O620_SW[192][5][5];
	static DATA_T O621_SW[224][5][5];
	static DATA_T O622_SW[224][5][5];
	static DATA_T O623_SW[224][5][5];
	static DATA_T O624_SW[192][5][5];
	static DATA_T O625_SW[256][5][5];
	static DATA_T O626_SW[192][5][5];
	static DATA_T O627_SW[256][5][5];
	static DATA_T O628_SW[192][5][5];
	static DATA_T O629_SW[256][5][5];
	static DATA_T O630_SW[448][5][5];
	static DATA_T O631_SW[2080][5][5];
	static DATA_T O632_SW[2080][5][5];
	static DATA_T O633_SW[2080][5][5];
	static DATA_T O634_SW[192][5][5];
	static DATA_T O635_SW[192][5][5];
	static DATA_T O636_SW[192][5][5];
	static DATA_T O637_SW[224][5][5];
	static DATA_T O638_SW[224][5][5];
	static DATA_T O639_SW[224][5][5];
	static DATA_T O640_SW[192][5][5];
	static DATA_T O641_SW[256][5][5];
	static DATA_T O642_SW[192][5][5];
	static DATA_T O643_SW[256][5][5];
	static DATA_T O644_SW[192][5][5];
	static DATA_T O645_SW[256][5][5];
	static DATA_T O646_SW[448][5][5];
	static DATA_T O647_SW[2080][5][5];
	static DATA_T O648_SW[2080][5][5];
	static DATA_T O649_SW[2080][5][5];
	static DATA_T O650_SW[192][5][5];
	static DATA_T O651_SW[192][5][5];
	static DATA_T O652_SW[192][5][5];
	static DATA_T O653_SW[224][5][5];
	static DATA_T O654_SW[224][5][5];
	static DATA_T O655_SW[224][5][5];
	static DATA_T O656_SW[192][5][5];
	static DATA_T O657_SW[256][5][5];
	static DATA_T O658_SW[192][5][5];
	static DATA_T O659_SW[256][5][5];
	static DATA_T O660_SW[192][5][5];
	static DATA_T O661_SW[256][5][5];
	static DATA_T O662_SW[448][5][5];
	static DATA_T O663_SW[2080][5][5];
	static DATA_T O664_SW[2080][5][5];
	static DATA_T O665_SW[2080][5][5];
	static DATA_T O666_SW[192][5][5];
	static DATA_T O667_SW[192][5][5];
	static DATA_T O668_SW[192][5][5];
	static DATA_T O669_SW[224][5][5];
	static DATA_T O670_SW[224][5][5];
	static DATA_T O671_SW[224][5][5];
	static DATA_T O672_SW[192][5][5];
	static DATA_T O673_SW[256][5][5];
	static DATA_T O674_SW[192][5][5];
	static DATA_T O675_SW[256][5][5];
	static DATA_T O676_SW[192][5][5];
	static DATA_T O677_SW[256][5][5];
	static DATA_T O678_SW[448][5][5];
	static DATA_T O679_SW[2080][5][5];
	static DATA_T O680_SW[2080][5][5];
	static DATA_T O681_SW[2080][5][5];
	static DATA_T O682_SW[192][5][5];
	static DATA_T O683_SW[192][5][5];
	static DATA_T O684_SW[192][5][5];
	static DATA_T O685_SW[224][5][5];
	static DATA_T O686_SW[224][5][5];
	static DATA_T O687_SW[224][5][5];
	static DATA_T O688_SW[192][5][5];
	static DATA_T O689_SW[256][5][5];
	static DATA_T O690_SW[192][5][5];
	static DATA_T O691_SW[256][5][5];
	static DATA_T O692_SW[192][5][5];
	static DATA_T O693_SW[256][5][5];
	static DATA_T O694_SW[448][5][5];
	static DATA_T O695_SW[2080][5][5];
	static DATA_T O696_SW[2080][5][5];
	static DATA_T O697_SW[2080][5][5];
	static DATA_T O698_SW[192][5][5];
	static DATA_T O699_SW[192][5][5];
	static DATA_T O700_SW[192][5][5];
	static DATA_T O701_SW[224][5][5];
	static DATA_T O702_SW[224][5][5];
	static DATA_T O703_SW[224][5][5];
	static DATA_T O704_SW[192][5][5];
	static DATA_T O705_SW[256][5][5];
	static DATA_T O706_SW[192][5][5];
	static DATA_T O707_SW[256][5][5];
	static DATA_T O708_SW[192][5][5];
	static DATA_T O709_SW[256][5][5];
	static DATA_T O710_SW[448][5][5];
	static DATA_T O711_SW[2080][5][5];
	static DATA_T O712_SW[2080][5][5];
	static DATA_T O713_SW[2080][5][5];
	static DATA_T O714_SW[192][5][5];
	static DATA_T O715_SW[192][5][5];
	static DATA_T O716_SW[192][5][5];
	static DATA_T O717_SW[224][5][5];
	static DATA_T O718_SW[224][5][5];
	static DATA_T O719_SW[224][5][5];
	static DATA_T O720_SW[192][5][5];
	static DATA_T O721_SW[256][5][5];
	static DATA_T O722_SW[192][5][5];
	static DATA_T O723_SW[256][5][5];
	static DATA_T O724_SW[192][5][5];
	static DATA_T O725_SW[256][5][5];
	static DATA_T O726_SW[448][5][5];
	static DATA_T O727_SW[2080][5][5];
	static DATA_T O728_SW[2080][5][5];
	static DATA_T O729_SW[2080][5][5];
	static DATA_T O730_SW[192][5][5];
	static DATA_T O731_SW[192][5][5];
	static DATA_T O732_SW[192][5][5];
	static DATA_T O733_SW[224][5][5];
	static DATA_T O734_SW[224][5][5];
	static DATA_T O735_SW[224][5][5];
	static DATA_T O736_SW[192][5][5];
	static DATA_T O737_SW[256][5][5];
	static DATA_T O738_SW[192][5][5];
	static DATA_T O739_SW[256][5][5];
	static DATA_T O740_SW[192][5][5];
	static DATA_T O741_SW[256][5][5];
	static DATA_T O742_SW[448][5][5];
	static DATA_T O743_SW[2080][5][5];
	static DATA_T O744_SW[2080][5][5];
	static DATA_T O745_SW[2080][5][5];
	static DATA_T O746_SW[192][5][5];
	static DATA_T O747_SW[192][5][5];
	static DATA_T O748_SW[192][5][5];
	static DATA_T O749_SW[224][5][5];
	static DATA_T O750_SW[224][5][5];
	static DATA_T O751_SW[224][5][5];
	static DATA_T O752_SW[192][5][5];
	static DATA_T O753_SW[256][5][5];
	static DATA_T O754_SW[192][5][5];
	static DATA_T O755_SW[256][5][5];
	static DATA_T O756_SW[192][5][5];
	static DATA_T O757_SW[256][5][5];
	static DATA_T O758_SW[448][5][5];
	static DATA_T O759_SW[2080][5][5];
	static DATA_T O760_SW[2080][5][5];
	static DATA_T O761_SW[2080][5][5];
	static DATA_T O762_SW[192][5][5];
	static DATA_T O763_SW[192][5][5];
	static DATA_T O764_SW[192][5][5];
	static DATA_T O765_SW[224][5][5];
	static DATA_T O766_SW[224][5][5];
	static DATA_T O767_SW[224][5][5];
	static DATA_T O768_SW[192][5][5];
	static DATA_T O769_SW[256][5][5];
	static DATA_T O770_SW[192][5][5];
	static DATA_T O771_SW[256][5][5];
	static DATA_T O772_SW[192][5][5];
	static DATA_T O773_SW[256][5][5];
	static DATA_T O774_SW[448][5][5];
	static DATA_T O775_SW[2080][5][5];
	static DATA_T O776_SW[2080][5][5];
	static DATA_T O777_SW[1536][5][5];
	static DATA_T O778_SW[1536][5][5];
	static DATA_T O779_SW[1536][5][5];
	static DATA_T O780_SW[1536];
	static DATA_T O781_SW[1000];
	

    FILE *w_stream = fopen(argv[1], "rb");
    if (w_stream == NULL) printf("weight file was not opened");
    FILE *i_stream = fopen(argv[2], "rb");
    if (i_stream == NULL) printf("input file was not opened");
    FILE *o_stream = fopen("../../cpp_generator/inceptionresnetv2/Output/C_output.txt", "w");
    if (o_stream == NULL) printf("Output file was not opened");
    FILE *c_num = fopen("../../cpp_generator/inceptionresnetv2/Output/c_output_num.txt", "w");
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
			for (j = 0; j < 96 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W28[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 96 ; m++) {
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
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W31[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B31[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 96 ; y++) {
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
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W35[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W41[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B41[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W42[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W44[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B44[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 48 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W45[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 48 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B45[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W46[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 48 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W47[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W50[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B50[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W51[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B51[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 48 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W52[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B52[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W53[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W54[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W55[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 320 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W60[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 320 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B60[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W63[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B63[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W64[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W66[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B66[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 48 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W67[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 48 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B67[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W68[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 48 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W69[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W72[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B72[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W73[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B73[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 48 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W74[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B74[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W75[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W76[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W77[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 320 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W82[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 320 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B82[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W85[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B85[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W86[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W88[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B88[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 48 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W89[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 48 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B89[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W90[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 48 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W91[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W94[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B94[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W95[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B95[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 48 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W96[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B96[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W97[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W98[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W99[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 320 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W104[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 320 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B104[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W107[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B107[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W108[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W110[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B110[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 48 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W111[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 48 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B111[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W112[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 48 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W113[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W116[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B116[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W117[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B117[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 48 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W118[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B118[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W119[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W120[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W121[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 320 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W126[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 320 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B126[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W129[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B129[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W130[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W132[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B132[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 48 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W133[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 48 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B133[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W134[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 48 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W135[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W138[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B138[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W139[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B139[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 48 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W140[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B140[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W141[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W142[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W143[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 320 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W148[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 320 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B148[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W151[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B151[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W152[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W154[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B154[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 48 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W155[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 48 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B155[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W156[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 48 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W157[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W160[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B160[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W161[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B161[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 48 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W162[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B162[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W163[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W164[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W165[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 320 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W170[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 320 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B170[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W173[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B173[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W174[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W176[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B176[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 48 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W177[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 48 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B177[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W178[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 48 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W179[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W182[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B182[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W183[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B183[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 48 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W184[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B184[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W185[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W186[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W187[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 320 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W192[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 320 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B192[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W195[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B195[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W196[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W198[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B198[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 48 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W199[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 48 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B199[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W200[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 48 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W201[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W204[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B204[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W205[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B205[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 48 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W206[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B206[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W207[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W208[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W209[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 320 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W214[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 320 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B214[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W217[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B217[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W218[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W220[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B220[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 48 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W221[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 48 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B221[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W222[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 48 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W223[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W226[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B226[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W227[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B227[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 48 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W228[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B228[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W229[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W230[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W231[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 320 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W236[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 320 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B236[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W239[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B239[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W240[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W242[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B242[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 48 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W243[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 48 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B243[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W244[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 48 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W245[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W248[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B248[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W249[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B249[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 48 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W250[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B250[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W251[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W252[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W253[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 320 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W258[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 320 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B258[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W261[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B261[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W262[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W264[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B264[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W265[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 384 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W267[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 384 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B267[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 384 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W268[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 384 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B268[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 384 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W269[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 384 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W270[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W275[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B275[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W276[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W278[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B278[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W279[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W281[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B281[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W282[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B282[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W283[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W284[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 1088 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W288[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 1088 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B288[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W291[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B291[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W292[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W294[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B294[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W295[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W297[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B297[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W298[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B298[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W299[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W300[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 1088 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W304[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 1088 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B304[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W307[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B307[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W308[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W310[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B310[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W311[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W313[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B313[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W314[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B314[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W315[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W316[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 1088 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W320[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 1088 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B320[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W323[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B323[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W324[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W326[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B326[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W327[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W329[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B329[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W330[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B330[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W331[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W332[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 1088 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W336[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 1088 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B336[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W339[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B339[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W340[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W342[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B342[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W343[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W345[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B345[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W346[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B346[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W347[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W348[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 1088 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W352[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 1088 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B352[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W355[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B355[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W356[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W358[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B358[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W359[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W361[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B361[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W362[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B362[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W363[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W364[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 1088 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W368[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 1088 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B368[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W371[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B371[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W372[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W374[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B374[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W375[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W377[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B377[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W378[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B378[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W379[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W380[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 1088 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W384[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 1088 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B384[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W387[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B387[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W388[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W390[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B390[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W391[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W393[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B393[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W394[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B394[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W395[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W396[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 1088 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W400[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 1088 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B400[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W403[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B403[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W404[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W406[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B406[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W407[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W409[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B409[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W410[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B410[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W411[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W412[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 1088 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W416[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 1088 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B416[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W419[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B419[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W420[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W422[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B422[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W423[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W425[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B425[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W426[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B426[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W427[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W428[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 1088 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W432[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 1088 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B432[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W435[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B435[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W436[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W438[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B438[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W439[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W441[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B441[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W442[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B442[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W443[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W444[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 1088 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W448[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 1088 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B448[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W451[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B451[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W452[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W454[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B454[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W455[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W457[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B457[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W458[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B458[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W459[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W460[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 1088 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W464[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 1088 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B464[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W467[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B467[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W468[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W470[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B470[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W471[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W473[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B473[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W474[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B474[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W475[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W476[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 1088 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W480[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 1088 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B480[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W483[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B483[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W484[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W486[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B486[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W487[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W489[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B489[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W490[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B490[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W491[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W492[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 1088 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W496[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 1088 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B496[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W499[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B499[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W500[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W502[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B502[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W503[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W505[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B505[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W506[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B506[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W507[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W508[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 1088 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W512[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 1088 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B512[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W515[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B515[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W516[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W518[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B518[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W519[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W521[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B521[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W522[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B522[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W523[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W524[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 1088 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W528[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 1088 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B528[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W531[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B531[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W532[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W534[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B534[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W535[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W537[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B537[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W538[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B538[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W539[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W540[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 1088 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W544[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 1088 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B544[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W547[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B547[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W548[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W550[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B550[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W551[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W553[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B553[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W554[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B554[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W555[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W556[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 1088 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W560[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 1088 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B560[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W563[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B563[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W564[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W566[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B566[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W567[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W569[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B569[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W570[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B570[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W571[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W572[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 1088 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W576[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 1088 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B576[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W579[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B579[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W580[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W582[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B582[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W583[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W585[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B585[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W586[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B586[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W587[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W588[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 1088 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W592[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 1088 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B592[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W595[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B595[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W596[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W598[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B598[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W599[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B599[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 288 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W600[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 288 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B600[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W601[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W602[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 288 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W603[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 384 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W607[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 384 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B607[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 288 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W608[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 288 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B608[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 288 ; i++) {
			for (j = 0; j < 320 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W609[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 320 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B609[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 384 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W610[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 288 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W611[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 320 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W612[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 2080 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W618[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B618[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W619[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 192 ; i++) {
			for (j = 0; j < 224 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W621[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 224 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B621[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 224 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W622[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 2080 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W624[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B624[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 224 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W625[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B625[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W626[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W627[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 448 ; i++) {
			for (j = 0; j < 2080 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W631[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 2080 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B631[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 2080 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W634[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B634[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W635[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 192 ; i++) {
			for (j = 0; j < 224 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W637[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 224 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B637[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 224 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W638[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 2080 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W640[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B640[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 224 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W641[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B641[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W642[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W643[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 448 ; i++) {
			for (j = 0; j < 2080 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W647[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 2080 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B647[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 2080 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W650[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B650[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W651[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 192 ; i++) {
			for (j = 0; j < 224 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W653[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 224 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B653[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 224 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W654[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 2080 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W656[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B656[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 224 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W657[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B657[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W658[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W659[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 448 ; i++) {
			for (j = 0; j < 2080 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W663[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 2080 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B663[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 2080 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W666[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B666[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W667[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 192 ; i++) {
			for (j = 0; j < 224 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W669[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 224 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B669[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 224 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W670[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 2080 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W672[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B672[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 224 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W673[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B673[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W674[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W675[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 448 ; i++) {
			for (j = 0; j < 2080 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W679[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 2080 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B679[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 2080 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W682[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B682[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W683[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 192 ; i++) {
			for (j = 0; j < 224 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W685[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 224 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B685[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 224 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W686[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 2080 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W688[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B688[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 224 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W689[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B689[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W690[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W691[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 448 ; i++) {
			for (j = 0; j < 2080 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W695[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 2080 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B695[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 2080 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W698[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B698[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W699[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 192 ; i++) {
			for (j = 0; j < 224 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W701[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 224 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B701[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 224 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W702[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 2080 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W704[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B704[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 224 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W705[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B705[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W706[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W707[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 448 ; i++) {
			for (j = 0; j < 2080 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W711[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 2080 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B711[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 2080 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W714[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B714[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W715[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 192 ; i++) {
			for (j = 0; j < 224 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W717[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 224 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B717[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 224 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W718[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 2080 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W720[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B720[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 224 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W721[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B721[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W722[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W723[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 448 ; i++) {
			for (j = 0; j < 2080 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W727[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 2080 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B727[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 2080 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W730[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B730[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W731[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 192 ; i++) {
			for (j = 0; j < 224 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W733[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 224 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B733[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 224 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W734[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 2080 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W736[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B736[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 224 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W737[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B737[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W738[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W739[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 448 ; i++) {
			for (j = 0; j < 2080 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W743[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 2080 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B743[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 2080 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W746[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B746[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W747[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 192 ; i++) {
			for (j = 0; j < 224 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W749[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 224 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B749[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 224 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W750[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 2080 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W752[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B752[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 224 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W753[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B753[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W754[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W755[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 448 ; i++) {
			for (j = 0; j < 2080 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W759[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 2080 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B759[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 2080 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W762[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B762[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W763[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 192 ; i++) {
			for (j = 0; j < 224 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W765[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 224 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B765[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 224 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W766[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 2080 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W768[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B768[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 224 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W769[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B769[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W770[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W771[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 448 ; i++) {
			for (j = 0; j < 2080 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W775[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 2080 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B775[m] = (DATA_T) trash;
}


	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 2080 ; i++) {
			for (j = 0; j < 1536 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W777[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 1536 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B777[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 1536 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W778[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1536 ; m++) {
	for (k = 0; k < 1000 ; k++) {
		fread(&trash, sizeof(int), 1, w_stream);
        W781[k][m] = (DATA_T) trash;
	}
}


for (m = 0; m < 1000 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B781[m] = (DATA_T) trash;
}

	
    printf("[C_verifier.cpp]Finish Initialization");

    printf("[C_verifier.cpp]InputLayer\n\n");
	printf("[C_verifier.cpp]Calculate Conv2D1\n\n");
	SW_conv2d_1(O0_SW,O1_SW,W1);
	printf("[C_verifier.cpp]Calculate BatchNormalization2\n\n");
	SW_batch_normalization_1(O1_SW,O2_SW, W2);
	printf("[C_verifier.cpp]Calculate Activation(Relu)3\n\n");
	SW_activation_1(O2_SW,O3_SW);
	printf("[C_verifier.cpp]Calculate Conv2D4\n\n");
	SW_conv2d_2(O3_SW,O4_SW,W4);
	printf("[C_verifier.cpp]Calculate BatchNormalization5\n\n");
	SW_batch_normalization_2(O4_SW,O5_SW, W5);
	printf("[C_verifier.cpp]Calculate Activation(Relu)6\n\n");
	SW_activation_2(O5_SW,O6_SW);
	printf("[C_verifier.cpp]Calculate Conv2D7\n\n");
	SW_conv2d_3(O6_SW,O7_SW,W7);
	printf("[C_verifier.cpp]Calculate BatchNormalization8\n\n");
	SW_batch_normalization_3(O7_SW,O8_SW, W8);
	printf("[C_verifier.cpp]Calculate Activation(Relu)9\n\n");
	SW_activation_3(O8_SW,O9_SW);
	printf("[C_verifier.cpp]Calculate MaxPooling2D10\n\n");
	SW_max_pooling2d_1(O9_SW,O10_SW);
	printf("[C_verifier.cpp]Calculate Conv2D11\n\n");
	SW_conv2d_4(O10_SW,O11_SW,W11);
	printf("[C_verifier.cpp]Calculate BatchNormalization12\n\n");
	SW_batch_normalization_4(O11_SW,O12_SW, W12);
	printf("[C_verifier.cpp]Calculate Activation(Relu)13\n\n");
	SW_activation_4(O12_SW,O13_SW);
	printf("[C_verifier.cpp]Calculate Conv2D14\n\n");
	SW_conv2d_5(O13_SW,O14_SW,W14);
	printf("[C_verifier.cpp]Calculate BatchNormalization15\n\n");
	SW_batch_normalization_5(O14_SW,O15_SW, W15);
	printf("[C_verifier.cpp]Calculate Activation(Relu)16\n\n");
	SW_activation_5(O15_SW,O16_SW);
	printf("[C_verifier.cpp]Calculate MaxPooling2D17\n\n");
	SW_max_pooling2d_2(O16_SW,O17_SW);
	printf("[C_verifier.cpp]Calculate Conv2D18\n\n");
	SW_conv2d_9(O17_SW,O18_SW,W18);
	printf("[C_verifier.cpp]Calculate BatchNormalization19\n\n");
	SW_batch_normalization_9(O18_SW,O19_SW, W19);
	printf("[C_verifier.cpp]Calculate Activation(Relu)20\n\n");
	SW_activation_9(O19_SW,O20_SW);
	printf("[C_verifier.cpp]Calculate Conv2D21\n\n");
	SW_conv2d_7(O17_SW,O21_SW,W21);
	printf("[C_verifier.cpp]Calculate Conv2D22\n\n");
	SW_conv2d_10(O20_SW,O22_SW,W22);
	printf("[C_verifier.cpp]Calculate BatchNormalization23\n\n");
	SW_batch_normalization_7(O21_SW,O23_SW, W23);
	printf("[C_verifier.cpp]Calculate BatchNormalization24\n\n");
	SW_batch_normalization_10(O22_SW,O24_SW, W24);
	printf("[C_verifier.cpp]Calculate Activation(Relu)25\n\n");
	SW_activation_7(O23_SW,O25_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)26\n\n");
	SW_activation_10(O24_SW,O26_SW);
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
	SW_activation_6(O32_SW,O36_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)37\n\n");
	SW_activation_8(O33_SW,O37_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)38\n\n");
	SW_activation_11(O34_SW,O38_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)39\n\n");
	SW_activation_12(O35_SW,O39_SW);
	printf("[C_verifier.cpp]Calculate Concatenate40\n\n");
	SW_mixed_5b(O36_SW, O37_SW, O38_SW, O39_SW,O40_SW);
	printf("[C_verifier.cpp]Calculate Conv2D41\n\n");
	SW_conv2d_16(O40_SW,O41_SW,W41);
	printf("[C_verifier.cpp]Calculate BatchNormalization42\n\n");
	SW_batch_normalization_16(O41_SW,O42_SW, W42);
	printf("[C_verifier.cpp]Calculate Activation(Relu)43\n\n");
	SW_activation_16(O42_SW,O43_SW);
	printf("[C_verifier.cpp]Calculate Conv2D44\n\n");
	SW_conv2d_14(O40_SW,O44_SW,W44);
	printf("[C_verifier.cpp]Calculate Conv2D45\n\n");
	SW_conv2d_17(O43_SW,O45_SW,W45);
	printf("[C_verifier.cpp]Calculate BatchNormalization46\n\n");
	SW_batch_normalization_14(O44_SW,O46_SW, W46);
	printf("[C_verifier.cpp]Calculate BatchNormalization47\n\n");
	SW_batch_normalization_17(O45_SW,O47_SW, W47);
	printf("[C_verifier.cpp]Calculate Activation(Relu)48\n\n");
	SW_activation_14(O46_SW,O48_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)49\n\n");
	SW_activation_17(O47_SW,O49_SW);
	printf("[C_verifier.cpp]Calculate Conv2D50\n\n");
	SW_conv2d_13(O40_SW,O50_SW,W50);
	printf("[C_verifier.cpp]Calculate Conv2D51\n\n");
	SW_conv2d_15(O48_SW,O51_SW,W51);
	printf("[C_verifier.cpp]Calculate Conv2D52\n\n");
	SW_conv2d_18(O49_SW,O52_SW,W52);
	printf("[C_verifier.cpp]Calculate BatchNormalization53\n\n");
	SW_batch_normalization_13(O50_SW,O53_SW, W53);
	printf("[C_verifier.cpp]Calculate BatchNormalization54\n\n");
	SW_batch_normalization_15(O51_SW,O54_SW, W54);
	printf("[C_verifier.cpp]Calculate BatchNormalization55\n\n");
	SW_batch_normalization_18(O52_SW,O55_SW, W55);
	printf("[C_verifier.cpp]Calculate Activation(Relu)56\n\n");
	SW_activation_13(O53_SW,O56_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)57\n\n");
	SW_activation_15(O54_SW,O57_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)58\n\n");
	SW_activation_18(O55_SW,O58_SW);
	printf("[C_verifier.cpp]Calculate Concatenate59\n\n");
	SW_block35_1_mixed(O56_SW, O57_SW, O58_SW,O59_SW);
	printf("[C_verifier.cpp]Calculate Conv2D60\n\n");
	SW_block35_1_conv(O59_SW,O60_SW,W60,B60);
	printf("[C_verifier.cpp]Calculate Lambda61\n\n");
	SW_block35_1(O40_SW, O60_SW, O61_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)62\n\n");
	SW_block35_1_ac(O61_SW,O62_SW);
	printf("[C_verifier.cpp]Calculate Conv2D63\n\n");
	SW_conv2d_22(O62_SW,O63_SW,W63);
	printf("[C_verifier.cpp]Calculate BatchNormalization64\n\n");
	SW_batch_normalization_22(O63_SW,O64_SW, W64);
	printf("[C_verifier.cpp]Calculate Activation(Relu)65\n\n");
	SW_activation_22(O64_SW,O65_SW);
	printf("[C_verifier.cpp]Calculate Conv2D66\n\n");
	SW_conv2d_20(O62_SW,O66_SW,W66);
	printf("[C_verifier.cpp]Calculate Conv2D67\n\n");
	SW_conv2d_23(O65_SW,O67_SW,W67);
	printf("[C_verifier.cpp]Calculate BatchNormalization68\n\n");
	SW_batch_normalization_20(O66_SW,O68_SW, W68);
	printf("[C_verifier.cpp]Calculate BatchNormalization69\n\n");
	SW_batch_normalization_23(O67_SW,O69_SW, W69);
	printf("[C_verifier.cpp]Calculate Activation(Relu)70\n\n");
	SW_activation_20(O68_SW,O70_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)71\n\n");
	SW_activation_23(O69_SW,O71_SW);
	printf("[C_verifier.cpp]Calculate Conv2D72\n\n");
	SW_conv2d_19(O62_SW,O72_SW,W72);
	printf("[C_verifier.cpp]Calculate Conv2D73\n\n");
	SW_conv2d_21(O70_SW,O73_SW,W73);
	printf("[C_verifier.cpp]Calculate Conv2D74\n\n");
	SW_conv2d_24(O71_SW,O74_SW,W74);
	printf("[C_verifier.cpp]Calculate BatchNormalization75\n\n");
	SW_batch_normalization_19(O72_SW,O75_SW, W75);
	printf("[C_verifier.cpp]Calculate BatchNormalization76\n\n");
	SW_batch_normalization_21(O73_SW,O76_SW, W76);
	printf("[C_verifier.cpp]Calculate BatchNormalization77\n\n");
	SW_batch_normalization_24(O74_SW,O77_SW, W77);
	printf("[C_verifier.cpp]Calculate Activation(Relu)78\n\n");
	SW_activation_19(O75_SW,O78_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)79\n\n");
	SW_activation_21(O76_SW,O79_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)80\n\n");
	SW_activation_24(O77_SW,O80_SW);
	printf("[C_verifier.cpp]Calculate Concatenate81\n\n");
	SW_block35_2_mixed(O78_SW, O79_SW, O80_SW,O81_SW);
	printf("[C_verifier.cpp]Calculate Conv2D82\n\n");
	SW_block35_2_conv(O81_SW,O82_SW,W82,B82);
	printf("[C_verifier.cpp]Calculate Lambda83\n\n");
	SW_block35_2(O62_SW, O82_SW, O83_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)84\n\n");
	SW_block35_2_ac(O83_SW,O84_SW);
	printf("[C_verifier.cpp]Calculate Conv2D85\n\n");
	SW_conv2d_28(O84_SW,O85_SW,W85);
	printf("[C_verifier.cpp]Calculate BatchNormalization86\n\n");
	SW_batch_normalization_28(O85_SW,O86_SW, W86);
	printf("[C_verifier.cpp]Calculate Activation(Relu)87\n\n");
	SW_activation_28(O86_SW,O87_SW);
	printf("[C_verifier.cpp]Calculate Conv2D88\n\n");
	SW_conv2d_26(O84_SW,O88_SW,W88);
	printf("[C_verifier.cpp]Calculate Conv2D89\n\n");
	SW_conv2d_29(O87_SW,O89_SW,W89);
	printf("[C_verifier.cpp]Calculate BatchNormalization90\n\n");
	SW_batch_normalization_26(O88_SW,O90_SW, W90);
	printf("[C_verifier.cpp]Calculate BatchNormalization91\n\n");
	SW_batch_normalization_29(O89_SW,O91_SW, W91);
	printf("[C_verifier.cpp]Calculate Activation(Relu)92\n\n");
	SW_activation_26(O90_SW,O92_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)93\n\n");
	SW_activation_29(O91_SW,O93_SW);
	printf("[C_verifier.cpp]Calculate Conv2D94\n\n");
	SW_conv2d_25(O84_SW,O94_SW,W94);
	printf("[C_verifier.cpp]Calculate Conv2D95\n\n");
	SW_conv2d_27(O92_SW,O95_SW,W95);
	printf("[C_verifier.cpp]Calculate Conv2D96\n\n");
	SW_conv2d_30(O93_SW,O96_SW,W96);
	printf("[C_verifier.cpp]Calculate BatchNormalization97\n\n");
	SW_batch_normalization_25(O94_SW,O97_SW, W97);
	printf("[C_verifier.cpp]Calculate BatchNormalization98\n\n");
	SW_batch_normalization_27(O95_SW,O98_SW, W98);
	printf("[C_verifier.cpp]Calculate BatchNormalization99\n\n");
	SW_batch_normalization_30(O96_SW,O99_SW, W99);
	printf("[C_verifier.cpp]Calculate Activation(Relu)100\n\n");
	SW_activation_25(O97_SW,O100_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)101\n\n");
	SW_activation_27(O98_SW,O101_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)102\n\n");
	SW_activation_30(O99_SW,O102_SW);
	printf("[C_verifier.cpp]Calculate Concatenate103\n\n");
	SW_block35_3_mixed(O100_SW, O101_SW, O102_SW,O103_SW);
	printf("[C_verifier.cpp]Calculate Conv2D104\n\n");
	SW_block35_3_conv(O103_SW,O104_SW,W104,B104);
	printf("[C_verifier.cpp]Calculate Lambda105\n\n");
	SW_block35_3(O84_SW, O104_SW, O105_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)106\n\n");
	SW_block35_3_ac(O105_SW,O106_SW);
	printf("[C_verifier.cpp]Calculate Conv2D107\n\n");
	SW_conv2d_34(O106_SW,O107_SW,W107);
	printf("[C_verifier.cpp]Calculate BatchNormalization108\n\n");
	SW_batch_normalization_34(O107_SW,O108_SW, W108);
	printf("[C_verifier.cpp]Calculate Activation(Relu)109\n\n");
	SW_activation_34(O108_SW,O109_SW);
	printf("[C_verifier.cpp]Calculate Conv2D110\n\n");
	SW_conv2d_32(O106_SW,O110_SW,W110);
	printf("[C_verifier.cpp]Calculate Conv2D111\n\n");
	SW_conv2d_35(O109_SW,O111_SW,W111);
	printf("[C_verifier.cpp]Calculate BatchNormalization112\n\n");
	SW_batch_normalization_32(O110_SW,O112_SW, W112);
	printf("[C_verifier.cpp]Calculate BatchNormalization113\n\n");
	SW_batch_normalization_35(O111_SW,O113_SW, W113);
	printf("[C_verifier.cpp]Calculate Activation(Relu)114\n\n");
	SW_activation_32(O112_SW,O114_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)115\n\n");
	SW_activation_35(O113_SW,O115_SW);
	printf("[C_verifier.cpp]Calculate Conv2D116\n\n");
	SW_conv2d_31(O106_SW,O116_SW,W116);
	printf("[C_verifier.cpp]Calculate Conv2D117\n\n");
	SW_conv2d_33(O114_SW,O117_SW,W117);
	printf("[C_verifier.cpp]Calculate Conv2D118\n\n");
	SW_conv2d_36(O115_SW,O118_SW,W118);
	printf("[C_verifier.cpp]Calculate BatchNormalization119\n\n");
	SW_batch_normalization_31(O116_SW,O119_SW, W119);
	printf("[C_verifier.cpp]Calculate BatchNormalization120\n\n");
	SW_batch_normalization_33(O117_SW,O120_SW, W120);
	printf("[C_verifier.cpp]Calculate BatchNormalization121\n\n");
	SW_batch_normalization_36(O118_SW,O121_SW, W121);
	printf("[C_verifier.cpp]Calculate Activation(Relu)122\n\n");
	SW_activation_31(O119_SW,O122_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)123\n\n");
	SW_activation_33(O120_SW,O123_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)124\n\n");
	SW_activation_36(O121_SW,O124_SW);
	printf("[C_verifier.cpp]Calculate Concatenate125\n\n");
	SW_block35_4_mixed(O122_SW, O123_SW, O124_SW,O125_SW);
	printf("[C_verifier.cpp]Calculate Conv2D126\n\n");
	SW_block35_4_conv(O125_SW,O126_SW,W126,B126);
	printf("[C_verifier.cpp]Calculate Lambda127\n\n");
	SW_block35_4(O106_SW, O126_SW, O127_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)128\n\n");
	SW_block35_4_ac(O127_SW,O128_SW);
	printf("[C_verifier.cpp]Calculate Conv2D129\n\n");
	SW_conv2d_40(O128_SW,O129_SW,W129);
	printf("[C_verifier.cpp]Calculate BatchNormalization130\n\n");
	SW_batch_normalization_40(O129_SW,O130_SW, W130);
	printf("[C_verifier.cpp]Calculate Activation(Relu)131\n\n");
	SW_activation_40(O130_SW,O131_SW);
	printf("[C_verifier.cpp]Calculate Conv2D132\n\n");
	SW_conv2d_38(O128_SW,O132_SW,W132);
	printf("[C_verifier.cpp]Calculate Conv2D133\n\n");
	SW_conv2d_41(O131_SW,O133_SW,W133);
	printf("[C_verifier.cpp]Calculate BatchNormalization134\n\n");
	SW_batch_normalization_38(O132_SW,O134_SW, W134);
	printf("[C_verifier.cpp]Calculate BatchNormalization135\n\n");
	SW_batch_normalization_41(O133_SW,O135_SW, W135);
	printf("[C_verifier.cpp]Calculate Activation(Relu)136\n\n");
	SW_activation_38(O134_SW,O136_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)137\n\n");
	SW_activation_41(O135_SW,O137_SW);
	printf("[C_verifier.cpp]Calculate Conv2D138\n\n");
	SW_conv2d_37(O128_SW,O138_SW,W138);
	printf("[C_verifier.cpp]Calculate Conv2D139\n\n");
	SW_conv2d_39(O136_SW,O139_SW,W139);
	printf("[C_verifier.cpp]Calculate Conv2D140\n\n");
	SW_conv2d_42(O137_SW,O140_SW,W140);
	printf("[C_verifier.cpp]Calculate BatchNormalization141\n\n");
	SW_batch_normalization_37(O138_SW,O141_SW, W141);
	printf("[C_verifier.cpp]Calculate BatchNormalization142\n\n");
	SW_batch_normalization_39(O139_SW,O142_SW, W142);
	printf("[C_verifier.cpp]Calculate BatchNormalization143\n\n");
	SW_batch_normalization_42(O140_SW,O143_SW, W143);
	printf("[C_verifier.cpp]Calculate Activation(Relu)144\n\n");
	SW_activation_37(O141_SW,O144_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)145\n\n");
	SW_activation_39(O142_SW,O145_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)146\n\n");
	SW_activation_42(O143_SW,O146_SW);
	printf("[C_verifier.cpp]Calculate Concatenate147\n\n");
	SW_block35_5_mixed(O144_SW, O145_SW, O146_SW,O147_SW);
	printf("[C_verifier.cpp]Calculate Conv2D148\n\n");
	SW_block35_5_conv(O147_SW,O148_SW,W148,B148);
	printf("[C_verifier.cpp]Calculate Lambda149\n\n");
	SW_block35_5(O128_SW, O148_SW, O149_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)150\n\n");
	SW_block35_5_ac(O149_SW,O150_SW);
	printf("[C_verifier.cpp]Calculate Conv2D151\n\n");
	SW_conv2d_46(O150_SW,O151_SW,W151);
	printf("[C_verifier.cpp]Calculate BatchNormalization152\n\n");
	SW_batch_normalization_46(O151_SW,O152_SW, W152);
	printf("[C_verifier.cpp]Calculate Activation(Relu)153\n\n");
	SW_activation_46(O152_SW,O153_SW);
	printf("[C_verifier.cpp]Calculate Conv2D154\n\n");
	SW_conv2d_44(O150_SW,O154_SW,W154);
	printf("[C_verifier.cpp]Calculate Conv2D155\n\n");
	SW_conv2d_47(O153_SW,O155_SW,W155);
	printf("[C_verifier.cpp]Calculate BatchNormalization156\n\n");
	SW_batch_normalization_44(O154_SW,O156_SW, W156);
	printf("[C_verifier.cpp]Calculate BatchNormalization157\n\n");
	SW_batch_normalization_47(O155_SW,O157_SW, W157);
	printf("[C_verifier.cpp]Calculate Activation(Relu)158\n\n");
	SW_activation_44(O156_SW,O158_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)159\n\n");
	SW_activation_47(O157_SW,O159_SW);
	printf("[C_verifier.cpp]Calculate Conv2D160\n\n");
	SW_conv2d_43(O150_SW,O160_SW,W160);
	printf("[C_verifier.cpp]Calculate Conv2D161\n\n");
	SW_conv2d_45(O158_SW,O161_SW,W161);
	printf("[C_verifier.cpp]Calculate Conv2D162\n\n");
	SW_conv2d_48(O159_SW,O162_SW,W162);
	printf("[C_verifier.cpp]Calculate BatchNormalization163\n\n");
	SW_batch_normalization_43(O160_SW,O163_SW, W163);
	printf("[C_verifier.cpp]Calculate BatchNormalization164\n\n");
	SW_batch_normalization_45(O161_SW,O164_SW, W164);
	printf("[C_verifier.cpp]Calculate BatchNormalization165\n\n");
	SW_batch_normalization_48(O162_SW,O165_SW, W165);
	printf("[C_verifier.cpp]Calculate Activation(Relu)166\n\n");
	SW_activation_43(O163_SW,O166_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)167\n\n");
	SW_activation_45(O164_SW,O167_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)168\n\n");
	SW_activation_48(O165_SW,O168_SW);
	printf("[C_verifier.cpp]Calculate Concatenate169\n\n");
	SW_block35_6_mixed(O166_SW, O167_SW, O168_SW,O169_SW);
	printf("[C_verifier.cpp]Calculate Conv2D170\n\n");
	SW_block35_6_conv(O169_SW,O170_SW,W170,B170);
	printf("[C_verifier.cpp]Calculate Lambda171\n\n");
	SW_block35_6(O150_SW, O170_SW, O171_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)172\n\n");
	SW_block35_6_ac(O171_SW,O172_SW);
	printf("[C_verifier.cpp]Calculate Conv2D173\n\n");
	SW_conv2d_52(O172_SW,O173_SW,W173);
	printf("[C_verifier.cpp]Calculate BatchNormalization174\n\n");
	SW_batch_normalization_52(O173_SW,O174_SW, W174);
	printf("[C_verifier.cpp]Calculate Activation(Relu)175\n\n");
	SW_activation_52(O174_SW,O175_SW);
	printf("[C_verifier.cpp]Calculate Conv2D176\n\n");
	SW_conv2d_50(O172_SW,O176_SW,W176);
	printf("[C_verifier.cpp]Calculate Conv2D177\n\n");
	SW_conv2d_53(O175_SW,O177_SW,W177);
	printf("[C_verifier.cpp]Calculate BatchNormalization178\n\n");
	SW_batch_normalization_50(O176_SW,O178_SW, W178);
	printf("[C_verifier.cpp]Calculate BatchNormalization179\n\n");
	SW_batch_normalization_53(O177_SW,O179_SW, W179);
	printf("[C_verifier.cpp]Calculate Activation(Relu)180\n\n");
	SW_activation_50(O178_SW,O180_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)181\n\n");
	SW_activation_53(O179_SW,O181_SW);
	printf("[C_verifier.cpp]Calculate Conv2D182\n\n");
	SW_conv2d_49(O172_SW,O182_SW,W182);
	printf("[C_verifier.cpp]Calculate Conv2D183\n\n");
	SW_conv2d_51(O180_SW,O183_SW,W183);
	printf("[C_verifier.cpp]Calculate Conv2D184\n\n");
	SW_conv2d_54(O181_SW,O184_SW,W184);
	printf("[C_verifier.cpp]Calculate BatchNormalization185\n\n");
	SW_batch_normalization_49(O182_SW,O185_SW, W185);
	printf("[C_verifier.cpp]Calculate BatchNormalization186\n\n");
	SW_batch_normalization_51(O183_SW,O186_SW, W186);
	printf("[C_verifier.cpp]Calculate BatchNormalization187\n\n");
	SW_batch_normalization_54(O184_SW,O187_SW, W187);
	printf("[C_verifier.cpp]Calculate Activation(Relu)188\n\n");
	SW_activation_49(O185_SW,O188_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)189\n\n");
	SW_activation_51(O186_SW,O189_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)190\n\n");
	SW_activation_54(O187_SW,O190_SW);
	printf("[C_verifier.cpp]Calculate Concatenate191\n\n");
	SW_block35_7_mixed(O188_SW, O189_SW, O190_SW,O191_SW);
	printf("[C_verifier.cpp]Calculate Conv2D192\n\n");
	SW_block35_7_conv(O191_SW,O192_SW,W192,B192);
	printf("[C_verifier.cpp]Calculate Lambda193\n\n");
	SW_block35_7(O172_SW, O192_SW, O193_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)194\n\n");
	SW_block35_7_ac(O193_SW,O194_SW);
	printf("[C_verifier.cpp]Calculate Conv2D195\n\n");
	SW_conv2d_58(O194_SW,O195_SW,W195);
	printf("[C_verifier.cpp]Calculate BatchNormalization196\n\n");
	SW_batch_normalization_58(O195_SW,O196_SW, W196);
	printf("[C_verifier.cpp]Calculate Activation(Relu)197\n\n");
	SW_activation_58(O196_SW,O197_SW);
	printf("[C_verifier.cpp]Calculate Conv2D198\n\n");
	SW_conv2d_56(O194_SW,O198_SW,W198);
	printf("[C_verifier.cpp]Calculate Conv2D199\n\n");
	SW_conv2d_59(O197_SW,O199_SW,W199);
	printf("[C_verifier.cpp]Calculate BatchNormalization200\n\n");
	SW_batch_normalization_56(O198_SW,O200_SW, W200);
	printf("[C_verifier.cpp]Calculate BatchNormalization201\n\n");
	SW_batch_normalization_59(O199_SW,O201_SW, W201);
	printf("[C_verifier.cpp]Calculate Activation(Relu)202\n\n");
	SW_activation_56(O200_SW,O202_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)203\n\n");
	SW_activation_59(O201_SW,O203_SW);
	printf("[C_verifier.cpp]Calculate Conv2D204\n\n");
	SW_conv2d_55(O194_SW,O204_SW,W204);
	printf("[C_verifier.cpp]Calculate Conv2D205\n\n");
	SW_conv2d_57(O202_SW,O205_SW,W205);
	printf("[C_verifier.cpp]Calculate Conv2D206\n\n");
	SW_conv2d_60(O203_SW,O206_SW,W206);
	printf("[C_verifier.cpp]Calculate BatchNormalization207\n\n");
	SW_batch_normalization_55(O204_SW,O207_SW, W207);
	printf("[C_verifier.cpp]Calculate BatchNormalization208\n\n");
	SW_batch_normalization_57(O205_SW,O208_SW, W208);
	printf("[C_verifier.cpp]Calculate BatchNormalization209\n\n");
	SW_batch_normalization_60(O206_SW,O209_SW, W209);
	printf("[C_verifier.cpp]Calculate Activation(Relu)210\n\n");
	SW_activation_55(O207_SW,O210_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)211\n\n");
	SW_activation_57(O208_SW,O211_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)212\n\n");
	SW_activation_60(O209_SW,O212_SW);
	printf("[C_verifier.cpp]Calculate Concatenate213\n\n");
	SW_block35_8_mixed(O210_SW, O211_SW, O212_SW,O213_SW);
	printf("[C_verifier.cpp]Calculate Conv2D214\n\n");
	SW_block35_8_conv(O213_SW,O214_SW,W214,B214);
	printf("[C_verifier.cpp]Calculate Lambda215\n\n");
	SW_block35_8(O194_SW, O214_SW, O215_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)216\n\n");
	SW_block35_8_ac(O215_SW,O216_SW);
	printf("[C_verifier.cpp]Calculate Conv2D217\n\n");
	SW_conv2d_64(O216_SW,O217_SW,W217);
	printf("[C_verifier.cpp]Calculate BatchNormalization218\n\n");
	SW_batch_normalization_64(O217_SW,O218_SW, W218);
	printf("[C_verifier.cpp]Calculate Activation(Relu)219\n\n");
	SW_activation_64(O218_SW,O219_SW);
	printf("[C_verifier.cpp]Calculate Conv2D220\n\n");
	SW_conv2d_62(O216_SW,O220_SW,W220);
	printf("[C_verifier.cpp]Calculate Conv2D221\n\n");
	SW_conv2d_65(O219_SW,O221_SW,W221);
	printf("[C_verifier.cpp]Calculate BatchNormalization222\n\n");
	SW_batch_normalization_62(O220_SW,O222_SW, W222);
	printf("[C_verifier.cpp]Calculate BatchNormalization223\n\n");
	SW_batch_normalization_65(O221_SW,O223_SW, W223);
	printf("[C_verifier.cpp]Calculate Activation(Relu)224\n\n");
	SW_activation_62(O222_SW,O224_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)225\n\n");
	SW_activation_65(O223_SW,O225_SW);
	printf("[C_verifier.cpp]Calculate Conv2D226\n\n");
	SW_conv2d_61(O216_SW,O226_SW,W226);
	printf("[C_verifier.cpp]Calculate Conv2D227\n\n");
	SW_conv2d_63(O224_SW,O227_SW,W227);
	printf("[C_verifier.cpp]Calculate Conv2D228\n\n");
	SW_conv2d_66(O225_SW,O228_SW,W228);
	printf("[C_verifier.cpp]Calculate BatchNormalization229\n\n");
	SW_batch_normalization_61(O226_SW,O229_SW, W229);
	printf("[C_verifier.cpp]Calculate BatchNormalization230\n\n");
	SW_batch_normalization_63(O227_SW,O230_SW, W230);
	printf("[C_verifier.cpp]Calculate BatchNormalization231\n\n");
	SW_batch_normalization_66(O228_SW,O231_SW, W231);
	printf("[C_verifier.cpp]Calculate Activation(Relu)232\n\n");
	SW_activation_61(O229_SW,O232_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)233\n\n");
	SW_activation_63(O230_SW,O233_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)234\n\n");
	SW_activation_66(O231_SW,O234_SW);
	printf("[C_verifier.cpp]Calculate Concatenate235\n\n");
	SW_block35_9_mixed(O232_SW, O233_SW, O234_SW,O235_SW);
	printf("[C_verifier.cpp]Calculate Conv2D236\n\n");
	SW_block35_9_conv(O235_SW,O236_SW,W236,B236);
	printf("[C_verifier.cpp]Calculate Lambda237\n\n");
	SW_block35_9(O216_SW, O236_SW, O237_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)238\n\n");
	SW_block35_9_ac(O237_SW,O238_SW);
	printf("[C_verifier.cpp]Calculate Conv2D239\n\n");
	SW_conv2d_70(O238_SW,O239_SW,W239);
	printf("[C_verifier.cpp]Calculate BatchNormalization240\n\n");
	SW_batch_normalization_70(O239_SW,O240_SW, W240);
	printf("[C_verifier.cpp]Calculate Activation(Relu)241\n\n");
	SW_activation_70(O240_SW,O241_SW);
	printf("[C_verifier.cpp]Calculate Conv2D242\n\n");
	SW_conv2d_68(O238_SW,O242_SW,W242);
	printf("[C_verifier.cpp]Calculate Conv2D243\n\n");
	SW_conv2d_71(O241_SW,O243_SW,W243);
	printf("[C_verifier.cpp]Calculate BatchNormalization244\n\n");
	SW_batch_normalization_68(O242_SW,O244_SW, W244);
	printf("[C_verifier.cpp]Calculate BatchNormalization245\n\n");
	SW_batch_normalization_71(O243_SW,O245_SW, W245);
	printf("[C_verifier.cpp]Calculate Activation(Relu)246\n\n");
	SW_activation_68(O244_SW,O246_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)247\n\n");
	SW_activation_71(O245_SW,O247_SW);
	printf("[C_verifier.cpp]Calculate Conv2D248\n\n");
	SW_conv2d_67(O238_SW,O248_SW,W248);
	printf("[C_verifier.cpp]Calculate Conv2D249\n\n");
	SW_conv2d_69(O246_SW,O249_SW,W249);
	printf("[C_verifier.cpp]Calculate Conv2D250\n\n");
	SW_conv2d_72(O247_SW,O250_SW,W250);
	printf("[C_verifier.cpp]Calculate BatchNormalization251\n\n");
	SW_batch_normalization_67(O248_SW,O251_SW, W251);
	printf("[C_verifier.cpp]Calculate BatchNormalization252\n\n");
	SW_batch_normalization_69(O249_SW,O252_SW, W252);
	printf("[C_verifier.cpp]Calculate BatchNormalization253\n\n");
	SW_batch_normalization_72(O250_SW,O253_SW, W253);
	printf("[C_verifier.cpp]Calculate Activation(Relu)254\n\n");
	SW_activation_67(O251_SW,O254_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)255\n\n");
	SW_activation_69(O252_SW,O255_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)256\n\n");
	SW_activation_72(O253_SW,O256_SW);
	printf("[C_verifier.cpp]Calculate Concatenate257\n\n");
	SW_block35_10_mixed(O254_SW, O255_SW, O256_SW,O257_SW);
	printf("[C_verifier.cpp]Calculate Conv2D258\n\n");
	SW_block35_10_conv(O257_SW,O258_SW,W258,B258);
	printf("[C_verifier.cpp]Calculate Lambda259\n\n");
	SW_block35_10(O238_SW, O258_SW, O259_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)260\n\n");
	SW_block35_10_ac(O259_SW,O260_SW);
	printf("[C_verifier.cpp]Calculate Conv2D261\n\n");
	SW_conv2d_74(O260_SW,O261_SW,W261);
	printf("[C_verifier.cpp]Calculate BatchNormalization262\n\n");
	SW_batch_normalization_74(O261_SW,O262_SW, W262);
	printf("[C_verifier.cpp]Calculate Activation(Relu)263\n\n");
	SW_activation_74(O262_SW,O263_SW);
	printf("[C_verifier.cpp]Calculate Conv2D264\n\n");
	SW_conv2d_75(O263_SW,O264_SW,W264);
	printf("[C_verifier.cpp]Calculate BatchNormalization265\n\n");
	SW_batch_normalization_75(O264_SW,O265_SW, W265);
	printf("[C_verifier.cpp]Calculate Activation(Relu)266\n\n");
	SW_activation_75(O265_SW,O266_SW);
	printf("[C_verifier.cpp]Calculate Conv2D267\n\n");
	SW_conv2d_73(O260_SW,O267_SW,W267);
	printf("[C_verifier.cpp]Calculate Conv2D268\n\n");
	SW_conv2d_76(O266_SW,O268_SW,W268);
	printf("[C_verifier.cpp]Calculate BatchNormalization269\n\n");
	SW_batch_normalization_73(O267_SW,O269_SW, W269);
	printf("[C_verifier.cpp]Calculate BatchNormalization270\n\n");
	SW_batch_normalization_76(O268_SW,O270_SW, W270);
	printf("[C_verifier.cpp]Calculate Activation(Relu)271\n\n");
	SW_activation_73(O269_SW,O271_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)272\n\n");
	SW_activation_76(O270_SW,O272_SW);
	printf("[C_verifier.cpp]Calculate MaxPooling2D273\n\n");
	SW_max_pooling2d_3(O260_SW,O273_SW);
	printf("[C_verifier.cpp]Calculate Concatenate274\n\n");
	SW_mixed_6a(O271_SW, O272_SW, O273_SW,O274_SW);
	printf("[C_verifier.cpp]Calculate Conv2D275\n\n");
	SW_conv2d_78(O274_SW,O275_SW,W275);
	printf("[C_verifier.cpp]Calculate BatchNormalization276\n\n");
	SW_batch_normalization_78(O275_SW,O276_SW, W276);
	printf("[C_verifier.cpp]Calculate Activation(Relu)277\n\n");
	SW_activation_78(O276_SW,O277_SW);
	printf("[C_verifier.cpp]Calculate Conv2D278\n\n");
	SW_conv2d_79(O277_SW,O278_SW,W278);
	printf("[C_verifier.cpp]Calculate BatchNormalization279\n\n");
	SW_batch_normalization_79(O278_SW,O279_SW, W279);
	printf("[C_verifier.cpp]Calculate Activation(Relu)280\n\n");
	SW_activation_79(O279_SW,O280_SW);
	printf("[C_verifier.cpp]Calculate Conv2D281\n\n");
	SW_conv2d_77(O274_SW,O281_SW,W281);
	printf("[C_verifier.cpp]Calculate Conv2D282\n\n");
	SW_conv2d_80(O280_SW,O282_SW,W282);
	printf("[C_verifier.cpp]Calculate BatchNormalization283\n\n");
	SW_batch_normalization_77(O281_SW,O283_SW, W283);
	printf("[C_verifier.cpp]Calculate BatchNormalization284\n\n");
	SW_batch_normalization_80(O282_SW,O284_SW, W284);
	printf("[C_verifier.cpp]Calculate Activation(Relu)285\n\n");
	SW_activation_77(O283_SW,O285_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)286\n\n");
	SW_activation_80(O284_SW,O286_SW);
	printf("[C_verifier.cpp]Calculate Concatenate287\n\n");
	SW_block17_1_mixed(O285_SW, O286_SW, O287_SW);
	printf("[C_verifier.cpp]Calculate Conv2D288\n\n");
	SW_block17_1_conv(O287_SW,O288_SW,W288,B288);
	printf("[C_verifier.cpp]Calculate Lambda289\n\n");
	SW_block17_1(O274_SW, O288_SW, O289_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)290\n\n");
	SW_block17_1_ac(O289_SW,O290_SW);
	printf("[C_verifier.cpp]Calculate Conv2D291\n\n");
	SW_conv2d_82(O290_SW,O291_SW,W291);
	printf("[C_verifier.cpp]Calculate BatchNormalization292\n\n");
	SW_batch_normalization_82(O291_SW,O292_SW, W292);
	printf("[C_verifier.cpp]Calculate Activation(Relu)293\n\n");
	SW_activation_82(O292_SW,O293_SW);
	printf("[C_verifier.cpp]Calculate Conv2D294\n\n");
	SW_conv2d_83(O293_SW,O294_SW,W294);
	printf("[C_verifier.cpp]Calculate BatchNormalization295\n\n");
	SW_batch_normalization_83(O294_SW,O295_SW, W295);
	printf("[C_verifier.cpp]Calculate Activation(Relu)296\n\n");
	SW_activation_83(O295_SW,O296_SW);
	printf("[C_verifier.cpp]Calculate Conv2D297\n\n");
	SW_conv2d_81(O290_SW,O297_SW,W297);
	printf("[C_verifier.cpp]Calculate Conv2D298\n\n");
	SW_conv2d_84(O296_SW,O298_SW,W298);
	printf("[C_verifier.cpp]Calculate BatchNormalization299\n\n");
	SW_batch_normalization_81(O297_SW,O299_SW, W299);
	printf("[C_verifier.cpp]Calculate BatchNormalization300\n\n");
	SW_batch_normalization_84(O298_SW,O300_SW, W300);
	printf("[C_verifier.cpp]Calculate Activation(Relu)301\n\n");
	SW_activation_81(O299_SW,O301_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)302\n\n");
	SW_activation_84(O300_SW,O302_SW);
	printf("[C_verifier.cpp]Calculate Concatenate303\n\n");
	SW_block17_2_mixed(O301_SW, O302_SW, O303_SW);
	printf("[C_verifier.cpp]Calculate Conv2D304\n\n");
	SW_block17_2_conv(O303_SW,O304_SW,W304,B304);
	printf("[C_verifier.cpp]Calculate Lambda305\n\n");
	SW_block17_2(O290_SW, O304_SW, O305_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)306\n\n");
	SW_block17_2_ac(O305_SW,O306_SW);
	printf("[C_verifier.cpp]Calculate Conv2D307\n\n");
	SW_conv2d_86(O306_SW,O307_SW,W307);
	printf("[C_verifier.cpp]Calculate BatchNormalization308\n\n");
	SW_batch_normalization_86(O307_SW,O308_SW, W308);
	printf("[C_verifier.cpp]Calculate Activation(Relu)309\n\n");
	SW_activation_86(O308_SW,O309_SW);
	printf("[C_verifier.cpp]Calculate Conv2D310\n\n");
	SW_conv2d_87(O309_SW,O310_SW,W310);
	printf("[C_verifier.cpp]Calculate BatchNormalization311\n\n");
	SW_batch_normalization_87(O310_SW,O311_SW, W311);
	printf("[C_verifier.cpp]Calculate Activation(Relu)312\n\n");
	SW_activation_87(O311_SW,O312_SW);
	printf("[C_verifier.cpp]Calculate Conv2D313\n\n");
	SW_conv2d_85(O306_SW,O313_SW,W313);
	printf("[C_verifier.cpp]Calculate Conv2D314\n\n");
	SW_conv2d_88(O312_SW,O314_SW,W314);
	printf("[C_verifier.cpp]Calculate BatchNormalization315\n\n");
	SW_batch_normalization_85(O313_SW,O315_SW, W315);
	printf("[C_verifier.cpp]Calculate BatchNormalization316\n\n");
	SW_batch_normalization_88(O314_SW,O316_SW, W316);
	printf("[C_verifier.cpp]Calculate Activation(Relu)317\n\n");
	SW_activation_85(O315_SW,O317_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)318\n\n");
	SW_activation_88(O316_SW,O318_SW);
	printf("[C_verifier.cpp]Calculate Concatenate319\n\n");
	SW_block17_3_mixed(O317_SW, O318_SW, O319_SW);
	printf("[C_verifier.cpp]Calculate Conv2D320\n\n");
	SW_block17_3_conv(O319_SW,O320_SW,W320,B320);
	printf("[C_verifier.cpp]Calculate Lambda321\n\n");
	SW_block17_3(O306_SW, O320_SW, O321_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)322\n\n");
	SW_block17_3_ac(O321_SW,O322_SW);
	printf("[C_verifier.cpp]Calculate Conv2D323\n\n");
	SW_conv2d_90(O322_SW,O323_SW,W323);
	printf("[C_verifier.cpp]Calculate BatchNormalization324\n\n");
	SW_batch_normalization_90(O323_SW,O324_SW, W324);
	printf("[C_verifier.cpp]Calculate Activation(Relu)325\n\n");
	SW_activation_90(O324_SW,O325_SW);
	printf("[C_verifier.cpp]Calculate Conv2D326\n\n");
	SW_conv2d_91(O325_SW,O326_SW,W326);
	printf("[C_verifier.cpp]Calculate BatchNormalization327\n\n");
	SW_batch_normalization_91(O326_SW,O327_SW, W327);
	printf("[C_verifier.cpp]Calculate Activation(Relu)328\n\n");
	SW_activation_91(O327_SW,O328_SW);
	printf("[C_verifier.cpp]Calculate Conv2D329\n\n");
	SW_conv2d_89(O322_SW,O329_SW,W329);
	printf("[C_verifier.cpp]Calculate Conv2D330\n\n");
	SW_conv2d_92(O328_SW,O330_SW,W330);
	printf("[C_verifier.cpp]Calculate BatchNormalization331\n\n");
	SW_batch_normalization_89(O329_SW,O331_SW, W331);
	printf("[C_verifier.cpp]Calculate BatchNormalization332\n\n");
	SW_batch_normalization_92(O330_SW,O332_SW, W332);
	printf("[C_verifier.cpp]Calculate Activation(Relu)333\n\n");
	SW_activation_89(O331_SW,O333_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)334\n\n");
	SW_activation_92(O332_SW,O334_SW);
	printf("[C_verifier.cpp]Calculate Concatenate335\n\n");
	SW_block17_4_mixed(O333_SW, O334_SW, O335_SW);
	printf("[C_verifier.cpp]Calculate Conv2D336\n\n");
	SW_block17_4_conv(O335_SW,O336_SW,W336,B336);
	printf("[C_verifier.cpp]Calculate Lambda337\n\n");
	SW_block17_4(O322_SW, O336_SW, O337_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)338\n\n");
	SW_block17_4_ac(O337_SW,O338_SW);
	printf("[C_verifier.cpp]Calculate Conv2D339\n\n");
	SW_conv2d_94(O338_SW,O339_SW,W339);
	printf("[C_verifier.cpp]Calculate BatchNormalization340\n\n");
	SW_batch_normalization_94(O339_SW,O340_SW, W340);
	printf("[C_verifier.cpp]Calculate Activation(Relu)341\n\n");
	SW_activation_94(O340_SW,O341_SW);
	printf("[C_verifier.cpp]Calculate Conv2D342\n\n");
	SW_conv2d_95(O341_SW,O342_SW,W342);
	printf("[C_verifier.cpp]Calculate BatchNormalization343\n\n");
	SW_batch_normalization_95(O342_SW,O343_SW, W343);
	printf("[C_verifier.cpp]Calculate Activation(Relu)344\n\n");
	SW_activation_95(O343_SW,O344_SW);
	printf("[C_verifier.cpp]Calculate Conv2D345\n\n");
	SW_conv2d_93(O338_SW,O345_SW,W345);
	printf("[C_verifier.cpp]Calculate Conv2D346\n\n");
	SW_conv2d_96(O344_SW,O346_SW,W346);
	printf("[C_verifier.cpp]Calculate BatchNormalization347\n\n");
	SW_batch_normalization_93(O345_SW,O347_SW, W347);
	printf("[C_verifier.cpp]Calculate BatchNormalization348\n\n");
	SW_batch_normalization_96(O346_SW,O348_SW, W348);
	printf("[C_verifier.cpp]Calculate Activation(Relu)349\n\n");
	SW_activation_93(O347_SW,O349_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)350\n\n");
	SW_activation_96(O348_SW,O350_SW);
	printf("[C_verifier.cpp]Calculate Concatenate351\n\n");
	SW_block17_5_mixed(O349_SW, O350_SW, O351_SW);
	printf("[C_verifier.cpp]Calculate Conv2D352\n\n");
	SW_block17_5_conv(O351_SW,O352_SW,W352,B352);
	printf("[C_verifier.cpp]Calculate Lambda353\n\n");
	SW_block17_5(O338_SW, O352_SW, O353_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)354\n\n");
	SW_block17_5_ac(O353_SW,O354_SW);
	printf("[C_verifier.cpp]Calculate Conv2D355\n\n");
	SW_conv2d_98(O354_SW,O355_SW,W355);
	printf("[C_verifier.cpp]Calculate BatchNormalization356\n\n");
	SW_batch_normalization_98(O355_SW,O356_SW, W356);
	printf("[C_verifier.cpp]Calculate Activation(Relu)357\n\n");
	SW_activation_98(O356_SW,O357_SW);
	printf("[C_verifier.cpp]Calculate Conv2D358\n\n");
	SW_conv2d_99(O357_SW,O358_SW,W358);
	printf("[C_verifier.cpp]Calculate BatchNormalization359\n\n");
	SW_batch_normalization_99(O358_SW,O359_SW, W359);
	printf("[C_verifier.cpp]Calculate Activation(Relu)360\n\n");
	SW_activation_99(O359_SW,O360_SW);
	printf("[C_verifier.cpp]Calculate Conv2D361\n\n");
	SW_conv2d_97(O354_SW,O361_SW,W361);
	printf("[C_verifier.cpp]Calculate Conv2D362\n\n");
	SW_conv2d_100(O360_SW,O362_SW,W362);
	printf("[C_verifier.cpp]Calculate BatchNormalization363\n\n");
	SW_batch_normalization_97(O361_SW,O363_SW, W363);
	printf("[C_verifier.cpp]Calculate BatchNormalization364\n\n");
	SW_batch_normalization_100(O362_SW,O364_SW, W364);
	printf("[C_verifier.cpp]Calculate Activation(Relu)365\n\n");
	SW_activation_97(O363_SW,O365_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)366\n\n");
	SW_activation_100(O364_SW,O366_SW);
	printf("[C_verifier.cpp]Calculate Concatenate367\n\n");
	SW_block17_6_mixed(O365_SW, O366_SW, O367_SW);
	printf("[C_verifier.cpp]Calculate Conv2D368\n\n");
	SW_block17_6_conv(O367_SW,O368_SW,W368,B368);
	printf("[C_verifier.cpp]Calculate Lambda369\n\n");
	SW_block17_6(O354_SW, O368_SW, O369_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)370\n\n");
	SW_block17_6_ac(O369_SW,O370_SW);
	printf("[C_verifier.cpp]Calculate Conv2D371\n\n");
	SW_conv2d_102(O370_SW,O371_SW,W371);
	printf("[C_verifier.cpp]Calculate BatchNormalization372\n\n");
	SW_batch_normalization_102(O371_SW,O372_SW, W372);
	printf("[C_verifier.cpp]Calculate Activation(Relu)373\n\n");
	SW_activation_102(O372_SW,O373_SW);
	printf("[C_verifier.cpp]Calculate Conv2D374\n\n");
	SW_conv2d_103(O373_SW,O374_SW,W374);
	printf("[C_verifier.cpp]Calculate BatchNormalization375\n\n");
	SW_batch_normalization_103(O374_SW,O375_SW, W375);
	printf("[C_verifier.cpp]Calculate Activation(Relu)376\n\n");
	SW_activation_103(O375_SW,O376_SW);
	printf("[C_verifier.cpp]Calculate Conv2D377\n\n");
	SW_conv2d_101(O370_SW,O377_SW,W377);
	printf("[C_verifier.cpp]Calculate Conv2D378\n\n");
	SW_conv2d_104(O376_SW,O378_SW,W378);
	printf("[C_verifier.cpp]Calculate BatchNormalization379\n\n");
	SW_batch_normalization_101(O377_SW,O379_SW, W379);
	printf("[C_verifier.cpp]Calculate BatchNormalization380\n\n");
	SW_batch_normalization_104(O378_SW,O380_SW, W380);
	printf("[C_verifier.cpp]Calculate Activation(Relu)381\n\n");
	SW_activation_101(O379_SW,O381_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)382\n\n");
	SW_activation_104(O380_SW,O382_SW);
	printf("[C_verifier.cpp]Calculate Concatenate383\n\n");
	SW_block17_7_mixed(O381_SW, O382_SW, O383_SW);
	printf("[C_verifier.cpp]Calculate Conv2D384\n\n");
	SW_block17_7_conv(O383_SW,O384_SW,W384,B384);
	printf("[C_verifier.cpp]Calculate Lambda385\n\n");
	SW_block17_7(O370_SW, O384_SW, O385_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)386\n\n");
	SW_block17_7_ac(O385_SW,O386_SW);
	printf("[C_verifier.cpp]Calculate Conv2D387\n\n");
	SW_conv2d_106(O386_SW,O387_SW,W387);
	printf("[C_verifier.cpp]Calculate BatchNormalization388\n\n");
	SW_batch_normalization_106(O387_SW,O388_SW, W388);
	printf("[C_verifier.cpp]Calculate Activation(Relu)389\n\n");
	SW_activation_106(O388_SW,O389_SW);
	printf("[C_verifier.cpp]Calculate Conv2D390\n\n");
	SW_conv2d_107(O389_SW,O390_SW,W390);
	printf("[C_verifier.cpp]Calculate BatchNormalization391\n\n");
	SW_batch_normalization_107(O390_SW,O391_SW, W391);
	printf("[C_verifier.cpp]Calculate Activation(Relu)392\n\n");
	SW_activation_107(O391_SW,O392_SW);
	printf("[C_verifier.cpp]Calculate Conv2D393\n\n");
	SW_conv2d_105(O386_SW,O393_SW,W393);
	printf("[C_verifier.cpp]Calculate Conv2D394\n\n");
	SW_conv2d_108(O392_SW,O394_SW,W394);
	printf("[C_verifier.cpp]Calculate BatchNormalization395\n\n");
	SW_batch_normalization_105(O393_SW,O395_SW, W395);
	printf("[C_verifier.cpp]Calculate BatchNormalization396\n\n");
	SW_batch_normalization_108(O394_SW,O396_SW, W396);
	printf("[C_verifier.cpp]Calculate Activation(Relu)397\n\n");
	SW_activation_105(O395_SW,O397_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)398\n\n");
	SW_activation_108(O396_SW,O398_SW);
	printf("[C_verifier.cpp]Calculate Concatenate399\n\n");
	SW_block17_8_mixed(O397_SW, O398_SW, O399_SW);
	printf("[C_verifier.cpp]Calculate Conv2D400\n\n");
	SW_block17_8_conv(O399_SW,O400_SW,W400,B400);
	printf("[C_verifier.cpp]Calculate Lambda401\n\n");
	SW_block17_8(O386_SW, O400_SW, O401_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)402\n\n");
	SW_block17_8_ac(O401_SW,O402_SW);
	printf("[C_verifier.cpp]Calculate Conv2D403\n\n");
	SW_conv2d_110(O402_SW,O403_SW,W403);
	printf("[C_verifier.cpp]Calculate BatchNormalization404\n\n");
	SW_batch_normalization_110(O403_SW,O404_SW, W404);
	printf("[C_verifier.cpp]Calculate Activation(Relu)405\n\n");
	SW_activation_110(O404_SW,O405_SW);
	printf("[C_verifier.cpp]Calculate Conv2D406\n\n");
	SW_conv2d_111(O405_SW,O406_SW,W406);
	printf("[C_verifier.cpp]Calculate BatchNormalization407\n\n");
	SW_batch_normalization_111(O406_SW,O407_SW, W407);
	printf("[C_verifier.cpp]Calculate Activation(Relu)408\n\n");
	SW_activation_111(O407_SW,O408_SW);
	printf("[C_verifier.cpp]Calculate Conv2D409\n\n");
	SW_conv2d_109(O402_SW,O409_SW,W409);
	printf("[C_verifier.cpp]Calculate Conv2D410\n\n");
	SW_conv2d_112(O408_SW,O410_SW,W410);
	printf("[C_verifier.cpp]Calculate BatchNormalization411\n\n");
	SW_batch_normalization_109(O409_SW,O411_SW, W411);
	printf("[C_verifier.cpp]Calculate BatchNormalization412\n\n");
	SW_batch_normalization_112(O410_SW,O412_SW, W412);
	printf("[C_verifier.cpp]Calculate Activation(Relu)413\n\n");
	SW_activation_109(O411_SW,O413_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)414\n\n");
	SW_activation_112(O412_SW,O414_SW);
	printf("[C_verifier.cpp]Calculate Concatenate415\n\n");
	SW_block17_9_mixed(O413_SW, O414_SW, O415_SW);
	printf("[C_verifier.cpp]Calculate Conv2D416\n\n");
	SW_block17_9_conv(O415_SW,O416_SW,W416,B416);
	printf("[C_verifier.cpp]Calculate Lambda417\n\n");
	SW_block17_9(O402_SW, O416_SW, O417_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)418\n\n");
	SW_block17_9_ac(O417_SW,O418_SW);
	printf("[C_verifier.cpp]Calculate Conv2D419\n\n");
	SW_conv2d_114(O418_SW,O419_SW,W419);
	printf("[C_verifier.cpp]Calculate BatchNormalization420\n\n");
	SW_batch_normalization_114(O419_SW,O420_SW, W420);
	printf("[C_verifier.cpp]Calculate Activation(Relu)421\n\n");
	SW_activation_114(O420_SW,O421_SW);
	printf("[C_verifier.cpp]Calculate Conv2D422\n\n");
	SW_conv2d_115(O421_SW,O422_SW,W422);
	printf("[C_verifier.cpp]Calculate BatchNormalization423\n\n");
	SW_batch_normalization_115(O422_SW,O423_SW, W423);
	printf("[C_verifier.cpp]Calculate Activation(Relu)424\n\n");
	SW_activation_115(O423_SW,O424_SW);
	printf("[C_verifier.cpp]Calculate Conv2D425\n\n");
	SW_conv2d_113(O418_SW,O425_SW,W425);
	printf("[C_verifier.cpp]Calculate Conv2D426\n\n");
	SW_conv2d_116(O424_SW,O426_SW,W426);
	printf("[C_verifier.cpp]Calculate BatchNormalization427\n\n");
	SW_batch_normalization_113(O425_SW,O427_SW, W427);
	printf("[C_verifier.cpp]Calculate BatchNormalization428\n\n");
	SW_batch_normalization_116(O426_SW,O428_SW, W428);
	printf("[C_verifier.cpp]Calculate Activation(Relu)429\n\n");
	SW_activation_113(O427_SW,O429_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)430\n\n");
	SW_activation_116(O428_SW,O430_SW);
	printf("[C_verifier.cpp]Calculate Concatenate431\n\n");
	SW_block17_10_mixed(O429_SW, O430_SW, O431_SW);
	printf("[C_verifier.cpp]Calculate Conv2D432\n\n");
	SW_block17_10_conv(O431_SW,O432_SW,W432,B432);
	printf("[C_verifier.cpp]Calculate Lambda433\n\n");
	SW_block17_10(O418_SW, O432_SW, O433_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)434\n\n");
	SW_block17_10_ac(O433_SW,O434_SW);
	printf("[C_verifier.cpp]Calculate Conv2D435\n\n");
	SW_conv2d_118(O434_SW,O435_SW,W435);
	printf("[C_verifier.cpp]Calculate BatchNormalization436\n\n");
	SW_batch_normalization_118(O435_SW,O436_SW, W436);
	printf("[C_verifier.cpp]Calculate Activation(Relu)437\n\n");
	SW_activation_118(O436_SW,O437_SW);
	printf("[C_verifier.cpp]Calculate Conv2D438\n\n");
	SW_conv2d_119(O437_SW,O438_SW,W438);
	printf("[C_verifier.cpp]Calculate BatchNormalization439\n\n");
	SW_batch_normalization_119(O438_SW,O439_SW, W439);
	printf("[C_verifier.cpp]Calculate Activation(Relu)440\n\n");
	SW_activation_119(O439_SW,O440_SW);
	printf("[C_verifier.cpp]Calculate Conv2D441\n\n");
	SW_conv2d_117(O434_SW,O441_SW,W441);
	printf("[C_verifier.cpp]Calculate Conv2D442\n\n");
	SW_conv2d_120(O440_SW,O442_SW,W442);
	printf("[C_verifier.cpp]Calculate BatchNormalization443\n\n");
	SW_batch_normalization_117(O441_SW,O443_SW, W443);
	printf("[C_verifier.cpp]Calculate BatchNormalization444\n\n");
	SW_batch_normalization_120(O442_SW,O444_SW, W444);
	printf("[C_verifier.cpp]Calculate Activation(Relu)445\n\n");
	SW_activation_117(O443_SW,O445_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)446\n\n");
	SW_activation_120(O444_SW,O446_SW);
	printf("[C_verifier.cpp]Calculate Concatenate447\n\n");
	SW_block17_11_mixed(O445_SW, O446_SW, O447_SW);
	printf("[C_verifier.cpp]Calculate Conv2D448\n\n");
	SW_block17_11_conv(O447_SW,O448_SW,W448,B448);
	printf("[C_verifier.cpp]Calculate Lambda449\n\n");
	SW_block17_11(O434_SW, O448_SW, O449_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)450\n\n");
	SW_block17_11_ac(O449_SW,O450_SW);
	printf("[C_verifier.cpp]Calculate Conv2D451\n\n");
	SW_conv2d_122(O450_SW,O451_SW,W451);
	printf("[C_verifier.cpp]Calculate BatchNormalization452\n\n");
	SW_batch_normalization_122(O451_SW,O452_SW, W452);
	printf("[C_verifier.cpp]Calculate Activation(Relu)453\n\n");
	SW_activation_122(O452_SW,O453_SW);
	printf("[C_verifier.cpp]Calculate Conv2D454\n\n");
	SW_conv2d_123(O453_SW,O454_SW,W454);
	printf("[C_verifier.cpp]Calculate BatchNormalization455\n\n");
	SW_batch_normalization_123(O454_SW,O455_SW, W455);
	printf("[C_verifier.cpp]Calculate Activation(Relu)456\n\n");
	SW_activation_123(O455_SW,O456_SW);
	printf("[C_verifier.cpp]Calculate Conv2D457\n\n");
	SW_conv2d_121(O450_SW,O457_SW,W457);
	printf("[C_verifier.cpp]Calculate Conv2D458\n\n");
	SW_conv2d_124(O456_SW,O458_SW,W458);
	printf("[C_verifier.cpp]Calculate BatchNormalization459\n\n");
	SW_batch_normalization_121(O457_SW,O459_SW, W459);
	printf("[C_verifier.cpp]Calculate BatchNormalization460\n\n");
	SW_batch_normalization_124(O458_SW,O460_SW, W460);
	printf("[C_verifier.cpp]Calculate Activation(Relu)461\n\n");
	SW_activation_121(O459_SW,O461_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)462\n\n");
	SW_activation_124(O460_SW,O462_SW);
	printf("[C_verifier.cpp]Calculate Concatenate463\n\n");
	SW_block17_12_mixed(O461_SW, O462_SW, O463_SW);
	printf("[C_verifier.cpp]Calculate Conv2D464\n\n");
	SW_block17_12_conv(O463_SW,O464_SW,W464,B464);
	printf("[C_verifier.cpp]Calculate Lambda465\n\n");
	SW_block17_12(O450_SW, O464_SW, O465_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)466\n\n");
	SW_block17_12_ac(O465_SW,O466_SW);
	printf("[C_verifier.cpp]Calculate Conv2D467\n\n");
	SW_conv2d_126(O466_SW,O467_SW,W467);
	printf("[C_verifier.cpp]Calculate BatchNormalization468\n\n");
	SW_batch_normalization_126(O467_SW,O468_SW, W468);
	printf("[C_verifier.cpp]Calculate Activation(Relu)469\n\n");
	SW_activation_126(O468_SW,O469_SW);
	printf("[C_verifier.cpp]Calculate Conv2D470\n\n");
	SW_conv2d_127(O469_SW,O470_SW,W470);
	printf("[C_verifier.cpp]Calculate BatchNormalization471\n\n");
	SW_batch_normalization_127(O470_SW,O471_SW, W471);
	printf("[C_verifier.cpp]Calculate Activation(Relu)472\n\n");
	SW_activation_127(O471_SW,O472_SW);
	printf("[C_verifier.cpp]Calculate Conv2D473\n\n");
	SW_conv2d_125(O466_SW,O473_SW,W473);
	printf("[C_verifier.cpp]Calculate Conv2D474\n\n");
	SW_conv2d_128(O472_SW,O474_SW,W474);
	printf("[C_verifier.cpp]Calculate BatchNormalization475\n\n");
	SW_batch_normalization_125(O473_SW,O475_SW, W475);
	printf("[C_verifier.cpp]Calculate BatchNormalization476\n\n");
	SW_batch_normalization_128(O474_SW,O476_SW, W476);
	printf("[C_verifier.cpp]Calculate Activation(Relu)477\n\n");
	SW_activation_125(O475_SW,O477_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)478\n\n");
	SW_activation_128(O476_SW,O478_SW);
	printf("[C_verifier.cpp]Calculate Concatenate479\n\n");
	SW_block17_13_mixed(O477_SW, O478_SW, O479_SW);
	printf("[C_verifier.cpp]Calculate Conv2D480\n\n");
	SW_block17_13_conv(O479_SW,O480_SW,W480,B480);
	printf("[C_verifier.cpp]Calculate Lambda481\n\n");
	SW_block17_13(O466_SW, O480_SW, O481_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)482\n\n");
	SW_block17_13_ac(O481_SW,O482_SW);
	printf("[C_verifier.cpp]Calculate Conv2D483\n\n");
	SW_conv2d_130(O482_SW,O483_SW,W483);
	printf("[C_verifier.cpp]Calculate BatchNormalization484\n\n");
	SW_batch_normalization_130(O483_SW,O484_SW, W484);
	printf("[C_verifier.cpp]Calculate Activation(Relu)485\n\n");
	SW_activation_130(O484_SW,O485_SW);
	printf("[C_verifier.cpp]Calculate Conv2D486\n\n");
	SW_conv2d_131(O485_SW,O486_SW,W486);
	printf("[C_verifier.cpp]Calculate BatchNormalization487\n\n");
	SW_batch_normalization_131(O486_SW,O487_SW, W487);
	printf("[C_verifier.cpp]Calculate Activation(Relu)488\n\n");
	SW_activation_131(O487_SW,O488_SW);
	printf("[C_verifier.cpp]Calculate Conv2D489\n\n");
	SW_conv2d_129(O482_SW,O489_SW,W489);
	printf("[C_verifier.cpp]Calculate Conv2D490\n\n");
	SW_conv2d_132(O488_SW,O490_SW,W490);
	printf("[C_verifier.cpp]Calculate BatchNormalization491\n\n");
	SW_batch_normalization_129(O489_SW,O491_SW, W491);
	printf("[C_verifier.cpp]Calculate BatchNormalization492\n\n");
	SW_batch_normalization_132(O490_SW,O492_SW, W492);
	printf("[C_verifier.cpp]Calculate Activation(Relu)493\n\n");
	SW_activation_129(O491_SW,O493_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)494\n\n");
	SW_activation_132(O492_SW,O494_SW);
	printf("[C_verifier.cpp]Calculate Concatenate495\n\n");
	SW_block17_14_mixed(O493_SW, O494_SW, O495_SW);
	printf("[C_verifier.cpp]Calculate Conv2D496\n\n");
	SW_block17_14_conv(O495_SW,O496_SW,W496,B496);
	printf("[C_verifier.cpp]Calculate Lambda497\n\n");
	SW_block17_14(O482_SW, O496_SW, O497_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)498\n\n");
	SW_block17_14_ac(O497_SW,O498_SW);
	printf("[C_verifier.cpp]Calculate Conv2D499\n\n");
	SW_conv2d_134(O498_SW,O499_SW,W499);
	printf("[C_verifier.cpp]Calculate BatchNormalization500\n\n");
	SW_batch_normalization_134(O499_SW,O500_SW, W500);
	printf("[C_verifier.cpp]Calculate Activation(Relu)501\n\n");
	SW_activation_134(O500_SW,O501_SW);
	printf("[C_verifier.cpp]Calculate Conv2D502\n\n");
	SW_conv2d_135(O501_SW,O502_SW,W502);
	printf("[C_verifier.cpp]Calculate BatchNormalization503\n\n");
	SW_batch_normalization_135(O502_SW,O503_SW, W503);
	printf("[C_verifier.cpp]Calculate Activation(Relu)504\n\n");
	SW_activation_135(O503_SW,O504_SW);
	printf("[C_verifier.cpp]Calculate Conv2D505\n\n");
	SW_conv2d_133(O498_SW,O505_SW,W505);
	printf("[C_verifier.cpp]Calculate Conv2D506\n\n");
	SW_conv2d_136(O504_SW,O506_SW,W506);
	printf("[C_verifier.cpp]Calculate BatchNormalization507\n\n");
	SW_batch_normalization_133(O505_SW,O507_SW, W507);
	printf("[C_verifier.cpp]Calculate BatchNormalization508\n\n");
	SW_batch_normalization_136(O506_SW,O508_SW, W508);
	printf("[C_verifier.cpp]Calculate Activation(Relu)509\n\n");
	SW_activation_133(O507_SW,O509_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)510\n\n");
	SW_activation_136(O508_SW,O510_SW);
	printf("[C_verifier.cpp]Calculate Concatenate511\n\n");
	SW_block17_15_mixed(O509_SW, O510_SW, O511_SW);
	printf("[C_verifier.cpp]Calculate Conv2D512\n\n");
	SW_block17_15_conv(O511_SW,O512_SW,W512,B512);
	printf("[C_verifier.cpp]Calculate Lambda513\n\n");
	SW_block17_15(O498_SW, O512_SW, O513_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)514\n\n");
	SW_block17_15_ac(O513_SW,O514_SW);
	printf("[C_verifier.cpp]Calculate Conv2D515\n\n");
	SW_conv2d_138(O514_SW,O515_SW,W515);
	printf("[C_verifier.cpp]Calculate BatchNormalization516\n\n");
	SW_batch_normalization_138(O515_SW,O516_SW, W516);
	printf("[C_verifier.cpp]Calculate Activation(Relu)517\n\n");
	SW_activation_138(O516_SW,O517_SW);
	printf("[C_verifier.cpp]Calculate Conv2D518\n\n");
	SW_conv2d_139(O517_SW,O518_SW,W518);
	printf("[C_verifier.cpp]Calculate BatchNormalization519\n\n");
	SW_batch_normalization_139(O518_SW,O519_SW, W519);
	printf("[C_verifier.cpp]Calculate Activation(Relu)520\n\n");
	SW_activation_139(O519_SW,O520_SW);
	printf("[C_verifier.cpp]Calculate Conv2D521\n\n");
	SW_conv2d_137(O514_SW,O521_SW,W521);
	printf("[C_verifier.cpp]Calculate Conv2D522\n\n");
	SW_conv2d_140(O520_SW,O522_SW,W522);
	printf("[C_verifier.cpp]Calculate BatchNormalization523\n\n");
	SW_batch_normalization_137(O521_SW,O523_SW, W523);
	printf("[C_verifier.cpp]Calculate BatchNormalization524\n\n");
	SW_batch_normalization_140(O522_SW,O524_SW, W524);
	printf("[C_verifier.cpp]Calculate Activation(Relu)525\n\n");
	SW_activation_137(O523_SW,O525_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)526\n\n");
	SW_activation_140(O524_SW,O526_SW);
	printf("[C_verifier.cpp]Calculate Concatenate527\n\n");
	SW_block17_16_mixed(O525_SW, O526_SW, O527_SW);
	printf("[C_verifier.cpp]Calculate Conv2D528\n\n");
	SW_block17_16_conv(O527_SW,O528_SW,W528,B528);
	printf("[C_verifier.cpp]Calculate Lambda529\n\n");
	SW_block17_16(O514_SW, O528_SW, O529_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)530\n\n");
	SW_block17_16_ac(O529_SW,O530_SW);
	printf("[C_verifier.cpp]Calculate Conv2D531\n\n");
	SW_conv2d_142(O530_SW,O531_SW,W531);
	printf("[C_verifier.cpp]Calculate BatchNormalization532\n\n");
	SW_batch_normalization_142(O531_SW,O532_SW, W532);
	printf("[C_verifier.cpp]Calculate Activation(Relu)533\n\n");
	SW_activation_142(O532_SW,O533_SW);
	printf("[C_verifier.cpp]Calculate Conv2D534\n\n");
	SW_conv2d_143(O533_SW,O534_SW,W534);
	printf("[C_verifier.cpp]Calculate BatchNormalization535\n\n");
	SW_batch_normalization_143(O534_SW,O535_SW, W535);
	printf("[C_verifier.cpp]Calculate Activation(Relu)536\n\n");
	SW_activation_143(O535_SW,O536_SW);
	printf("[C_verifier.cpp]Calculate Conv2D537\n\n");
	SW_conv2d_141(O530_SW,O537_SW,W537);
	printf("[C_verifier.cpp]Calculate Conv2D538\n\n");
	SW_conv2d_144(O536_SW,O538_SW,W538);
	printf("[C_verifier.cpp]Calculate BatchNormalization539\n\n");
	SW_batch_normalization_141(O537_SW,O539_SW, W539);
	printf("[C_verifier.cpp]Calculate BatchNormalization540\n\n");
	SW_batch_normalization_144(O538_SW,O540_SW, W540);
	printf("[C_verifier.cpp]Calculate Activation(Relu)541\n\n");
	SW_activation_141(O539_SW,O541_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)542\n\n");
	SW_activation_144(O540_SW,O542_SW);
	printf("[C_verifier.cpp]Calculate Concatenate543\n\n");
	SW_block17_17_mixed(O541_SW, O542_SW, O543_SW);
	printf("[C_verifier.cpp]Calculate Conv2D544\n\n");
	SW_block17_17_conv(O543_SW,O544_SW,W544,B544);
	printf("[C_verifier.cpp]Calculate Lambda545\n\n");
	SW_block17_17(O530_SW, O544_SW, O545_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)546\n\n");
	SW_block17_17_ac(O545_SW,O546_SW);
	printf("[C_verifier.cpp]Calculate Conv2D547\n\n");
	SW_conv2d_146(O546_SW,O547_SW,W547);
	printf("[C_verifier.cpp]Calculate BatchNormalization548\n\n");
	SW_batch_normalization_146(O547_SW,O548_SW, W548);
	printf("[C_verifier.cpp]Calculate Activation(Relu)549\n\n");
	SW_activation_146(O548_SW,O549_SW);
	printf("[C_verifier.cpp]Calculate Conv2D550\n\n");
	SW_conv2d_147(O549_SW,O550_SW,W550);
	printf("[C_verifier.cpp]Calculate BatchNormalization551\n\n");
	SW_batch_normalization_147(O550_SW,O551_SW, W551);
	printf("[C_verifier.cpp]Calculate Activation(Relu)552\n\n");
	SW_activation_147(O551_SW,O552_SW);
	printf("[C_verifier.cpp]Calculate Conv2D553\n\n");
	SW_conv2d_145(O546_SW,O553_SW,W553);
	printf("[C_verifier.cpp]Calculate Conv2D554\n\n");
	SW_conv2d_148(O552_SW,O554_SW,W554);
	printf("[C_verifier.cpp]Calculate BatchNormalization555\n\n");
	SW_batch_normalization_145(O553_SW,O555_SW, W555);
	printf("[C_verifier.cpp]Calculate BatchNormalization556\n\n");
	SW_batch_normalization_148(O554_SW,O556_SW, W556);
	printf("[C_verifier.cpp]Calculate Activation(Relu)557\n\n");
	SW_activation_145(O555_SW,O557_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)558\n\n");
	SW_activation_148(O556_SW,O558_SW);
	printf("[C_verifier.cpp]Calculate Concatenate559\n\n");
	SW_block17_18_mixed(O557_SW, O558_SW, O559_SW);
	printf("[C_verifier.cpp]Calculate Conv2D560\n\n");
	SW_block17_18_conv(O559_SW,O560_SW,W560,B560);
	printf("[C_verifier.cpp]Calculate Lambda561\n\n");
	SW_block17_18(O546_SW, O560_SW, O561_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)562\n\n");
	SW_block17_18_ac(O561_SW,O562_SW);
	printf("[C_verifier.cpp]Calculate Conv2D563\n\n");
	SW_conv2d_150(O562_SW,O563_SW,W563);
	printf("[C_verifier.cpp]Calculate BatchNormalization564\n\n");
	SW_batch_normalization_150(O563_SW,O564_SW, W564);
	printf("[C_verifier.cpp]Calculate Activation(Relu)565\n\n");
	SW_activation_150(O564_SW,O565_SW);
	printf("[C_verifier.cpp]Calculate Conv2D566\n\n");
	SW_conv2d_151(O565_SW,O566_SW,W566);
	printf("[C_verifier.cpp]Calculate BatchNormalization567\n\n");
	SW_batch_normalization_151(O566_SW,O567_SW, W567);
	printf("[C_verifier.cpp]Calculate Activation(Relu)568\n\n");
	SW_activation_151(O567_SW,O568_SW);
	printf("[C_verifier.cpp]Calculate Conv2D569\n\n");
	SW_conv2d_149(O562_SW,O569_SW,W569);
	printf("[C_verifier.cpp]Calculate Conv2D570\n\n");
	SW_conv2d_152(O568_SW,O570_SW,W570);
	printf("[C_verifier.cpp]Calculate BatchNormalization571\n\n");
	SW_batch_normalization_149(O569_SW,O571_SW, W571);
	printf("[C_verifier.cpp]Calculate BatchNormalization572\n\n");
	SW_batch_normalization_152(O570_SW,O572_SW, W572);
	printf("[C_verifier.cpp]Calculate Activation(Relu)573\n\n");
	SW_activation_149(O571_SW,O573_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)574\n\n");
	SW_activation_152(O572_SW,O574_SW);
	printf("[C_verifier.cpp]Calculate Concatenate575\n\n");
	SW_block17_19_mixed(O573_SW, O574_SW, O575_SW);
	printf("[C_verifier.cpp]Calculate Conv2D576\n\n");
	SW_block17_19_conv(O575_SW,O576_SW,W576,B576);
	printf("[C_verifier.cpp]Calculate Lambda577\n\n");
	SW_block17_19(O562_SW, O576_SW, O577_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)578\n\n");
	SW_block17_19_ac(O577_SW,O578_SW);
	printf("[C_verifier.cpp]Calculate Conv2D579\n\n");
	SW_conv2d_154(O578_SW,O579_SW,W579);
	printf("[C_verifier.cpp]Calculate BatchNormalization580\n\n");
	SW_batch_normalization_154(O579_SW,O580_SW, W580);
	printf("[C_verifier.cpp]Calculate Activation(Relu)581\n\n");
	SW_activation_154(O580_SW,O581_SW);
	printf("[C_verifier.cpp]Calculate Conv2D582\n\n");
	SW_conv2d_155(O581_SW,O582_SW,W582);
	printf("[C_verifier.cpp]Calculate BatchNormalization583\n\n");
	SW_batch_normalization_155(O582_SW,O583_SW, W583);
	printf("[C_verifier.cpp]Calculate Activation(Relu)584\n\n");
	SW_activation_155(O583_SW,O584_SW);
	printf("[C_verifier.cpp]Calculate Conv2D585\n\n");
	SW_conv2d_153(O578_SW,O585_SW,W585);
	printf("[C_verifier.cpp]Calculate Conv2D586\n\n");
	SW_conv2d_156(O584_SW,O586_SW,W586);
	printf("[C_verifier.cpp]Calculate BatchNormalization587\n\n");
	SW_batch_normalization_153(O585_SW,O587_SW, W587);
	printf("[C_verifier.cpp]Calculate BatchNormalization588\n\n");
	SW_batch_normalization_156(O586_SW,O588_SW, W588);
	printf("[C_verifier.cpp]Calculate Activation(Relu)589\n\n");
	SW_activation_153(O587_SW,O589_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)590\n\n");
	SW_activation_156(O588_SW,O590_SW);
	printf("[C_verifier.cpp]Calculate Concatenate591\n\n");
	SW_block17_20_mixed(O589_SW, O590_SW, O591_SW);
	printf("[C_verifier.cpp]Calculate Conv2D592\n\n");
	SW_block17_20_conv(O591_SW,O592_SW,W592,B592);
	printf("[C_verifier.cpp]Calculate Lambda593\n\n");
	SW_block17_20(O578_SW, O592_SW, O593_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)594\n\n");
	SW_block17_20_ac(O593_SW,O594_SW);
	printf("[C_verifier.cpp]Calculate Conv2D595\n\n");
	SW_conv2d_161(O594_SW,O595_SW,W595);
	printf("[C_verifier.cpp]Calculate BatchNormalization596\n\n");
	SW_batch_normalization_161(O595_SW,O596_SW, W596);
	printf("[C_verifier.cpp]Calculate Activation(Relu)597\n\n");
	SW_activation_161(O596_SW,O597_SW);
	printf("[C_verifier.cpp]Calculate Conv2D598\n\n");
	SW_conv2d_157(O594_SW,O598_SW,W598);
	printf("[C_verifier.cpp]Calculate Conv2D599\n\n");
	SW_conv2d_159(O594_SW,O599_SW,W599);
	printf("[C_verifier.cpp]Calculate Conv2D600\n\n");
	SW_conv2d_162(O597_SW,O600_SW,W600);
	printf("[C_verifier.cpp]Calculate BatchNormalization601\n\n");
	SW_batch_normalization_157(O598_SW,O601_SW, W601);
	printf("[C_verifier.cpp]Calculate BatchNormalization602\n\n");
	SW_batch_normalization_159(O599_SW,O602_SW, W602);
	printf("[C_verifier.cpp]Calculate BatchNormalization603\n\n");
	SW_batch_normalization_162(O600_SW,O603_SW, W603);
	printf("[C_verifier.cpp]Calculate Activation(Relu)604\n\n");
	SW_activation_157(O601_SW,O604_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)605\n\n");
	SW_activation_159(O602_SW,O605_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)606\n\n");
	SW_activation_162(O603_SW,O606_SW);
	printf("[C_verifier.cpp]Calculate Conv2D607\n\n");
	SW_conv2d_158(O604_SW,O607_SW,W607);
	printf("[C_verifier.cpp]Calculate Conv2D608\n\n");
	SW_conv2d_160(O605_SW,O608_SW,W608);
	printf("[C_verifier.cpp]Calculate Conv2D609\n\n");
	SW_conv2d_163(O606_SW,O609_SW,W609);
	printf("[C_verifier.cpp]Calculate BatchNormalization610\n\n");
	SW_batch_normalization_158(O607_SW,O610_SW, W610);
	printf("[C_verifier.cpp]Calculate BatchNormalization611\n\n");
	SW_batch_normalization_160(O608_SW,O611_SW, W611);
	printf("[C_verifier.cpp]Calculate BatchNormalization612\n\n");
	SW_batch_normalization_163(O609_SW,O612_SW, W612);
	printf("[C_verifier.cpp]Calculate Activation(Relu)613\n\n");
	SW_activation_158(O610_SW,O613_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)614\n\n");
	SW_activation_160(O611_SW,O614_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)615\n\n");
	SW_activation_163(O612_SW,O615_SW);
	printf("[C_verifier.cpp]Calculate MaxPooling2D616\n\n");
	SW_max_pooling2d_4(O594_SW,O616_SW);
	printf("[C_verifier.cpp]Calculate Concatenate617\n\n");
	SW_mixed_7a(O613_SW, O614_SW, O615_SW, O616_SW,O617_SW);
	printf("[C_verifier.cpp]Calculate Conv2D618\n\n");
	SW_conv2d_165(O617_SW,O618_SW,W618);
	printf("[C_verifier.cpp]Calculate BatchNormalization619\n\n");
	SW_batch_normalization_165(O618_SW,O619_SW, W619);
	printf("[C_verifier.cpp]Calculate Activation(Relu)620\n\n");
	SW_activation_165(O619_SW,O620_SW);
	printf("[C_verifier.cpp]Calculate Conv2D621\n\n");
	SW_conv2d_166(O620_SW,O621_SW,W621);
	printf("[C_verifier.cpp]Calculate BatchNormalization622\n\n");
	SW_batch_normalization_166(O621_SW,O622_SW, W622);
	printf("[C_verifier.cpp]Calculate Activation(Relu)623\n\n");
	SW_activation_166(O622_SW,O623_SW);
	printf("[C_verifier.cpp]Calculate Conv2D624\n\n");
	SW_conv2d_164(O617_SW,O624_SW,W624);
	printf("[C_verifier.cpp]Calculate Conv2D625\n\n");
	SW_conv2d_167(O623_SW,O625_SW,W625);
	printf("[C_verifier.cpp]Calculate BatchNormalization626\n\n");
	SW_batch_normalization_164(O624_SW,O626_SW, W626);
	printf("[C_verifier.cpp]Calculate BatchNormalization627\n\n");
	SW_batch_normalization_167(O625_SW,O627_SW, W627);
	printf("[C_verifier.cpp]Calculate Activation(Relu)628\n\n");
	SW_activation_164(O626_SW,O628_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)629\n\n");
	SW_activation_167(O627_SW,O629_SW);
	printf("[C_verifier.cpp]Calculate Concatenate630\n\n");
	SW_block8_1_mixed(O628_SW, O629_SW, O630_SW);
	printf("[C_verifier.cpp]Calculate Conv2D631\n\n");
	SW_block8_1_conv(O630_SW,O631_SW,W631,B631);
	printf("[C_verifier.cpp]Calculate Lambda632\n\n");
	SW_block8_1(O617_SW, O631_SW, O632_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)633\n\n");
	SW_block8_1_ac(O632_SW,O633_SW);
	printf("[C_verifier.cpp]Calculate Conv2D634\n\n");
	SW_conv2d_169(O633_SW,O634_SW,W634);
	printf("[C_verifier.cpp]Calculate BatchNormalization635\n\n");
	SW_batch_normalization_169(O634_SW,O635_SW, W635);
	printf("[C_verifier.cpp]Calculate Activation(Relu)636\n\n");
	SW_activation_169(O635_SW,O636_SW);
	printf("[C_verifier.cpp]Calculate Conv2D637\n\n");
	SW_conv2d_170(O636_SW,O637_SW,W637);
	printf("[C_verifier.cpp]Calculate BatchNormalization638\n\n");
	SW_batch_normalization_170(O637_SW,O638_SW, W638);
	printf("[C_verifier.cpp]Calculate Activation(Relu)639\n\n");
	SW_activation_170(O638_SW,O639_SW);
	printf("[C_verifier.cpp]Calculate Conv2D640\n\n");
	SW_conv2d_168(O633_SW,O640_SW,W640);
	printf("[C_verifier.cpp]Calculate Conv2D641\n\n");
	SW_conv2d_171(O639_SW,O641_SW,W641);
	printf("[C_verifier.cpp]Calculate BatchNormalization642\n\n");
	SW_batch_normalization_168(O640_SW,O642_SW, W642);
	printf("[C_verifier.cpp]Calculate BatchNormalization643\n\n");
	SW_batch_normalization_171(O641_SW,O643_SW, W643);
	printf("[C_verifier.cpp]Calculate Activation(Relu)644\n\n");
	SW_activation_168(O642_SW,O644_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)645\n\n");
	SW_activation_171(O643_SW,O645_SW);
	printf("[C_verifier.cpp]Calculate Concatenate646\n\n");
	SW_block8_2_mixed(O644_SW, O645_SW, O646_SW);
	printf("[C_verifier.cpp]Calculate Conv2D647\n\n");
	SW_block8_2_conv(O646_SW,O647_SW,W647,B647);
	printf("[C_verifier.cpp]Calculate Lambda648\n\n");
	SW_block8_2(O633_SW, O647_SW, O648_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)649\n\n");
	SW_block8_2_ac(O648_SW,O649_SW);
	printf("[C_verifier.cpp]Calculate Conv2D650\n\n");
	SW_conv2d_173(O649_SW,O650_SW,W650);
	printf("[C_verifier.cpp]Calculate BatchNormalization651\n\n");
	SW_batch_normalization_173(O650_SW,O651_SW, W651);
	printf("[C_verifier.cpp]Calculate Activation(Relu)652\n\n");
	SW_activation_173(O651_SW,O652_SW);
	printf("[C_verifier.cpp]Calculate Conv2D653\n\n");
	SW_conv2d_174(O652_SW,O653_SW,W653);
	printf("[C_verifier.cpp]Calculate BatchNormalization654\n\n");
	SW_batch_normalization_174(O653_SW,O654_SW, W654);
	printf("[C_verifier.cpp]Calculate Activation(Relu)655\n\n");
	SW_activation_174(O654_SW,O655_SW);
	printf("[C_verifier.cpp]Calculate Conv2D656\n\n");
	SW_conv2d_172(O649_SW,O656_SW,W656);
	printf("[C_verifier.cpp]Calculate Conv2D657\n\n");
	SW_conv2d_175(O655_SW,O657_SW,W657);
	printf("[C_verifier.cpp]Calculate BatchNormalization658\n\n");
	SW_batch_normalization_172(O656_SW,O658_SW, W658);
	printf("[C_verifier.cpp]Calculate BatchNormalization659\n\n");
	SW_batch_normalization_175(O657_SW,O659_SW, W659);
	printf("[C_verifier.cpp]Calculate Activation(Relu)660\n\n");
	SW_activation_172(O658_SW,O660_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)661\n\n");
	SW_activation_175(O659_SW,O661_SW);
	printf("[C_verifier.cpp]Calculate Concatenate662\n\n");
	SW_block8_3_mixed(O660_SW, O661_SW, O662_SW);
	printf("[C_verifier.cpp]Calculate Conv2D663\n\n");
	SW_block8_3_conv(O662_SW,O663_SW,W663,B663);
	printf("[C_verifier.cpp]Calculate Lambda664\n\n");
	SW_block8_3(O649_SW, O663_SW, O664_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)665\n\n");
	SW_block8_3_ac(O664_SW,O665_SW);
	printf("[C_verifier.cpp]Calculate Conv2D666\n\n");
	SW_conv2d_177(O665_SW,O666_SW,W666);
	printf("[C_verifier.cpp]Calculate BatchNormalization667\n\n");
	SW_batch_normalization_177(O666_SW,O667_SW, W667);
	printf("[C_verifier.cpp]Calculate Activation(Relu)668\n\n");
	SW_activation_177(O667_SW,O668_SW);
	printf("[C_verifier.cpp]Calculate Conv2D669\n\n");
	SW_conv2d_178(O668_SW,O669_SW,W669);
	printf("[C_verifier.cpp]Calculate BatchNormalization670\n\n");
	SW_batch_normalization_178(O669_SW,O670_SW, W670);
	printf("[C_verifier.cpp]Calculate Activation(Relu)671\n\n");
	SW_activation_178(O670_SW,O671_SW);
	printf("[C_verifier.cpp]Calculate Conv2D672\n\n");
	SW_conv2d_176(O665_SW,O672_SW,W672);
	printf("[C_verifier.cpp]Calculate Conv2D673\n\n");
	SW_conv2d_179(O671_SW,O673_SW,W673);
	printf("[C_verifier.cpp]Calculate BatchNormalization674\n\n");
	SW_batch_normalization_176(O672_SW,O674_SW, W674);
	printf("[C_verifier.cpp]Calculate BatchNormalization675\n\n");
	SW_batch_normalization_179(O673_SW,O675_SW, W675);
	printf("[C_verifier.cpp]Calculate Activation(Relu)676\n\n");
	SW_activation_176(O674_SW,O676_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)677\n\n");
	SW_activation_179(O675_SW,O677_SW);
	printf("[C_verifier.cpp]Calculate Concatenate678\n\n");
	SW_block8_4_mixed(O676_SW, O677_SW, O678_SW);
	printf("[C_verifier.cpp]Calculate Conv2D679\n\n");
	SW_block8_4_conv(O678_SW,O679_SW,W679,B679);
	printf("[C_verifier.cpp]Calculate Lambda680\n\n");
	SW_block8_4(O665_SW, O679_SW, O680_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)681\n\n");
	SW_block8_4_ac(O680_SW,O681_SW);
	printf("[C_verifier.cpp]Calculate Conv2D682\n\n");
	SW_conv2d_181(O681_SW,O682_SW,W682);
	printf("[C_verifier.cpp]Calculate BatchNormalization683\n\n");
	SW_batch_normalization_181(O682_SW,O683_SW, W683);
	printf("[C_verifier.cpp]Calculate Activation(Relu)684\n\n");
	SW_activation_181(O683_SW,O684_SW);
	printf("[C_verifier.cpp]Calculate Conv2D685\n\n");
	SW_conv2d_182(O684_SW,O685_SW,W685);
	printf("[C_verifier.cpp]Calculate BatchNormalization686\n\n");
	SW_batch_normalization_182(O685_SW,O686_SW, W686);
	printf("[C_verifier.cpp]Calculate Activation(Relu)687\n\n");
	SW_activation_182(O686_SW,O687_SW);
	printf("[C_verifier.cpp]Calculate Conv2D688\n\n");
	SW_conv2d_180(O681_SW,O688_SW,W688);
	printf("[C_verifier.cpp]Calculate Conv2D689\n\n");
	SW_conv2d_183(O687_SW,O689_SW,W689);
	printf("[C_verifier.cpp]Calculate BatchNormalization690\n\n");
	SW_batch_normalization_180(O688_SW,O690_SW, W690);
	printf("[C_verifier.cpp]Calculate BatchNormalization691\n\n");
	SW_batch_normalization_183(O689_SW,O691_SW, W691);
	printf("[C_verifier.cpp]Calculate Activation(Relu)692\n\n");
	SW_activation_180(O690_SW,O692_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)693\n\n");
	SW_activation_183(O691_SW,O693_SW);
	printf("[C_verifier.cpp]Calculate Concatenate694\n\n");
	SW_block8_5_mixed(O692_SW, O693_SW, O694_SW);
	printf("[C_verifier.cpp]Calculate Conv2D695\n\n");
	SW_block8_5_conv(O694_SW,O695_SW,W695,B695);
	printf("[C_verifier.cpp]Calculate Lambda696\n\n");
	SW_block8_5(O681_SW, O695_SW, O696_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)697\n\n");
	SW_block8_5_ac(O696_SW,O697_SW);
	printf("[C_verifier.cpp]Calculate Conv2D698\n\n");
	SW_conv2d_185(O697_SW,O698_SW,W698);
	printf("[C_verifier.cpp]Calculate BatchNormalization699\n\n");
	SW_batch_normalization_185(O698_SW,O699_SW, W699);
	printf("[C_verifier.cpp]Calculate Activation(Relu)700\n\n");
	SW_activation_185(O699_SW,O700_SW);
	printf("[C_verifier.cpp]Calculate Conv2D701\n\n");
	SW_conv2d_186(O700_SW,O701_SW,W701);
	printf("[C_verifier.cpp]Calculate BatchNormalization702\n\n");
	SW_batch_normalization_186(O701_SW,O702_SW, W702);
	printf("[C_verifier.cpp]Calculate Activation(Relu)703\n\n");
	SW_activation_186(O702_SW,O703_SW);
	printf("[C_verifier.cpp]Calculate Conv2D704\n\n");
	SW_conv2d_184(O697_SW,O704_SW,W704);
	printf("[C_verifier.cpp]Calculate Conv2D705\n\n");
	SW_conv2d_187(O703_SW,O705_SW,W705);
	printf("[C_verifier.cpp]Calculate BatchNormalization706\n\n");
	SW_batch_normalization_184(O704_SW,O706_SW, W706);
	printf("[C_verifier.cpp]Calculate BatchNormalization707\n\n");
	SW_batch_normalization_187(O705_SW,O707_SW, W707);
	printf("[C_verifier.cpp]Calculate Activation(Relu)708\n\n");
	SW_activation_184(O706_SW,O708_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)709\n\n");
	SW_activation_187(O707_SW,O709_SW);
	printf("[C_verifier.cpp]Calculate Concatenate710\n\n");
	SW_block8_6_mixed(O708_SW, O709_SW, O710_SW);
	printf("[C_verifier.cpp]Calculate Conv2D711\n\n");
	SW_block8_6_conv(O710_SW,O711_SW,W711,B711);
	printf("[C_verifier.cpp]Calculate Lambda712\n\n");
	SW_block8_6(O697_SW, O711_SW, O712_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)713\n\n");
	SW_block8_6_ac(O712_SW,O713_SW);
	printf("[C_verifier.cpp]Calculate Conv2D714\n\n");
	SW_conv2d_189(O713_SW,O714_SW,W714);
	printf("[C_verifier.cpp]Calculate BatchNormalization715\n\n");
	SW_batch_normalization_189(O714_SW,O715_SW, W715);
	printf("[C_verifier.cpp]Calculate Activation(Relu)716\n\n");
	SW_activation_189(O715_SW,O716_SW);
	printf("[C_verifier.cpp]Calculate Conv2D717\n\n");
	SW_conv2d_190(O716_SW,O717_SW,W717);
	printf("[C_verifier.cpp]Calculate BatchNormalization718\n\n");
	SW_batch_normalization_190(O717_SW,O718_SW, W718);
	printf("[C_verifier.cpp]Calculate Activation(Relu)719\n\n");
	SW_activation_190(O718_SW,O719_SW);
	printf("[C_verifier.cpp]Calculate Conv2D720\n\n");
	SW_conv2d_188(O713_SW,O720_SW,W720);
	printf("[C_verifier.cpp]Calculate Conv2D721\n\n");
	SW_conv2d_191(O719_SW,O721_SW,W721);
	printf("[C_verifier.cpp]Calculate BatchNormalization722\n\n");
	SW_batch_normalization_188(O720_SW,O722_SW, W722);
	printf("[C_verifier.cpp]Calculate BatchNormalization723\n\n");
	SW_batch_normalization_191(O721_SW,O723_SW, W723);
	printf("[C_verifier.cpp]Calculate Activation(Relu)724\n\n");
	SW_activation_188(O722_SW,O724_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)725\n\n");
	SW_activation_191(O723_SW,O725_SW);
	printf("[C_verifier.cpp]Calculate Concatenate726\n\n");
	SW_block8_7_mixed(O724_SW, O725_SW, O726_SW);
	printf("[C_verifier.cpp]Calculate Conv2D727\n\n");
	SW_block8_7_conv(O726_SW,O727_SW,W727,B727);
	printf("[C_verifier.cpp]Calculate Lambda728\n\n");
	SW_block8_7(O713_SW, O727_SW, O728_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)729\n\n");
	SW_block8_7_ac(O728_SW,O729_SW);
	printf("[C_verifier.cpp]Calculate Conv2D730\n\n");
	SW_conv2d_193(O729_SW,O730_SW,W730);
	printf("[C_verifier.cpp]Calculate BatchNormalization731\n\n");
	SW_batch_normalization_193(O730_SW,O731_SW, W731);
	printf("[C_verifier.cpp]Calculate Activation(Relu)732\n\n");
	SW_activation_193(O731_SW,O732_SW);
	printf("[C_verifier.cpp]Calculate Conv2D733\n\n");
	SW_conv2d_194(O732_SW,O733_SW,W733);
	printf("[C_verifier.cpp]Calculate BatchNormalization734\n\n");
	SW_batch_normalization_194(O733_SW,O734_SW, W734);
	printf("[C_verifier.cpp]Calculate Activation(Relu)735\n\n");
	SW_activation_194(O734_SW,O735_SW);
	printf("[C_verifier.cpp]Calculate Conv2D736\n\n");
	SW_conv2d_192(O729_SW,O736_SW,W736);
	printf("[C_verifier.cpp]Calculate Conv2D737\n\n");
	SW_conv2d_195(O735_SW,O737_SW,W737);
	printf("[C_verifier.cpp]Calculate BatchNormalization738\n\n");
	SW_batch_normalization_192(O736_SW,O738_SW, W738);
	printf("[C_verifier.cpp]Calculate BatchNormalization739\n\n");
	SW_batch_normalization_195(O737_SW,O739_SW, W739);
	printf("[C_verifier.cpp]Calculate Activation(Relu)740\n\n");
	SW_activation_192(O738_SW,O740_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)741\n\n");
	SW_activation_195(O739_SW,O741_SW);
	printf("[C_verifier.cpp]Calculate Concatenate742\n\n");
	SW_block8_8_mixed(O740_SW, O741_SW, O742_SW);
	printf("[C_verifier.cpp]Calculate Conv2D743\n\n");
	SW_block8_8_conv(O742_SW,O743_SW,W743,B743);
	printf("[C_verifier.cpp]Calculate Lambda744\n\n");
	SW_block8_8(O729_SW, O743_SW, O744_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)745\n\n");
	SW_block8_8_ac(O744_SW,O745_SW);
	printf("[C_verifier.cpp]Calculate Conv2D746\n\n");
	SW_conv2d_197(O745_SW,O746_SW,W746);
	printf("[C_verifier.cpp]Calculate BatchNormalization747\n\n");
	SW_batch_normalization_197(O746_SW,O747_SW, W747);
	printf("[C_verifier.cpp]Calculate Activation(Relu)748\n\n");
	SW_activation_197(O747_SW,O748_SW);
	printf("[C_verifier.cpp]Calculate Conv2D749\n\n");
	SW_conv2d_198(O748_SW,O749_SW,W749);
	printf("[C_verifier.cpp]Calculate BatchNormalization750\n\n");
	SW_batch_normalization_198(O749_SW,O750_SW, W750);
	printf("[C_verifier.cpp]Calculate Activation(Relu)751\n\n");
	SW_activation_198(O750_SW,O751_SW);
	printf("[C_verifier.cpp]Calculate Conv2D752\n\n");
	SW_conv2d_196(O745_SW,O752_SW,W752);
	printf("[C_verifier.cpp]Calculate Conv2D753\n\n");
	SW_conv2d_199(O751_SW,O753_SW,W753);
	printf("[C_verifier.cpp]Calculate BatchNormalization754\n\n");
	SW_batch_normalization_196(O752_SW,O754_SW, W754);
	printf("[C_verifier.cpp]Calculate BatchNormalization755\n\n");
	SW_batch_normalization_199(O753_SW,O755_SW, W755);
	printf("[C_verifier.cpp]Calculate Activation(Relu)756\n\n");
	SW_activation_196(O754_SW,O756_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)757\n\n");
	SW_activation_199(O755_SW,O757_SW);
	printf("[C_verifier.cpp]Calculate Concatenate758\n\n");
	SW_block8_9_mixed(O756_SW, O757_SW, O758_SW);
	printf("[C_verifier.cpp]Calculate Conv2D759\n\n");
	SW_block8_9_conv(O758_SW,O759_SW,W759,B759);
	printf("[C_verifier.cpp]Calculate Lambda760\n\n");
	SW_block8_9(O745_SW, O759_SW, O760_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)761\n\n");
	SW_block8_9_ac(O760_SW,O761_SW);
	printf("[C_verifier.cpp]Calculate Conv2D762\n\n");
	SW_conv2d_201(O761_SW,O762_SW,W762);
	printf("[C_verifier.cpp]Calculate BatchNormalization763\n\n");
	SW_batch_normalization_201(O762_SW,O763_SW, W763);
	printf("[C_verifier.cpp]Calculate Activation(Relu)764\n\n");
	SW_activation_201(O763_SW,O764_SW);
	printf("[C_verifier.cpp]Calculate Conv2D765\n\n");
	SW_conv2d_202(O764_SW,O765_SW,W765);
	printf("[C_verifier.cpp]Calculate BatchNormalization766\n\n");
	SW_batch_normalization_202(O765_SW,O766_SW, W766);
	printf("[C_verifier.cpp]Calculate Activation(Relu)767\n\n");
	SW_activation_202(O766_SW,O767_SW);
	printf("[C_verifier.cpp]Calculate Conv2D768\n\n");
	SW_conv2d_200(O761_SW,O768_SW,W768);
	printf("[C_verifier.cpp]Calculate Conv2D769\n\n");
	SW_conv2d_203(O767_SW,O769_SW,W769);
	printf("[C_verifier.cpp]Calculate BatchNormalization770\n\n");
	SW_batch_normalization_200(O768_SW,O770_SW, W770);
	printf("[C_verifier.cpp]Calculate BatchNormalization771\n\n");
	SW_batch_normalization_203(O769_SW,O771_SW, W771);
	printf("[C_verifier.cpp]Calculate Activation(Relu)772\n\n");
	SW_activation_200(O770_SW,O772_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)773\n\n");
	SW_activation_203(O771_SW,O773_SW);
	printf("[C_verifier.cpp]Calculate Concatenate774\n\n");
	SW_block8_10_mixed(O772_SW, O773_SW, O774_SW);
	printf("[C_verifier.cpp]Calculate Conv2D775\n\n");
	SW_block8_10_conv(O774_SW,O775_SW,W775,B775);
	printf("[C_verifier.cpp]Calculate Lambda776\n\n");
	SW_block8_10(O761_SW, O775_SW, O776_SW);
	printf("[C_verifier.cpp]Calculate Conv2D777\n\n");
	SW_conv_7b(O776_SW,O777_SW,W777);
	printf("[C_verifier.cpp]Calculate BatchNormalization778\n\n");
	SW_conv_7b_bn(O777_SW,O778_SW, W778);
	printf("[C_verifier.cpp]Calculate Activation(Relu)779\n\n");
	SW_conv_7b_ac(O778_SW,O779_SW);
	printf("[C_verifier.cpp]Calculate GlobalAveragePooling2D780\n\n");
	SW_avg_pool(O779_SW,O780_SW);
	printf("[C_verifier.cpp]Calculate Dense781\n\n");
	SW_predictions(O780_SW,O781_SW,W781,B781);
	

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
		for(y = 0; y < 96 ; y++) {
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
		for(y = 0; y < 64 ; y++) {
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
		for(y = 0; y < 96 ; y++) {
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
		for(y = 0; y < 64 ; y++) {
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
		for(y = 0; y < 96 ; y++) {
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
		for(y = 0; y < 64 ; y++) {
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
		for(y = 0; y < 320 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O41_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O41_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O42_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O42_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O43_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O43_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O44_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O44_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O45_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O45_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O46_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O46_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O47_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O47_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O48_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O48_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O49_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O49_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O50_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O50_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O51_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O51_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O52_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O52_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O53_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O53_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O54_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O54_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O55_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O55_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O56_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O56_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O57_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O57_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O58_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O58_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O59_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O59_SW[y][k][x]);
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
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O60_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O60_SW[y][k][x]);
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


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O61_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O61_SW[y][k][x]);
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
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O62_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O62_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O63_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O63_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O64_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O64_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O65_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O65_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O66_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O66_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O67_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O67_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O68_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O68_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O69_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O69_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O70_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O70_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O71_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O71_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O72_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O72_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O73_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O73_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O74_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O74_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O75_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O75_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O76_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O76_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O77_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O77_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O78_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O78_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O79_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O79_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O80_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O80_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O81_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O81_SW[y][k][x]);
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
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O82_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O82_SW[y][k][x]);
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


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O83_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O83_SW[y][k][x]);
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
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O84_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O84_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O85_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O85_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O86_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O86_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O87_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O87_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O88_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O88_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O89_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O89_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O90_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O90_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O91_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O91_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O92_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O92_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O93_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O93_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O94_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O94_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O95_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O95_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O96_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O96_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O97_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O97_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O98_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O98_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O99_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O99_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O100_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O100_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O101_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O101_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O102_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O102_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O103_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O103_SW[y][k][x]);
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
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O104_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O104_SW[y][k][x]);
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


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O105_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O105_SW[y][k][x]);
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
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O106_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O106_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O107_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O107_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O108_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O108_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O109_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O109_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O110_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O110_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O111_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O111_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O112_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O112_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O113_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O113_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O114_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O114_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O115_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O115_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O116_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O116_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O117_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O117_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O118_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O118_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O119_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O119_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O120_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O120_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O121_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O121_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O122_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O122_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O123_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O123_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O124_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O124_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O125_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O125_SW[y][k][x]);
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
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O126_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O126_SW[y][k][x]);
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


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O127_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O127_SW[y][k][x]);
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
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O128_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O128_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O129_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O129_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O130_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O130_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O131_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O131_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O132_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O132_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O133_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O133_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O134_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O134_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O135_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O135_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O136_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O136_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O137_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O137_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O138_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O138_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O139_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O139_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O140_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O140_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O141_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O141_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O142_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O142_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O143_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O143_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O144_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O144_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O145_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O145_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O146_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O146_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O147_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O147_SW[y][k][x]);
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
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O148_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O148_SW[y][k][x]);
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


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O149_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O149_SW[y][k][x]);
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
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O150_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O150_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O151_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O151_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O152_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O152_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O153_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O153_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O154_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O154_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O155_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O155_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O156_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O156_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O157_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O157_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O158_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O158_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O159_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O159_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O160_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O160_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O161_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O161_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O162_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O162_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O163_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O163_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O164_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O164_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O165_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O165_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O166_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O166_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O167_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O167_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O168_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O168_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O169_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O169_SW[y][k][x]);
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
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O170_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O170_SW[y][k][x]);
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


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O171_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O171_SW[y][k][x]);
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
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O172_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O172_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O173_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O173_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O174_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O174_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O175_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O175_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O176_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O176_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O177_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O177_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O178_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O178_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O179_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O179_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O180_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O180_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O181_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O181_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O182_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O182_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O183_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O183_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O184_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O184_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O185_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O185_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O186_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O186_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O187_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O187_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O188_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O188_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O189_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O189_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O190_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O190_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O191_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O191_SW[y][k][x]);
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
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O192_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O192_SW[y][k][x]);
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


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O193_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O193_SW[y][k][x]);
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
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O194_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O194_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O195_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O195_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O196_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O196_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O197_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O197_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O198_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O198_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O199_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O199_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O200_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O200_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O201_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O201_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O202_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O202_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O203_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O203_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O204_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O204_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O205_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O205_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O206_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O206_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O207_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O207_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O208_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O208_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O209_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O209_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O210_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O210_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O211_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O211_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O212_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O212_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O213_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O213_SW[y][k][x]);
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
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O214_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O214_SW[y][k][x]);
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


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O215_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O215_SW[y][k][x]);
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
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O216_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O216_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O217_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O217_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O218_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O218_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O219_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O219_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O220_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O220_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O221_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O221_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O222_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O222_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O223_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O223_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O224_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O224_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O225_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O225_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O226_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O226_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O227_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O227_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O228_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O228_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O229_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O229_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O230_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O230_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O231_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O231_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O232_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O232_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O233_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O233_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O234_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O234_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O235_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O235_SW[y][k][x]);
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
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O236_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O236_SW[y][k][x]);
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


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O237_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O237_SW[y][k][x]);
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
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O238_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O238_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O239_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O239_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O240_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O240_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O241_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O241_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O242_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O242_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O243_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O243_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O244_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O244_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O245_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O245_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O246_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O246_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O247_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O247_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O248_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O248_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O249_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O249_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O250_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O250_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O251_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O251_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O252_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O252_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O253_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O253_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O254_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O254_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O255_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O255_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O256_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O256_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O257_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O257_SW[y][k][x]);
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
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O258_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O258_SW[y][k][x]);
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


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O259_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O259_SW[y][k][x]);
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
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O260_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O260_SW[y][k][x]);
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
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O261_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O261_SW[y][k][x]);
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
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O262_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O262_SW[y][k][x]);
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
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O263_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O263_SW[y][k][x]);
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
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O264_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O264_SW[y][k][x]);
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
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O265_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O265_SW[y][k][x]);
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
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O266_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O266_SW[y][k][x]);
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
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O267_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O267_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O268_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O268_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O269_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O269_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O270_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O270_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O271_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O271_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O272_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O272_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","MaxPooling2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O273_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O273_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O274_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O274_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O275_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O275_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O276_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O276_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O277_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O277_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O278_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O278_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O279_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O279_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O280_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O280_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O281_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O281_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O282_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O282_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O283_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O283_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O284_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O284_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O285_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O285_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O286_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O286_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O287_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O287_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O288_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O288_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O289_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O289_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O290_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O290_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O291_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O291_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O292_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O292_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O293_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O293_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O294_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O294_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O295_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O295_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O296_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O296_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O297_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O297_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O298_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O298_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O299_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O299_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O300_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O300_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O301_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O301_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O302_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O302_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O303_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O303_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O304_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O304_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O305_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O305_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O306_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O306_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O307_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O307_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O308_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O308_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O309_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O309_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O310_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O310_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O311_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O311_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O312_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O312_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O313_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O313_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O314_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O314_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O315_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O315_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O316_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O316_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O317_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O317_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O318_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O318_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O319_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O319_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O320_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O320_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O321_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O321_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O322_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O322_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O323_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O323_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O324_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O324_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O325_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O325_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O326_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O326_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O327_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O327_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O328_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O328_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O329_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O329_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O330_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O330_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O331_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O331_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O332_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O332_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O333_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O333_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O334_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O334_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O335_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O335_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O336_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O336_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O337_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O337_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O338_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O338_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O339_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O339_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O340_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O340_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O341_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O341_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O342_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O342_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O343_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O343_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O344_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O344_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O345_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O345_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O346_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O346_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O347_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O347_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O348_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O348_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O349_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O349_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O350_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O350_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O351_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O351_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O352_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O352_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O353_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O353_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O354_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O354_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O355_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O355_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O356_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O356_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O357_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O357_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O358_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O358_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O359_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O359_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O360_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O360_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O361_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O361_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O362_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O362_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O363_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O363_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O364_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O364_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O365_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O365_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O366_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O366_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O367_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O367_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O368_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O368_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O369_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O369_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O370_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O370_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O371_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O371_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O372_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O372_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O373_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O373_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O374_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O374_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O375_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O375_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O376_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O376_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O377_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O377_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O378_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O378_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O379_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O379_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O380_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O380_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O381_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O381_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O382_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O382_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O383_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O383_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O384_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O384_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O385_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O385_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O386_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O386_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O387_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O387_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O388_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O388_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O389_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O389_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O390_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O390_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O391_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O391_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O392_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O392_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O393_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O393_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O394_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O394_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O395_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O395_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O396_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O396_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O397_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O397_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O398_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O398_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O399_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O399_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O400_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O400_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O401_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O401_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O402_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O402_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O403_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O403_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O404_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O404_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O405_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O405_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O406_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O406_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O407_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O407_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O408_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O408_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O409_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O409_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O410_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O410_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O411_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O411_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O412_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O412_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O413_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O413_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O414_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O414_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O415_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O415_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O416_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O416_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O417_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O417_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O418_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O418_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O419_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O419_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O420_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O420_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O421_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O421_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O422_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O422_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O423_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O423_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O424_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O424_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O425_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O425_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O426_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O426_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O427_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O427_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O428_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O428_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O429_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O429_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O430_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O430_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O431_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O431_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O432_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O432_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O433_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O433_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O434_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O434_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O435_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O435_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O436_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O436_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O437_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O437_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O438_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O438_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O439_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O439_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O440_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O440_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O441_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O441_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O442_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O442_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O443_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O443_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O444_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O444_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O445_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O445_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O446_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O446_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O447_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O447_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O448_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O448_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O449_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O449_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O450_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O450_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O451_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O451_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O452_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O452_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O453_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O453_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O454_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O454_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O455_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O455_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O456_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O456_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O457_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O457_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O458_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O458_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O459_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O459_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O460_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O460_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O461_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O461_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O462_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O462_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O463_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O463_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O464_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O464_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O465_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O465_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O466_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O466_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O467_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O467_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O468_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O468_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O469_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O469_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O470_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O470_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O471_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O471_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O472_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O472_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O473_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O473_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O474_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O474_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O475_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O475_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O476_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O476_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O477_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O477_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O478_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O478_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O479_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O479_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O480_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O480_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O481_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O481_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O482_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O482_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O483_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O483_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O484_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O484_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O485_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O485_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O486_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O486_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O487_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O487_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O488_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O488_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O489_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O489_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O490_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O490_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O491_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O491_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O492_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O492_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O493_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O493_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O494_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O494_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O495_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O495_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O496_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O496_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O497_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O497_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O498_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O498_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O499_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O499_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O500_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O500_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O501_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O501_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O502_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O502_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O503_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O503_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O504_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O504_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O505_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O505_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O506_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O506_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O507_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O507_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O508_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O508_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O509_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O509_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O510_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O510_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O511_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O511_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O512_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O512_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O513_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O513_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O514_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O514_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O515_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O515_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O516_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O516_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O517_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O517_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O518_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O518_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O519_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O519_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O520_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O520_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O521_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O521_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O522_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O522_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O523_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O523_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O524_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O524_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O525_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O525_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O526_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O526_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O527_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O527_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O528_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O528_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O529_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O529_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O530_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O530_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O531_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O531_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O532_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O532_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O533_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O533_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O534_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O534_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O535_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O535_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O536_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O536_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O537_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O537_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O538_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O538_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O539_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O539_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O540_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O540_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O541_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O541_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O542_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O542_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O543_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O543_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O544_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O544_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O545_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O545_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O546_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O546_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O547_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O547_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O548_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O548_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O549_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O549_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O550_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O550_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O551_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O551_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O552_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O552_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O553_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O553_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O554_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O554_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O555_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O555_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O556_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O556_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O557_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O557_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O558_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O558_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O559_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O559_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O560_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O560_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O561_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O561_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O562_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O562_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O563_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O563_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O564_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O564_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O565_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O565_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O566_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O566_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O567_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O567_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O568_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O568_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O569_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O569_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O570_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O570_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O571_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O571_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O572_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O572_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O573_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O573_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O574_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O574_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O575_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O575_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O576_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O576_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O577_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O577_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O578_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O578_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O579_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O579_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O580_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O580_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O581_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O581_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O582_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O582_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O583_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O583_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O584_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O584_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O585_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O585_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O586_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O586_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O587_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O587_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O588_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O588_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O589_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O589_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O590_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O590_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O591_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O591_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O592_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O592_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O593_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O593_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O594_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O594_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O595_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O595_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O596_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O596_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O597_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O597_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O598_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O598_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O599_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O599_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 288 ; y++) {
			fprintf(o_stream,"%.6f ",O600_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O600_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O601_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O601_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O602_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O602_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 288 ; y++) {
			fprintf(o_stream,"%.6f ",O603_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O603_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O604_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O604_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O605_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O605_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 288 ; y++) {
			fprintf(o_stream,"%.6f ",O606_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O606_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O607_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O607_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 288 ; y++) {
			fprintf(o_stream,"%.6f ",O608_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O608_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O609_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O609_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O610_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O610_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 288 ; y++) {
			fprintf(o_stream,"%.6f ",O611_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O611_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O612_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O612_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O613_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O613_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 288 ; y++) {
			fprintf(o_stream,"%.6f ",O614_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O614_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O615_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O615_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","MaxPooling2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O616_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O616_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2080 ; y++) {
			fprintf(o_stream,"%.6f ",O617_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O617_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O618_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O618_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O619_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O619_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O620_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O620_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O621_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O621_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O622_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O622_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O623_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O623_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O624_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O624_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O625_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O625_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O626_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O626_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O627_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O627_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O628_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O628_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O629_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O629_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 448 ; y++) {
			fprintf(o_stream,"%.6f ",O630_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O630_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2080 ; y++) {
			fprintf(o_stream,"%.6f ",O631_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O631_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2080 ; y++) {
			fprintf(o_stream,"%.6f ",O632_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O632_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2080 ; y++) {
			fprintf(o_stream,"%.6f ",O633_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O633_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O634_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O634_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O635_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O635_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O636_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O636_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O637_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O637_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O638_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O638_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O639_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O639_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O640_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O640_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O641_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O641_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O642_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O642_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O643_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O643_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O644_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O644_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O645_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O645_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 448 ; y++) {
			fprintf(o_stream,"%.6f ",O646_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O646_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2080 ; y++) {
			fprintf(o_stream,"%.6f ",O647_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O647_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2080 ; y++) {
			fprintf(o_stream,"%.6f ",O648_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O648_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2080 ; y++) {
			fprintf(o_stream,"%.6f ",O649_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O649_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O650_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O650_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O651_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O651_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O652_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O652_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O653_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O653_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O654_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O654_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O655_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O655_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O656_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O656_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O657_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O657_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O658_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O658_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O659_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O659_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O660_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O660_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O661_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O661_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 448 ; y++) {
			fprintf(o_stream,"%.6f ",O662_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O662_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2080 ; y++) {
			fprintf(o_stream,"%.6f ",O663_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O663_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2080 ; y++) {
			fprintf(o_stream,"%.6f ",O664_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O664_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2080 ; y++) {
			fprintf(o_stream,"%.6f ",O665_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O665_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O666_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O666_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O667_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O667_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O668_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O668_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O669_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O669_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O670_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O670_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O671_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O671_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O672_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O672_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O673_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O673_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O674_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O674_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O675_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O675_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O676_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O676_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O677_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O677_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 448 ; y++) {
			fprintf(o_stream,"%.6f ",O678_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O678_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2080 ; y++) {
			fprintf(o_stream,"%.6f ",O679_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O679_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2080 ; y++) {
			fprintf(o_stream,"%.6f ",O680_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O680_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2080 ; y++) {
			fprintf(o_stream,"%.6f ",O681_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O681_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O682_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O682_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O683_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O683_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O684_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O684_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O685_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O685_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O686_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O686_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O687_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O687_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O688_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O688_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O689_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O689_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O690_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O690_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O691_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O691_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O692_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O692_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O693_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O693_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 448 ; y++) {
			fprintf(o_stream,"%.6f ",O694_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O694_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2080 ; y++) {
			fprintf(o_stream,"%.6f ",O695_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O695_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2080 ; y++) {
			fprintf(o_stream,"%.6f ",O696_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O696_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2080 ; y++) {
			fprintf(o_stream,"%.6f ",O697_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O697_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O698_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O698_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O699_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O699_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O700_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O700_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O701_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O701_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O702_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O702_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O703_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O703_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O704_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O704_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O705_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O705_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O706_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O706_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O707_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O707_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O708_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O708_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O709_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O709_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 448 ; y++) {
			fprintf(o_stream,"%.6f ",O710_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O710_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2080 ; y++) {
			fprintf(o_stream,"%.6f ",O711_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O711_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2080 ; y++) {
			fprintf(o_stream,"%.6f ",O712_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O712_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2080 ; y++) {
			fprintf(o_stream,"%.6f ",O713_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O713_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O714_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O714_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O715_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O715_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O716_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O716_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O717_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O717_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O718_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O718_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O719_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O719_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O720_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O720_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O721_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O721_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O722_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O722_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O723_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O723_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O724_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O724_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O725_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O725_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 448 ; y++) {
			fprintf(o_stream,"%.6f ",O726_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O726_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2080 ; y++) {
			fprintf(o_stream,"%.6f ",O727_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O727_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2080 ; y++) {
			fprintf(o_stream,"%.6f ",O728_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O728_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2080 ; y++) {
			fprintf(o_stream,"%.6f ",O729_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O729_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O730_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O730_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O731_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O731_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O732_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O732_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O733_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O733_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O734_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O734_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O735_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O735_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O736_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O736_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O737_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O737_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O738_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O738_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O739_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O739_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O740_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O740_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O741_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O741_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 448 ; y++) {
			fprintf(o_stream,"%.6f ",O742_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O742_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2080 ; y++) {
			fprintf(o_stream,"%.6f ",O743_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O743_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2080 ; y++) {
			fprintf(o_stream,"%.6f ",O744_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O744_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2080 ; y++) {
			fprintf(o_stream,"%.6f ",O745_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O745_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O746_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O746_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O747_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O747_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O748_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O748_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O749_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O749_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O750_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O750_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O751_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O751_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O752_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O752_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O753_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O753_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O754_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O754_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O755_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O755_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O756_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O756_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O757_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O757_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 448 ; y++) {
			fprintf(o_stream,"%.6f ",O758_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O758_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2080 ; y++) {
			fprintf(o_stream,"%.6f ",O759_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O759_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2080 ; y++) {
			fprintf(o_stream,"%.6f ",O760_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O760_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2080 ; y++) {
			fprintf(o_stream,"%.6f ",O761_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O761_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O762_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O762_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O763_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O763_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O764_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O764_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O765_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O765_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O766_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O766_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O767_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O767_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O768_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O768_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O769_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O769_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O770_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O770_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O771_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O771_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O772_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O772_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O773_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O773_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 448 ; y++) {
			fprintf(o_stream,"%.6f ",O774_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O774_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2080 ; y++) {
			fprintf(o_stream,"%.6f ",O775_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O775_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Lambda : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2080 ; y++) {
			fprintf(o_stream,"%.6f ",O776_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O776_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1536 ; y++) {
			fprintf(o_stream,"%.6f ",O777_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O777_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1536 ; y++) {
			fprintf(o_stream,"%.6f ",O778_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O778_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1536 ; y++) {
			fprintf(o_stream,"%.6f ",O779_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O779_SW[y][k][x]);
		}
		if(x != 5 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 5 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","GlobalAveragePooling2D : [[");
for (k = 0; k <  1536 ; k++) {
	fprintf(o_stream,"%.6f ",O780_SW[k]);
	fprintf(c_num,"%.6f ",O780_SW[k]);
}
fprintf(o_stream,"%s","]]\n\n");


fprintf(o_stream,"%s","Dense : [[");
for (k = 0; k <  1000 ; k++) {
	fprintf(o_stream,"%.6f ",O781_SW[k]);
	fprintf(c_num,"%.6f ",O781_SW[k]);
}
fprintf(o_stream,"%s","]]\n\n");





    fclose(w_stream);
    fclose(i_stream);
    fclose(o_stream);
    fclose(c_num);

    return 0;
}
