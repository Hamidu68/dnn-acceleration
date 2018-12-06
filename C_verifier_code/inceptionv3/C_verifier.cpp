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
void SW_activation_50(DATA_T I[32][111][111], DATA_T O[32][111][111]) {
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
void SW_activation_51(DATA_T I[32][109][109], DATA_T O[32][109][109]) {
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
void SW_activation_52(DATA_T I[64][109][109], DATA_T O[64][109][109]) {
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
void SW_activation_53(DATA_T I[80][54][54], DATA_T O[80][54][54]) {
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
void SW_activation_54(DATA_T I[192][52][52], DATA_T O[192][52][52]) {
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
void SW_activation_58(DATA_T I[64][25][25], DATA_T O[64][25][25]) {
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
void SW_activation_56(DATA_T I[48][25][25], DATA_T O[48][25][25]) {
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

void SW_activation_59(DATA_T I[96][25][25], DATA_T O[96][25][25]) {
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
void SW_activation_55(DATA_T I[64][25][25], DATA_T O[64][25][25]) {
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

void SW_activation_57(DATA_T I[64][25][25], DATA_T O[64][25][25]) {
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

void SW_activation_60(DATA_T I[96][25][25], DATA_T O[96][25][25]) {
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

void SW_mixed0(DATA_T I1[64][25][25], DATA_T I2[64][25][25], DATA_T I3[96][25][25], DATA_T I4[32][25][25], DATA_T O[256][25][25]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 25; x++) {
		for(y = 0; y < 25; y++) {
			ch=0;
			for(k = ch; k < ch+64; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=64;
			for(k = ch; k < ch+64; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
			ch+=64;
			for(k = ch; k < ch+96; k++){
				O[k][x][y] = I3[k-ch][x][y];
			}
			ch+=96;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I4[k-ch][x][y];
			}
		}
	}

}
void SW_conv2d_16(DATA_T I[256][25][25], DATA_T O[64][25][25], DATA_T W[64][256][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<256; k++) {
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

void SW_batch_normalization_16(DATA_T I[64][25][25], DATA_T O[64][25][25], DATA_T W[3][64]) {
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
void SW_activation_65(DATA_T I[64][25][25], DATA_T O[64][25][25]) {
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

void SW_conv2d_14(DATA_T I[256][25][25], DATA_T O[48][25][25], DATA_T W[48][256][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<48; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<256; k++) {
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

void SW_conv2d_17(DATA_T I[64][25][25], DATA_T O[96][25][25], DATA_T W[96][64][3][3]) {
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

void SW_batch_normalization_14(DATA_T I[48][25][25], DATA_T O[48][25][25], DATA_T W[3][48]) {
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
void SW_batch_normalization_17(DATA_T I[96][25][25], DATA_T O[96][25][25], DATA_T W[3][96]) {
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
void SW_activation_63(DATA_T I[48][25][25], DATA_T O[48][25][25]) {
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

void SW_activation_66(DATA_T I[96][25][25], DATA_T O[96][25][25]) {
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

void SW_average_pooling2d_2(DATA_T I[256][25][25], DATA_T O[256][25][25])
{
	int m, x, y, i, j;
	DATA_T sum;
	int p = (1*(25-1) - 25 + 3)/2;
    DATA_T div;
	for (m = 0; m<256; m++) {
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
void SW_conv2d_13(DATA_T I[256][25][25], DATA_T O[64][25][25], DATA_T W[64][256][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<256; k++) {
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

void SW_conv2d_15(DATA_T I[48][25][25], DATA_T O[64][25][25], DATA_T W[64][48][5][5]) {
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

void SW_conv2d_18(DATA_T I[96][25][25], DATA_T O[96][25][25], DATA_T W[96][96][3][3]) {
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

void SW_conv2d_19(DATA_T I[256][25][25], DATA_T O[64][25][25], DATA_T W[64][256][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<256; k++) {
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

void SW_batch_normalization_13(DATA_T I[64][25][25], DATA_T O[64][25][25], DATA_T W[3][64]) {
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
void SW_batch_normalization_15(DATA_T I[64][25][25], DATA_T O[64][25][25], DATA_T W[3][64]) {
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
void SW_batch_normalization_18(DATA_T I[96][25][25], DATA_T O[96][25][25], DATA_T W[3][96]) {
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
void SW_batch_normalization_19(DATA_T I[64][25][25], DATA_T O[64][25][25], DATA_T W[3][64]) {
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
void SW_activation_62(DATA_T I[64][25][25], DATA_T O[64][25][25]) {
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

void SW_activation_64(DATA_T I[64][25][25], DATA_T O[64][25][25]) {
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

void SW_activation_67(DATA_T I[96][25][25], DATA_T O[96][25][25]) {
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

void SW_activation_68(DATA_T I[64][25][25], DATA_T O[64][25][25]) {
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

void SW_mixed1(DATA_T I1[64][25][25], DATA_T I2[64][25][25], DATA_T I3[96][25][25], DATA_T I4[64][25][25], DATA_T O[288][25][25]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 25; x++) {
		for(y = 0; y < 25; y++) {
			ch=0;
			for(k = ch; k < ch+64; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=64;
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
void SW_conv2d_23(DATA_T I[288][25][25], DATA_T O[64][25][25], DATA_T W[64][288][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<288; k++) {
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

void SW_batch_normalization_23(DATA_T I[64][25][25], DATA_T O[64][25][25], DATA_T W[3][64]) {
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

void SW_conv2d_21(DATA_T I[288][25][25], DATA_T O[48][25][25], DATA_T W[48][288][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<48; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<288; k++) {
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

void SW_conv2d_24(DATA_T I[64][25][25], DATA_T O[96][25][25], DATA_T W[96][64][3][3]) {
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

void SW_batch_normalization_21(DATA_T I[48][25][25], DATA_T O[48][25][25], DATA_T W[3][48]) {
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
void SW_batch_normalization_24(DATA_T I[96][25][25], DATA_T O[96][25][25], DATA_T W[3][96]) {
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
void SW_activation_70(DATA_T I[48][25][25], DATA_T O[48][25][25]) {
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

void SW_activation_73(DATA_T I[96][25][25], DATA_T O[96][25][25]) {
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

void SW_average_pooling2d_3(DATA_T I[288][25][25], DATA_T O[288][25][25])
{
	int m, x, y, i, j;
	DATA_T sum;
	int p = (1*(25-1) - 25 + 3)/2;
    DATA_T div;
	for (m = 0; m<288; m++) {
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
void SW_conv2d_20(DATA_T I[288][25][25], DATA_T O[64][25][25], DATA_T W[64][288][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<288; k++) {
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

void SW_conv2d_22(DATA_T I[48][25][25], DATA_T O[64][25][25], DATA_T W[64][48][5][5]) {
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

void SW_conv2d_25(DATA_T I[96][25][25], DATA_T O[96][25][25], DATA_T W[96][96][3][3]) {
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

void SW_conv2d_26(DATA_T I[288][25][25], DATA_T O[64][25][25], DATA_T W[64][288][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<288; k++) {
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

void SW_batch_normalization_20(DATA_T I[64][25][25], DATA_T O[64][25][25], DATA_T W[3][64]) {
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
void SW_batch_normalization_22(DATA_T I[64][25][25], DATA_T O[64][25][25], DATA_T W[3][64]) {
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
void SW_batch_normalization_25(DATA_T I[96][25][25], DATA_T O[96][25][25], DATA_T W[3][96]) {
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
void SW_batch_normalization_26(DATA_T I[64][25][25], DATA_T O[64][25][25], DATA_T W[3][64]) {
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
void SW_activation_69(DATA_T I[64][25][25], DATA_T O[64][25][25]) {
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

void SW_activation_71(DATA_T I[64][25][25], DATA_T O[64][25][25]) {
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

void SW_activation_74(DATA_T I[96][25][25], DATA_T O[96][25][25]) {
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

void SW_activation_75(DATA_T I[64][25][25], DATA_T O[64][25][25]) {
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

void SW_mixed2(DATA_T I1[64][25][25], DATA_T I2[64][25][25], DATA_T I3[96][25][25], DATA_T I4[64][25][25], DATA_T O[288][25][25]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 25; x++) {
		for(y = 0; y < 25; y++) {
			ch=0;
			for(k = ch; k < ch+64; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=64;
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
void SW_conv2d_28(DATA_T I[288][25][25], DATA_T O[64][25][25], DATA_T W[64][288][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(25 - 1) - 25 + 1)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<25; x++) {
			for (y = 0; y<25; y++) {
				ofm = 0;
				for (k = 0; k<288; k++) {
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

void SW_batch_normalization_28(DATA_T I[64][25][25], DATA_T O[64][25][25], DATA_T W[3][64]) {
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
void SW_activation_77(DATA_T I[64][25][25], DATA_T O[64][25][25]) {
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

void SW_conv2d_29(DATA_T I[64][25][25], DATA_T O[96][25][25], DATA_T W[96][64][3][3]) {
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

void SW_batch_normalization_29(DATA_T I[96][25][25], DATA_T O[96][25][25], DATA_T W[3][96]) {
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
void SW_activation_78(DATA_T I[96][25][25], DATA_T O[96][25][25]) {
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

void SW_conv2d_27(DATA_T I[288][25][25], DATA_T O[384][12][12], DATA_T W[384][288][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<384; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<288; k++) {
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

void SW_conv2d_30(DATA_T I[96][25][25], DATA_T O[96][12][12], DATA_T W[96][96][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<96; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<96; k++) {
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

void SW_batch_normalization_27(DATA_T I[384][12][12], DATA_T O[384][12][12], DATA_T W[3][384]) {
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
void SW_batch_normalization_30(DATA_T I[96][12][12], DATA_T O[96][12][12], DATA_T W[3][96]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 96; m++){
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

void SW_activation_79(DATA_T I[96][12][12], DATA_T O[96][12][12]) {
	int m, x, y, i, j, k;
	for (m = 0; m<96; m++) {
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

void SW_max_pooling2d_4(DATA_T I[288][25][25], DATA_T O[288][12][12])
{
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<288; m++) {
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
void SW_mixed3(DATA_T I1[384][12][12], DATA_T I2[96][12][12], DATA_T I3[288][12][12], DATA_T O[768][12][12]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 12; x++) {
		for(y = 0; y < 12; y++) {
			ch=0;
			for(k = ch; k < ch+384; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=384;
			for(k = ch; k < ch+96; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
			ch+=96;
			for(k = ch; k < ch+288; k++){
				O[k][x][y] = I3[k-ch][x][y];
			}
		}
	}

}
void SW_conv2d_35(DATA_T I[768][12][12], DATA_T O[128][12][12], DATA_T W[128][768][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<768; k++) {
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

void SW_batch_normalization_35(DATA_T I[128][12][12], DATA_T O[128][12][12], DATA_T W[3][128]) {
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
void SW_activation_84(DATA_T I[128][12][12], DATA_T O[128][12][12]) {
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

void SW_conv2d_36(DATA_T I[128][12][12], DATA_T O[128][12][12], DATA_T W[128][128][7][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 7)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
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

void SW_batch_normalization_36(DATA_T I[128][12][12], DATA_T O[128][12][12], DATA_T W[3][128]) {
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
void SW_activation_85(DATA_T I[128][12][12], DATA_T O[128][12][12]) {
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

void SW_conv2d_32(DATA_T I[768][12][12], DATA_T O[128][12][12], DATA_T W[128][768][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<768; k++) {
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

void SW_conv2d_37(DATA_T I[128][12][12], DATA_T O[128][12][12], DATA_T W[128][128][1][7]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<128; m++) {
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

void SW_batch_normalization_32(DATA_T I[128][12][12], DATA_T O[128][12][12], DATA_T W[3][128]) {
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
void SW_batch_normalization_37(DATA_T I[128][12][12], DATA_T O[128][12][12], DATA_T W[3][128]) {
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
void SW_activation_81(DATA_T I[128][12][12], DATA_T O[128][12][12]) {
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

void SW_conv2d_33(DATA_T I[128][12][12], DATA_T O[128][12][12], DATA_T W[128][128][1][7]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<128; m++) {
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

void SW_conv2d_38(DATA_T I[128][12][12], DATA_T O[128][12][12], DATA_T W[128][128][7][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 7)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
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

void SW_batch_normalization_33(DATA_T I[128][12][12], DATA_T O[128][12][12], DATA_T W[3][128]) {
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
void SW_batch_normalization_38(DATA_T I[128][12][12], DATA_T O[128][12][12], DATA_T W[3][128]) {
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

void SW_activation_87(DATA_T I[128][12][12], DATA_T O[128][12][12]) {
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

void SW_average_pooling2d_4(DATA_T I[768][12][12], DATA_T O[768][12][12])
{
	int m, x, y, i, j;
	DATA_T sum;
	int p = (1*(12-1) - 12 + 3)/2;
    DATA_T div;
	for (m = 0; m<768; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
                sum = (DATA_T)0;
                if(x!=0 && x!=12-1 && y!=0 && y!=12-1)
                    div = 9;
                else if((x==0 || x==12-1) && (y==0 || y==12-1))
                    div = 4;
                else
                    div = 6;
				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {

						if (x + i < 12 + p && y + j < 12 + p && x + i -p >= 0 && y + j -p >= 0)
						    sum += I[m][x*1 + i -p][y*1 + j -p];
					}
				}
				O[m][x][y] = sum/div;
			}
		}
	}
}
void SW_conv2d_31(DATA_T I[768][12][12], DATA_T O[192][12][12], DATA_T W[192][768][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<768; k++) {
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

void SW_conv2d_34(DATA_T I[128][12][12], DATA_T O[192][12][12], DATA_T W[192][128][7][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 7)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
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

void SW_conv2d_39(DATA_T I[128][12][12], DATA_T O[192][12][12], DATA_T W[192][128][1][7]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
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

void SW_conv2d_40(DATA_T I[768][12][12], DATA_T O[192][12][12], DATA_T W[192][768][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<768; k++) {
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

void SW_batch_normalization_31(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
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
void SW_batch_normalization_34(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
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
void SW_batch_normalization_39(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
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
void SW_batch_normalization_40(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
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

void SW_activation_83(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
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

void SW_mixed4(DATA_T I1[192][12][12], DATA_T I2[192][12][12], DATA_T I3[192][12][12], DATA_T I4[192][12][12], DATA_T O[768][12][12]) {
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
			ch+=192;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I3[k-ch][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I4[k-ch][x][y];
			}
		}
	}

}
void SW_conv2d_45(DATA_T I[768][12][12], DATA_T O[160][12][12], DATA_T W[160][768][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<768; k++) {
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

void SW_batch_normalization_45(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[3][160]) {
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
void SW_activation_94(DATA_T I[160][12][12], DATA_T O[160][12][12]) {
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

void SW_conv2d_46(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[160][160][7][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 7)/2;
	for (m = 0; m<160; m++) {
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

void SW_batch_normalization_46(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[3][160]) {
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

void SW_conv2d_42(DATA_T I[768][12][12], DATA_T O[160][12][12], DATA_T W[160][768][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<768; k++) {
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

void SW_conv2d_47(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[160][160][1][7]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<160; k++) {
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

void SW_batch_normalization_42(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[3][160]) {
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
void SW_batch_normalization_47(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[3][160]) {
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

void SW_activation_96(DATA_T I[160][12][12], DATA_T O[160][12][12]) {
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

void SW_conv2d_43(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[160][160][1][7]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<160; k++) {
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

void SW_conv2d_48(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[160][160][7][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 7)/2;
	for (m = 0; m<160; m++) {
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

void SW_batch_normalization_43(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[3][160]) {
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
void SW_batch_normalization_48(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[3][160]) {
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
void SW_activation_92(DATA_T I[160][12][12], DATA_T O[160][12][12]) {
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

void SW_activation_97(DATA_T I[160][12][12], DATA_T O[160][12][12]) {
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

void SW_average_pooling2d_5(DATA_T I[768][12][12], DATA_T O[768][12][12])
{
	int m, x, y, i, j;
	DATA_T sum;
	int p = (1*(12-1) - 12 + 3)/2;
    DATA_T div;
	for (m = 0; m<768; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
                sum = (DATA_T)0;
                if(x!=0 && x!=12-1 && y!=0 && y!=12-1)
                    div = 9;
                else if((x==0 || x==12-1) && (y==0 || y==12-1))
                    div = 4;
                else
                    div = 6;
				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {

						if (x + i < 12 + p && y + j < 12 + p && x + i -p >= 0 && y + j -p >= 0)
						    sum += I[m][x*1 + i -p][y*1 + j -p];
					}
				}
				O[m][x][y] = sum/div;
			}
		}
	}
}
void SW_conv2d_41(DATA_T I[768][12][12], DATA_T O[192][12][12], DATA_T W[192][768][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<768; k++) {
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

void SW_conv2d_44(DATA_T I[160][12][12], DATA_T O[192][12][12], DATA_T W[192][160][7][1]) {
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

void SW_conv2d_49(DATA_T I[160][12][12], DATA_T O[192][12][12], DATA_T W[192][160][1][7]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<160; k++) {
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

void SW_conv2d_50(DATA_T I[768][12][12], DATA_T O[192][12][12], DATA_T W[192][768][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<768; k++) {
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

void SW_batch_normalization_41(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
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
void SW_batch_normalization_44(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
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
void SW_batch_normalization_49(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
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
void SW_batch_normalization_50(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
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
void SW_activation_90(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
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

void SW_activation_98(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
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

void SW_activation_99(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
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

void SW_mixed5(DATA_T I1[192][12][12], DATA_T I2[192][12][12], DATA_T I3[192][12][12], DATA_T I4[192][12][12], DATA_T O[768][12][12]) {
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
			ch+=192;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I3[k-ch][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I4[k-ch][x][y];
			}
		}
	}

}
void SW_conv2d_55(DATA_T I[768][12][12], DATA_T O[160][12][12], DATA_T W[160][768][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<768; k++) {
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

void SW_batch_normalization_55(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[3][160]) {
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
void SW_activation_104(DATA_T I[160][12][12], DATA_T O[160][12][12]) {
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

void SW_conv2d_56(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[160][160][7][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 7)/2;
	for (m = 0; m<160; m++) {
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

void SW_batch_normalization_56(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[3][160]) {
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
void SW_activation_105(DATA_T I[160][12][12], DATA_T O[160][12][12]) {
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

void SW_conv2d_52(DATA_T I[768][12][12], DATA_T O[160][12][12], DATA_T W[160][768][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<768; k++) {
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

void SW_conv2d_57(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[160][160][1][7]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<160; k++) {
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

void SW_batch_normalization_52(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[3][160]) {
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
void SW_batch_normalization_57(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[3][160]) {
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
void SW_activation_101(DATA_T I[160][12][12], DATA_T O[160][12][12]) {
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

void SW_activation_106(DATA_T I[160][12][12], DATA_T O[160][12][12]) {
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

void SW_conv2d_53(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[160][160][1][7]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<160; k++) {
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

void SW_conv2d_58(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[160][160][7][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 7)/2;
	for (m = 0; m<160; m++) {
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

void SW_batch_normalization_53(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[3][160]) {
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
void SW_batch_normalization_58(DATA_T I[160][12][12], DATA_T O[160][12][12], DATA_T W[3][160]) {
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
void SW_activation_102(DATA_T I[160][12][12], DATA_T O[160][12][12]) {
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

void SW_average_pooling2d_6(DATA_T I[768][12][12], DATA_T O[768][12][12])
{
	int m, x, y, i, j;
	DATA_T sum;
	int p = (1*(12-1) - 12 + 3)/2;
    DATA_T div;
	for (m = 0; m<768; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
                sum = (DATA_T)0;
                if(x!=0 && x!=12-1 && y!=0 && y!=12-1)
                    div = 9;
                else if((x==0 || x==12-1) && (y==0 || y==12-1))
                    div = 4;
                else
                    div = 6;
				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {

						if (x + i < 12 + p && y + j < 12 + p && x + i -p >= 0 && y + j -p >= 0)
						    sum += I[m][x*1 + i -p][y*1 + j -p];
					}
				}
				O[m][x][y] = sum/div;
			}
		}
	}
}
void SW_conv2d_51(DATA_T I[768][12][12], DATA_T O[192][12][12], DATA_T W[192][768][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<768; k++) {
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

void SW_conv2d_54(DATA_T I[160][12][12], DATA_T O[192][12][12], DATA_T W[192][160][7][1]) {
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

void SW_conv2d_59(DATA_T I[160][12][12], DATA_T O[192][12][12], DATA_T W[192][160][1][7]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<160; k++) {
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

void SW_conv2d_60(DATA_T I[768][12][12], DATA_T O[192][12][12], DATA_T W[192][768][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<768; k++) {
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

void SW_batch_normalization_51(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
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
void SW_batch_normalization_54(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
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
void SW_batch_normalization_59(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
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
void SW_batch_normalization_60(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
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

void SW_activation_103(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
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

void SW_mixed6(DATA_T I1[192][12][12], DATA_T I2[192][12][12], DATA_T I3[192][12][12], DATA_T I4[192][12][12], DATA_T O[768][12][12]) {
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
			ch+=192;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I3[k-ch][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I4[k-ch][x][y];
			}
		}
	}

}
void SW_conv2d_65(DATA_T I[768][12][12], DATA_T O[192][12][12], DATA_T W[192][768][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<768; k++) {
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

void SW_batch_normalization_65(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
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
void SW_activation_114(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
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

void SW_conv2d_66(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[192][192][7][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 7)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<192; k++) {
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

void SW_batch_normalization_66(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
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
void SW_activation_115(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
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

void SW_conv2d_62(DATA_T I[768][12][12], DATA_T O[192][12][12], DATA_T W[192][768][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<768; k++) {
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

void SW_conv2d_67(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[192][192][1][7]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<192; k++) {
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

void SW_batch_normalization_62(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
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
void SW_batch_normalization_67(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
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
void SW_activation_111(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
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

void SW_conv2d_63(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[192][192][1][7]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<192; k++) {
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

void SW_conv2d_68(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[192][192][7][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 7)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<192; k++) {
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

void SW_batch_normalization_63(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
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
void SW_batch_normalization_68(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
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

void SW_average_pooling2d_7(DATA_T I[768][12][12], DATA_T O[768][12][12])
{
	int m, x, y, i, j;
	DATA_T sum;
	int p = (1*(12-1) - 12 + 3)/2;
    DATA_T div;
	for (m = 0; m<768; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
                sum = (DATA_T)0;
                if(x!=0 && x!=12-1 && y!=0 && y!=12-1)
                    div = 9;
                else if((x==0 || x==12-1) && (y==0 || y==12-1))
                    div = 4;
                else
                    div = 6;
				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {

						if (x + i < 12 + p && y + j < 12 + p && x + i -p >= 0 && y + j -p >= 0)
						    sum += I[m][x*1 + i -p][y*1 + j -p];
					}
				}
				O[m][x][y] = sum/div;
			}
		}
	}
}
void SW_conv2d_61(DATA_T I[768][12][12], DATA_T O[192][12][12], DATA_T W[192][768][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<768; k++) {
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

void SW_conv2d_64(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[192][192][7][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 7)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<192; k++) {
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

void SW_conv2d_69(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[192][192][1][7]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<192; k++) {
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

void SW_conv2d_70(DATA_T I[768][12][12], DATA_T O[192][12][12], DATA_T W[192][768][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<768; k++) {
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

void SW_batch_normalization_61(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
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
void SW_batch_normalization_64(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
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
void SW_batch_normalization_69(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
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
void SW_batch_normalization_70(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
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
void SW_activation_110(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
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

void SW_activation_118(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
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

void SW_activation_119(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
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

void SW_mixed7(DATA_T I1[192][12][12], DATA_T I2[192][12][12], DATA_T I3[192][12][12], DATA_T I4[192][12][12], DATA_T O[768][12][12]) {
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
			ch+=192;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I3[k-ch][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I4[k-ch][x][y];
			}
		}
	}

}
void SW_conv2d_73(DATA_T I[768][12][12], DATA_T O[192][12][12], DATA_T W[192][768][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<768; k++) {
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

void SW_batch_normalization_73(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
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
void SW_activation_122(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
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

void SW_conv2d_74(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[192][192][1][7]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<192; k++) {
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

void SW_batch_normalization_74(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
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
void SW_activation_123(DATA_T I[192][12][12], DATA_T O[192][12][12]) {
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

void SW_conv2d_71(DATA_T I[768][12][12], DATA_T O[192][12][12], DATA_T W[192][768][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<768; k++) {
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

void SW_conv2d_75(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[192][192][7][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(12 - 1) - 12 + 7)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<12; x++) {
			for (y = 0; y<12; y++) {
				ofm = 0;
				for (k = 0; k<192; k++) {
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

void SW_batch_normalization_71(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
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
void SW_batch_normalization_75(DATA_T I[192][12][12], DATA_T O[192][12][12], DATA_T W[3][192]) {
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

void SW_conv2d_72(DATA_T I[192][12][12], DATA_T O[320][5][5], DATA_T W[320][192][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<320; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<192; k++) {
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

void SW_conv2d_76(DATA_T I[192][12][12], DATA_T O[192][5][5], DATA_T W[192][192][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<192; k++) {
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

void SW_batch_normalization_72(DATA_T I[320][5][5], DATA_T O[320][5][5], DATA_T W[3][320]) {
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
void SW_batch_normalization_76(DATA_T I[192][5][5], DATA_T O[192][5][5], DATA_T W[3][192]) {
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
void SW_activation_121(DATA_T I[320][5][5], DATA_T O[320][5][5]) {
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

void SW_activation_125(DATA_T I[192][5][5], DATA_T O[192][5][5]) {
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

void SW_max_pooling2d_5(DATA_T I[768][12][12], DATA_T O[768][5][5])
{
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<768; m++) {
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
void SW_mixed8(DATA_T I1[320][5][5], DATA_T I2[192][5][5], DATA_T I3[768][5][5], DATA_T O[1280][5][5]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 5; x++) {
		for(y = 0; y < 5; y++) {
			ch=0;
			for(k = ch; k < ch+320; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=320;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+768; k++){
				O[k][x][y] = I3[k-ch][x][y];
			}
		}
	}

}
void SW_conv2d_81(DATA_T I[1280][5][5], DATA_T O[448][5][5], DATA_T W[448][1280][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<448; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<1280; k++) {
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

void SW_batch_normalization_81(DATA_T I[448][5][5], DATA_T O[448][5][5], DATA_T W[3][448]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 448; m++){
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
void SW_activation_130(DATA_T I[448][5][5], DATA_T O[448][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<448; m++) {
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

void SW_conv2d_78(DATA_T I[1280][5][5], DATA_T O[384][5][5], DATA_T W[384][1280][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<384; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<1280; k++) {
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

void SW_conv2d_82(DATA_T I[448][5][5], DATA_T O[384][5][5], DATA_T W[384][448][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 3)/2;
	for (m = 0; m<384; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<448; k++) {
					for (i = 0; i<3; i++) {
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

void SW_batch_normalization_78(DATA_T I[384][5][5], DATA_T O[384][5][5], DATA_T W[3][384]) {
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
void SW_batch_normalization_82(DATA_T I[384][5][5], DATA_T O[384][5][5], DATA_T W[3][384]) {
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
void SW_activation_127(DATA_T I[384][5][5], DATA_T O[384][5][5]) {
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

void SW_activation_131(DATA_T I[384][5][5], DATA_T O[384][5][5]) {
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

void SW_conv2d_79(DATA_T I[384][5][5], DATA_T O[384][5][5], DATA_T W[384][384][1][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<384; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
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

void SW_conv2d_80(DATA_T I[384][5][5], DATA_T O[384][5][5], DATA_T W[384][384][3][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 3)/2;
	for (m = 0; m<384; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
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

void SW_conv2d_83(DATA_T I[384][5][5], DATA_T O[384][5][5], DATA_T W[384][384][1][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<384; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
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

void SW_conv2d_84(DATA_T I[384][5][5], DATA_T O[384][5][5], DATA_T W[384][384][3][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 3)/2;
	for (m = 0; m<384; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
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

void SW_average_pooling2d_8(DATA_T I[1280][5][5], DATA_T O[1280][5][5])
{
	int m, x, y, i, j;
	DATA_T sum;
	int p = (1*(5-1) - 5 + 3)/2;
    DATA_T div;
	for (m = 0; m<1280; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
                sum = (DATA_T)0;
                if(x!=0 && x!=5-1 && y!=0 && y!=5-1)
                    div = 9;
                else if((x==0 || x==5-1) && (y==0 || y==5-1))
                    div = 4;
                else
                    div = 6;
				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {

						if (x + i < 5 + p && y + j < 5 + p && x + i -p >= 0 && y + j -p >= 0)
						    sum += I[m][x*1 + i -p][y*1 + j -p];
					}
				}
				O[m][x][y] = sum/div;
			}
		}
	}
}
void SW_conv2d_77(DATA_T I[1280][5][5], DATA_T O[320][5][5], DATA_T W[320][1280][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<320; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<1280; k++) {
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

void SW_batch_normalization_79(DATA_T I[384][5][5], DATA_T O[384][5][5], DATA_T W[3][384]) {
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
void SW_batch_normalization_80(DATA_T I[384][5][5], DATA_T O[384][5][5], DATA_T W[3][384]) {
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
void SW_batch_normalization_83(DATA_T I[384][5][5], DATA_T O[384][5][5], DATA_T W[3][384]) {
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
void SW_batch_normalization_84(DATA_T I[384][5][5], DATA_T O[384][5][5], DATA_T W[3][384]) {
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
void SW_conv2d_85(DATA_T I[1280][5][5], DATA_T O[192][5][5], DATA_T W[192][1280][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<1280; k++) {
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

void SW_batch_normalization_77(DATA_T I[320][5][5], DATA_T O[320][5][5], DATA_T W[3][320]) {
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
void SW_activation_128(DATA_T I[384][5][5], DATA_T O[384][5][5]) {
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

void SW_activation_129(DATA_T I[384][5][5], DATA_T O[384][5][5]) {
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

void SW_activation_132(DATA_T I[384][5][5], DATA_T O[384][5][5]) {
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

void SW_activation_133(DATA_T I[384][5][5], DATA_T O[384][5][5]) {
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

void SW_batch_normalization_85(DATA_T I[192][5][5], DATA_T O[192][5][5], DATA_T W[3][192]) {
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
void SW_activation_126(DATA_T I[320][5][5], DATA_T O[320][5][5]) {
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

void SW_activation_134(DATA_T I[192][5][5], DATA_T O[192][5][5]) {
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

void SW_mixed9(DATA_T I1[320][5][5], DATA_T I2[768][5][5], DATA_T I3[768][5][5], DATA_T I4[192][5][5], DATA_T O[2048][5][5]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 5; x++) {
		for(y = 0; y < 5; y++) {
			ch=0;
			for(k = ch; k < ch+320; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=320;
			for(k = ch; k < ch+768; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
			ch+=768;
			for(k = ch; k < ch+768; k++){
				O[k][x][y] = I3[k-ch][x][y];
			}
			ch+=768;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I4[k-ch][x][y];
			}
		}
	}

}
void SW_conv2d_90(DATA_T I[2048][5][5], DATA_T O[448][5][5], DATA_T W[448][2048][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<448; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<2048; k++) {
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

void SW_batch_normalization_90(DATA_T I[448][5][5], DATA_T O[448][5][5], DATA_T W[3][448]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 448; m++){
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
void SW_activation_139(DATA_T I[448][5][5], DATA_T O[448][5][5]) {
	int m, x, y, i, j, k;
	for (m = 0; m<448; m++) {
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

void SW_conv2d_87(DATA_T I[2048][5][5], DATA_T O[384][5][5], DATA_T W[384][2048][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<384; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<2048; k++) {
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

void SW_conv2d_91(DATA_T I[448][5][5], DATA_T O[384][5][5], DATA_T W[384][448][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 3)/2;
	for (m = 0; m<384; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<448; k++) {
					for (i = 0; i<3; i++) {
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

void SW_batch_normalization_87(DATA_T I[384][5][5], DATA_T O[384][5][5], DATA_T W[3][384]) {
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
void SW_batch_normalization_91(DATA_T I[384][5][5], DATA_T O[384][5][5], DATA_T W[3][384]) {
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
void SW_activation_136(DATA_T I[384][5][5], DATA_T O[384][5][5]) {
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

void SW_activation_140(DATA_T I[384][5][5], DATA_T O[384][5][5]) {
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

void SW_conv2d_88(DATA_T I[384][5][5], DATA_T O[384][5][5], DATA_T W[384][384][1][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<384; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
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

void SW_conv2d_89(DATA_T I[384][5][5], DATA_T O[384][5][5], DATA_T W[384][384][3][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 3)/2;
	for (m = 0; m<384; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
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

void SW_conv2d_92(DATA_T I[384][5][5], DATA_T O[384][5][5], DATA_T W[384][384][1][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<384; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
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

void SW_conv2d_93(DATA_T I[384][5][5], DATA_T O[384][5][5], DATA_T W[384][384][3][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 3)/2;
	for (m = 0; m<384; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
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

void SW_average_pooling2d_9(DATA_T I[2048][5][5], DATA_T O[2048][5][5])
{
	int m, x, y, i, j;
	DATA_T sum;
	int p = (1*(5-1) - 5 + 3)/2;
    DATA_T div;
	for (m = 0; m<2048; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
                sum = (DATA_T)0;
                if(x!=0 && x!=5-1 && y!=0 && y!=5-1)
                    div = 9;
                else if((x==0 || x==5-1) && (y==0 || y==5-1))
                    div = 4;
                else
                    div = 6;
				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {

						if (x + i < 5 + p && y + j < 5 + p && x + i -p >= 0 && y + j -p >= 0)
						    sum += I[m][x*1 + i -p][y*1 + j -p];
					}
				}
				O[m][x][y] = sum/div;
			}
		}
	}
}
void SW_conv2d_86(DATA_T I[2048][5][5], DATA_T O[320][5][5], DATA_T W[320][2048][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<320; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<2048; k++) {
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

void SW_batch_normalization_88(DATA_T I[384][5][5], DATA_T O[384][5][5], DATA_T W[3][384]) {
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
void SW_batch_normalization_89(DATA_T I[384][5][5], DATA_T O[384][5][5], DATA_T W[3][384]) {
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
void SW_batch_normalization_92(DATA_T I[384][5][5], DATA_T O[384][5][5], DATA_T W[3][384]) {
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
void SW_batch_normalization_93(DATA_T I[384][5][5], DATA_T O[384][5][5], DATA_T W[3][384]) {
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
void SW_conv2d_94(DATA_T I[2048][5][5], DATA_T O[192][5][5], DATA_T W[192][2048][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(5 - 1) - 5 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<5; x++) {
			for (y = 0; y<5; y++) {
				ofm = 0;
				for (k = 0; k<2048; k++) {
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

void SW_batch_normalization_86(DATA_T I[320][5][5], DATA_T O[320][5][5], DATA_T W[3][320]) {
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
void SW_activation_137(DATA_T I[384][5][5], DATA_T O[384][5][5]) {
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

void SW_activation_138(DATA_T I[384][5][5], DATA_T O[384][5][5]) {
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

void SW_activation_141(DATA_T I[384][5][5], DATA_T O[384][5][5]) {
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

void SW_activation_142(DATA_T I[384][5][5], DATA_T O[384][5][5]) {
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

void SW_batch_normalization_94(DATA_T I[192][5][5], DATA_T O[192][5][5], DATA_T W[3][192]) {
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
void SW_activation_135(DATA_T I[320][5][5], DATA_T O[320][5][5]) {
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

void SW_activation_143(DATA_T I[192][5][5], DATA_T O[192][5][5]) {
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

void SW_mixed10(DATA_T I1[320][5][5], DATA_T I2[768][5][5], DATA_T I3[768][5][5], DATA_T I4[192][5][5], DATA_T O[2048][5][5]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 5; x++) {
		for(y = 0; y < 5; y++) {
			ch=0;
			for(k = ch; k < ch+320; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=320;
			for(k = ch; k < ch+768; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
			ch+=768;
			for(k = ch; k < ch+768; k++){
				O[k][x][y] = I3[k-ch][x][y];
			}
			ch+=768;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I4[k-ch][x][y];
			}
		}
	}

}
void SW_avg_pool(DATA_T I[2048][5][5], DATA_T O[2048]) {
	int m, x, y;
	double avg;
	int div = 5 * 5;
	for (m = 0; m < 2048; m++){
		avg = 0;
		for (x = 0; x < 5; x++) {
			for (y = 0; y < 5; y++) {
				avg += I[m][x][y];
			}
		}
		O[m] = avg/div;
	}
}
void SW_predictions(DATA_T I[2048], DATA_T O[1000], DATA_T W[1000][2048], DATA_T B[1000])
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
	static DATA_T W41[64][256][1][1];
	static DATA_T W42[3][64];
	static DATA_T W44[48][256][1][1];
	static DATA_T W45[96][64][3][3];
	static DATA_T W46[3][48];
	static DATA_T W47[3][96];
	static DATA_T W51[64][256][1][1];
	static DATA_T W52[64][48][5][5];
	static DATA_T W53[96][96][3][3];
	static DATA_T W54[64][256][1][1];
	static DATA_T W55[3][64];
	static DATA_T W56[3][64];
	static DATA_T W57[3][96];
	static DATA_T W58[3][64];
	static DATA_T W64[64][288][1][1];
	static DATA_T W65[3][64];
	static DATA_T W67[48][288][1][1];
	static DATA_T W68[96][64][3][3];
	static DATA_T W69[3][48];
	static DATA_T W70[3][96];
	static DATA_T W74[64][288][1][1];
	static DATA_T W75[64][48][5][5];
	static DATA_T W76[96][96][3][3];
	static DATA_T W77[64][288][1][1];
	static DATA_T W78[3][64];
	static DATA_T W79[3][64];
	static DATA_T W80[3][96];
	static DATA_T W81[3][64];
	static DATA_T W87[64][288][1][1];
	static DATA_T W88[3][64];
	static DATA_T W90[96][64][3][3];
	static DATA_T W91[3][96];
	static DATA_T W93[384][288][3][3];
	static DATA_T W94[96][96][3][3];
	static DATA_T W95[3][384];
	static DATA_T W96[3][96];
	static DATA_T W101[128][768][1][1];
	static DATA_T W102[3][128];
	static DATA_T W104[128][128][7][1];
	static DATA_T W105[3][128];
	static DATA_T W107[128][768][1][1];
	static DATA_T W108[128][128][1][7];
	static DATA_T W109[3][128];
	static DATA_T W110[3][128];
	static DATA_T W113[128][128][1][7];
	static DATA_T W114[128][128][7][1];
	static DATA_T W115[3][128];
	static DATA_T W116[3][128];
	static DATA_T W120[192][768][1][1];
	static DATA_T W121[192][128][7][1];
	static DATA_T W122[192][128][1][7];
	static DATA_T W123[192][768][1][1];
	static DATA_T W124[3][192];
	static DATA_T W125[3][192];
	static DATA_T W126[3][192];
	static DATA_T W127[3][192];
	static DATA_T W133[160][768][1][1];
	static DATA_T W134[3][160];
	static DATA_T W136[160][160][7][1];
	static DATA_T W137[3][160];
	static DATA_T W139[160][768][1][1];
	static DATA_T W140[160][160][1][7];
	static DATA_T W141[3][160];
	static DATA_T W142[3][160];
	static DATA_T W145[160][160][1][7];
	static DATA_T W146[160][160][7][1];
	static DATA_T W147[3][160];
	static DATA_T W148[3][160];
	static DATA_T W152[192][768][1][1];
	static DATA_T W153[192][160][7][1];
	static DATA_T W154[192][160][1][7];
	static DATA_T W155[192][768][1][1];
	static DATA_T W156[3][192];
	static DATA_T W157[3][192];
	static DATA_T W158[3][192];
	static DATA_T W159[3][192];
	static DATA_T W165[160][768][1][1];
	static DATA_T W166[3][160];
	static DATA_T W168[160][160][7][1];
	static DATA_T W169[3][160];
	static DATA_T W171[160][768][1][1];
	static DATA_T W172[160][160][1][7];
	static DATA_T W173[3][160];
	static DATA_T W174[3][160];
	static DATA_T W177[160][160][1][7];
	static DATA_T W178[160][160][7][1];
	static DATA_T W179[3][160];
	static DATA_T W180[3][160];
	static DATA_T W184[192][768][1][1];
	static DATA_T W185[192][160][7][1];
	static DATA_T W186[192][160][1][7];
	static DATA_T W187[192][768][1][1];
	static DATA_T W188[3][192];
	static DATA_T W189[3][192];
	static DATA_T W190[3][192];
	static DATA_T W191[3][192];
	static DATA_T W197[192][768][1][1];
	static DATA_T W198[3][192];
	static DATA_T W200[192][192][7][1];
	static DATA_T W201[3][192];
	static DATA_T W203[192][768][1][1];
	static DATA_T W204[192][192][1][7];
	static DATA_T W205[3][192];
	static DATA_T W206[3][192];
	static DATA_T W209[192][192][1][7];
	static DATA_T W210[192][192][7][1];
	static DATA_T W211[3][192];
	static DATA_T W212[3][192];
	static DATA_T W216[192][768][1][1];
	static DATA_T W217[192][192][7][1];
	static DATA_T W218[192][192][1][7];
	static DATA_T W219[192][768][1][1];
	static DATA_T W220[3][192];
	static DATA_T W221[3][192];
	static DATA_T W222[3][192];
	static DATA_T W223[3][192];
	static DATA_T W229[192][768][1][1];
	static DATA_T W230[3][192];
	static DATA_T W232[192][192][1][7];
	static DATA_T W233[3][192];
	static DATA_T W235[192][768][1][1];
	static DATA_T W236[192][192][7][1];
	static DATA_T W237[3][192];
	static DATA_T W238[3][192];
	static DATA_T W241[320][192][3][3];
	static DATA_T W242[192][192][3][3];
	static DATA_T W243[3][320];
	static DATA_T W244[3][192];
	static DATA_T W249[448][1280][1][1];
	static DATA_T W250[3][448];
	static DATA_T W252[384][1280][1][1];
	static DATA_T W253[384][448][3][3];
	static DATA_T W254[3][384];
	static DATA_T W255[3][384];
	static DATA_T W258[384][384][1][3];
	static DATA_T W259[384][384][3][1];
	static DATA_T W260[384][384][1][3];
	static DATA_T W261[384][384][3][1];
	static DATA_T W263[320][1280][1][1];
	static DATA_T W264[3][384];
	static DATA_T W265[3][384];
	static DATA_T W266[3][384];
	static DATA_T W267[3][384];
	static DATA_T W268[192][1280][1][1];
	static DATA_T W269[3][320];
	static DATA_T W274[3][192];
	static DATA_T W280[448][2048][1][1];
	static DATA_T W281[3][448];
	static DATA_T W283[384][2048][1][1];
	static DATA_T W284[384][448][3][3];
	static DATA_T W285[3][384];
	static DATA_T W286[3][384];
	static DATA_T W289[384][384][1][3];
	static DATA_T W290[384][384][3][1];
	static DATA_T W291[384][384][1][3];
	static DATA_T W292[384][384][3][1];
	static DATA_T W294[320][2048][1][1];
	static DATA_T W295[3][384];
	static DATA_T W296[3][384];
	static DATA_T W297[3][384];
	static DATA_T W298[3][384];
	static DATA_T W299[192][2048][1][1];
	static DATA_T W300[3][320];
	static DATA_T W305[3][192];
	static DATA_T B312[1000];
	static DATA_T W312[1000][2048];
	

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
	static DATA_T O41_SW[64][25][25];
	static DATA_T O42_SW[64][25][25];
	static DATA_T O43_SW[64][25][25];
	static DATA_T O44_SW[48][25][25];
	static DATA_T O45_SW[96][25][25];
	static DATA_T O46_SW[48][25][25];
	static DATA_T O47_SW[96][25][25];
	static DATA_T O48_SW[48][25][25];
	static DATA_T O49_SW[96][25][25];
	static DATA_T O50_SW[256][25][25];
	static DATA_T O51_SW[64][25][25];
	static DATA_T O52_SW[64][25][25];
	static DATA_T O53_SW[96][25][25];
	static DATA_T O54_SW[64][25][25];
	static DATA_T O55_SW[64][25][25];
	static DATA_T O56_SW[64][25][25];
	static DATA_T O57_SW[96][25][25];
	static DATA_T O58_SW[64][25][25];
	static DATA_T O59_SW[64][25][25];
	static DATA_T O60_SW[64][25][25];
	static DATA_T O61_SW[96][25][25];
	static DATA_T O62_SW[64][25][25];
	static DATA_T O63_SW[288][25][25];
	static DATA_T O64_SW[64][25][25];
	static DATA_T O65_SW[64][25][25];
	static DATA_T O66_SW[64][25][25];
	static DATA_T O67_SW[48][25][25];
	static DATA_T O68_SW[96][25][25];
	static DATA_T O69_SW[48][25][25];
	static DATA_T O70_SW[96][25][25];
	static DATA_T O71_SW[48][25][25];
	static DATA_T O72_SW[96][25][25];
	static DATA_T O73_SW[288][25][25];
	static DATA_T O74_SW[64][25][25];
	static DATA_T O75_SW[64][25][25];
	static DATA_T O76_SW[96][25][25];
	static DATA_T O77_SW[64][25][25];
	static DATA_T O78_SW[64][25][25];
	static DATA_T O79_SW[64][25][25];
	static DATA_T O80_SW[96][25][25];
	static DATA_T O81_SW[64][25][25];
	static DATA_T O82_SW[64][25][25];
	static DATA_T O83_SW[64][25][25];
	static DATA_T O84_SW[96][25][25];
	static DATA_T O85_SW[64][25][25];
	static DATA_T O86_SW[288][25][25];
	static DATA_T O87_SW[64][25][25];
	static DATA_T O88_SW[64][25][25];
	static DATA_T O89_SW[64][25][25];
	static DATA_T O90_SW[96][25][25];
	static DATA_T O91_SW[96][25][25];
	static DATA_T O92_SW[96][25][25];
	static DATA_T O93_SW[384][12][12];
	static DATA_T O94_SW[96][12][12];
	static DATA_T O95_SW[384][12][12];
	static DATA_T O96_SW[96][12][12];
	static DATA_T O97_SW[384][12][12];
	static DATA_T O98_SW[96][12][12];
	static DATA_T O99_SW[288][12][12];
	static DATA_T O100_SW[768][12][12];
	static DATA_T O101_SW[128][12][12];
	static DATA_T O102_SW[128][12][12];
	static DATA_T O103_SW[128][12][12];
	static DATA_T O104_SW[128][12][12];
	static DATA_T O105_SW[128][12][12];
	static DATA_T O106_SW[128][12][12];
	static DATA_T O107_SW[128][12][12];
	static DATA_T O108_SW[128][12][12];
	static DATA_T O109_SW[128][12][12];
	static DATA_T O110_SW[128][12][12];
	static DATA_T O111_SW[128][12][12];
	static DATA_T O112_SW[128][12][12];
	static DATA_T O113_SW[128][12][12];
	static DATA_T O114_SW[128][12][12];
	static DATA_T O115_SW[128][12][12];
	static DATA_T O116_SW[128][12][12];
	static DATA_T O117_SW[128][12][12];
	static DATA_T O118_SW[128][12][12];
	static DATA_T O119_SW[768][12][12];
	static DATA_T O120_SW[192][12][12];
	static DATA_T O121_SW[192][12][12];
	static DATA_T O122_SW[192][12][12];
	static DATA_T O123_SW[192][12][12];
	static DATA_T O124_SW[192][12][12];
	static DATA_T O125_SW[192][12][12];
	static DATA_T O126_SW[192][12][12];
	static DATA_T O127_SW[192][12][12];
	static DATA_T O128_SW[192][12][12];
	static DATA_T O129_SW[192][12][12];
	static DATA_T O130_SW[192][12][12];
	static DATA_T O131_SW[192][12][12];
	static DATA_T O132_SW[768][12][12];
	static DATA_T O133_SW[160][12][12];
	static DATA_T O134_SW[160][12][12];
	static DATA_T O135_SW[160][12][12];
	static DATA_T O136_SW[160][12][12];
	static DATA_T O137_SW[160][12][12];
	static DATA_T O138_SW[160][12][12];
	static DATA_T O139_SW[160][12][12];
	static DATA_T O140_SW[160][12][12];
	static DATA_T O141_SW[160][12][12];
	static DATA_T O142_SW[160][12][12];
	static DATA_T O143_SW[160][12][12];
	static DATA_T O144_SW[160][12][12];
	static DATA_T O145_SW[160][12][12];
	static DATA_T O146_SW[160][12][12];
	static DATA_T O147_SW[160][12][12];
	static DATA_T O148_SW[160][12][12];
	static DATA_T O149_SW[160][12][12];
	static DATA_T O150_SW[160][12][12];
	static DATA_T O151_SW[768][12][12];
	static DATA_T O152_SW[192][12][12];
	static DATA_T O153_SW[192][12][12];
	static DATA_T O154_SW[192][12][12];
	static DATA_T O155_SW[192][12][12];
	static DATA_T O156_SW[192][12][12];
	static DATA_T O157_SW[192][12][12];
	static DATA_T O158_SW[192][12][12];
	static DATA_T O159_SW[192][12][12];
	static DATA_T O160_SW[192][12][12];
	static DATA_T O161_SW[192][12][12];
	static DATA_T O162_SW[192][12][12];
	static DATA_T O163_SW[192][12][12];
	static DATA_T O164_SW[768][12][12];
	static DATA_T O165_SW[160][12][12];
	static DATA_T O166_SW[160][12][12];
	static DATA_T O167_SW[160][12][12];
	static DATA_T O168_SW[160][12][12];
	static DATA_T O169_SW[160][12][12];
	static DATA_T O170_SW[160][12][12];
	static DATA_T O171_SW[160][12][12];
	static DATA_T O172_SW[160][12][12];
	static DATA_T O173_SW[160][12][12];
	static DATA_T O174_SW[160][12][12];
	static DATA_T O175_SW[160][12][12];
	static DATA_T O176_SW[160][12][12];
	static DATA_T O177_SW[160][12][12];
	static DATA_T O178_SW[160][12][12];
	static DATA_T O179_SW[160][12][12];
	static DATA_T O180_SW[160][12][12];
	static DATA_T O181_SW[160][12][12];
	static DATA_T O182_SW[160][12][12];
	static DATA_T O183_SW[768][12][12];
	static DATA_T O184_SW[192][12][12];
	static DATA_T O185_SW[192][12][12];
	static DATA_T O186_SW[192][12][12];
	static DATA_T O187_SW[192][12][12];
	static DATA_T O188_SW[192][12][12];
	static DATA_T O189_SW[192][12][12];
	static DATA_T O190_SW[192][12][12];
	static DATA_T O191_SW[192][12][12];
	static DATA_T O192_SW[192][12][12];
	static DATA_T O193_SW[192][12][12];
	static DATA_T O194_SW[192][12][12];
	static DATA_T O195_SW[192][12][12];
	static DATA_T O196_SW[768][12][12];
	static DATA_T O197_SW[192][12][12];
	static DATA_T O198_SW[192][12][12];
	static DATA_T O199_SW[192][12][12];
	static DATA_T O200_SW[192][12][12];
	static DATA_T O201_SW[192][12][12];
	static DATA_T O202_SW[192][12][12];
	static DATA_T O203_SW[192][12][12];
	static DATA_T O204_SW[192][12][12];
	static DATA_T O205_SW[192][12][12];
	static DATA_T O206_SW[192][12][12];
	static DATA_T O207_SW[192][12][12];
	static DATA_T O208_SW[192][12][12];
	static DATA_T O209_SW[192][12][12];
	static DATA_T O210_SW[192][12][12];
	static DATA_T O211_SW[192][12][12];
	static DATA_T O212_SW[192][12][12];
	static DATA_T O213_SW[192][12][12];
	static DATA_T O214_SW[192][12][12];
	static DATA_T O215_SW[768][12][12];
	static DATA_T O216_SW[192][12][12];
	static DATA_T O217_SW[192][12][12];
	static DATA_T O218_SW[192][12][12];
	static DATA_T O219_SW[192][12][12];
	static DATA_T O220_SW[192][12][12];
	static DATA_T O221_SW[192][12][12];
	static DATA_T O222_SW[192][12][12];
	static DATA_T O223_SW[192][12][12];
	static DATA_T O224_SW[192][12][12];
	static DATA_T O225_SW[192][12][12];
	static DATA_T O226_SW[192][12][12];
	static DATA_T O227_SW[192][12][12];
	static DATA_T O228_SW[768][12][12];
	static DATA_T O229_SW[192][12][12];
	static DATA_T O230_SW[192][12][12];
	static DATA_T O231_SW[192][12][12];
	static DATA_T O232_SW[192][12][12];
	static DATA_T O233_SW[192][12][12];
	static DATA_T O234_SW[192][12][12];
	static DATA_T O235_SW[192][12][12];
	static DATA_T O236_SW[192][12][12];
	static DATA_T O237_SW[192][12][12];
	static DATA_T O238_SW[192][12][12];
	static DATA_T O239_SW[192][12][12];
	static DATA_T O240_SW[192][12][12];
	static DATA_T O241_SW[320][5][5];
	static DATA_T O242_SW[192][5][5];
	static DATA_T O243_SW[320][5][5];
	static DATA_T O244_SW[192][5][5];
	static DATA_T O245_SW[320][5][5];
	static DATA_T O246_SW[192][5][5];
	static DATA_T O247_SW[768][5][5];
	static DATA_T O248_SW[1280][5][5];
	static DATA_T O249_SW[448][5][5];
	static DATA_T O250_SW[448][5][5];
	static DATA_T O251_SW[448][5][5];
	static DATA_T O252_SW[384][5][5];
	static DATA_T O253_SW[384][5][5];
	static DATA_T O254_SW[384][5][5];
	static DATA_T O255_SW[384][5][5];
	static DATA_T O256_SW[384][5][5];
	static DATA_T O257_SW[384][5][5];
	static DATA_T O258_SW[384][5][5];
	static DATA_T O259_SW[384][5][5];
	static DATA_T O260_SW[384][5][5];
	static DATA_T O261_SW[384][5][5];
	static DATA_T O262_SW[1280][5][5];
	static DATA_T O263_SW[320][5][5];
	static DATA_T O264_SW[384][5][5];
	static DATA_T O265_SW[384][5][5];
	static DATA_T O266_SW[384][5][5];
	static DATA_T O267_SW[384][5][5];
	static DATA_T O268_SW[192][5][5];
	static DATA_T O269_SW[320][5][5];
	static DATA_T O270_SW[384][5][5];
	static DATA_T O271_SW[384][5][5];
	static DATA_T O272_SW[384][5][5];
	static DATA_T O273_SW[384][5][5];
	static DATA_T O274_SW[192][5][5];
	static DATA_T O275_SW[320][5][5];
	static DATA_T O276_SW[768][5][5];
	static DATA_T O277_SW[768][5][5];
	static DATA_T O278_SW[192][5][5];
	static DATA_T O279_SW[2048][5][5];
	static DATA_T O280_SW[448][5][5];
	static DATA_T O281_SW[448][5][5];
	static DATA_T O282_SW[448][5][5];
	static DATA_T O283_SW[384][5][5];
	static DATA_T O284_SW[384][5][5];
	static DATA_T O285_SW[384][5][5];
	static DATA_T O286_SW[384][5][5];
	static DATA_T O287_SW[384][5][5];
	static DATA_T O288_SW[384][5][5];
	static DATA_T O289_SW[384][5][5];
	static DATA_T O290_SW[384][5][5];
	static DATA_T O291_SW[384][5][5];
	static DATA_T O292_SW[384][5][5];
	static DATA_T O293_SW[2048][5][5];
	static DATA_T O294_SW[320][5][5];
	static DATA_T O295_SW[384][5][5];
	static DATA_T O296_SW[384][5][5];
	static DATA_T O297_SW[384][5][5];
	static DATA_T O298_SW[384][5][5];
	static DATA_T O299_SW[192][5][5];
	static DATA_T O300_SW[320][5][5];
	static DATA_T O301_SW[384][5][5];
	static DATA_T O302_SW[384][5][5];
	static DATA_T O303_SW[384][5][5];
	static DATA_T O304_SW[384][5][5];
	static DATA_T O305_SW[192][5][5];
	static DATA_T O306_SW[320][5][5];
	static DATA_T O307_SW[768][5][5];
	static DATA_T O308_SW[768][5][5];
	static DATA_T O309_SW[192][5][5];
	static DATA_T O310_SW[2048][5][5];
	static DATA_T O311_SW[2048];
	static DATA_T O312_SW[1000];
	

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
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W41[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B41[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W42[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 48 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W44[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 48 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B44[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 96 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W45[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 96 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B45[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 48 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W46[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 96 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W47[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W51[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B51[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  5 ; m++) {
	for (k = 0; k < 5 ; k++) {
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

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 96 ; i++) {
			for (j = 0; j < 96 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W53[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 96 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B53[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W54[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B54[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W55[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W56[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 96 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W57[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W58[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 288 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W64[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B64[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W65[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 288 ; i++) {
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

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 96 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W68[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 96 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B68[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 48 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W69[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 96 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W70[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 288 ; i++) {
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

	for (m = 0; m <  5 ; m++) {
	for (k = 0; k < 5 ; k++) {
		for (i = 0; i < 48 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W75[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B75[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 96 ; i++) {
			for (j = 0; j < 96 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W76[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 96 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B76[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 288 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W77[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B77[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W78[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W79[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 96 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W80[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W81[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 288 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W87[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B87[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W88[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 96 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W90[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 96 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B90[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 96 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W91[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 288 ; i++) {
			for (j = 0; j < 384 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W93[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 384 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B93[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 96 ; i++) {
			for (j = 0; j < 96 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W94[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 96 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B94[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 384 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W95[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 96 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W96[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 768 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W101[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B101[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W102[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W104[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B104[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W105[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 768 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W107[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B107[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W108[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B108[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W109[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W110[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W113[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B113[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W114[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B114[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W115[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W116[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 768 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W120[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B120[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W121[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B121[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W122[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B122[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 768 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W123[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B123[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W124[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W125[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W126[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W127[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 768 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W133[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B133[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W134[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W136[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B136[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W137[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 768 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W139[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B139[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W140[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B140[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W141[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W142[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W145[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B145[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W146[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B146[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W147[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W148[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 768 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W152[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B152[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W153[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B153[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W154[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B154[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 768 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W155[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B155[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W156[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W157[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W158[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W159[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 768 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W165[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B165[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W166[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W168[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B168[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W169[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 768 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W171[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B171[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W172[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B172[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W173[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W174[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W177[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B177[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W178[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B178[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W179[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W180[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 768 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W184[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B184[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W185[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B185[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W186[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B186[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 768 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W187[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B187[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W188[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W189[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W190[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W191[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 768 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W197[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B197[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W198[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 192 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W200[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B200[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W201[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 768 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W203[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B203[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 192 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W204[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B204[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W205[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W206[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 192 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W209[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B209[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 192 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W210[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B210[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W211[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W212[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 768 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W216[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B216[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 192 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W217[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B217[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 192 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W218[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B218[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 768 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W219[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B219[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W220[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W221[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W222[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W223[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 768 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W229[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B229[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W230[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 7 ; k++) {
		for (i = 0; i < 192 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W232[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B232[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W233[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 768 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W235[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B235[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  7 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 192 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W236[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B236[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W237[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W238[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 192 ; i++) {
			for (j = 0; j < 320 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W241[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 320 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B241[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 192 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W242[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B242[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 320 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W243[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W244[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1280 ; i++) {
			for (j = 0; j < 448 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W249[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 448 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B249[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 448 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W250[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1280 ; i++) {
			for (j = 0; j < 384 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W252[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 384 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B252[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 448 ; i++) {
			for (j = 0; j < 384 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W253[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 384 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B253[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 384 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W254[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 384 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W255[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 384 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W258[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 384 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B258[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 384 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W259[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 384 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B259[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 384 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W260[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 384 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B260[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 384 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W261[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 384 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B261[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1280 ; i++) {
			for (j = 0; j < 320 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W263[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 320 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B263[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 384 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W264[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 384 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W265[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 384 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W266[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 384 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W267[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1280 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W268[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B268[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 320 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W269[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W274[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 2048 ; i++) {
			for (j = 0; j < 448 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W280[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 448 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B280[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 448 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W281[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 2048 ; i++) {
			for (j = 0; j < 384 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W283[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 384 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B283[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 448 ; i++) {
			for (j = 0; j < 384 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W284[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 384 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B284[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 384 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W285[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 384 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W286[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 384 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W289[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 384 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B289[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 384 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W290[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 384 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B290[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 384 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W291[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 384 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B291[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 384 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W292[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 384 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B292[m] = (DATA_T) trash;
}
*/

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 2048 ; i++) {
			for (j = 0; j < 320 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W294[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 320 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B294[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 384 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W295[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 384 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W296[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 384 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W297[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 384 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W298[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 2048 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W299[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B299[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 3; x++) {
    for (y = 0; y < 320 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W300[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 3; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W305[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  2048 ; m++) {
	for (k = 0; k < 1000 ; k++) {
		fread(&trash, sizeof(int), 1, w_stream);
        W312[k][m] = (DATA_T) trash;
	}
}


for (m = 0; m < 1000 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B312[m] = (DATA_T) trash;
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
	printf("[C_verifier.cpp]Calculate Conv2D41\n\n");
	SW_conv2d_16(O40_SW,O41_SW,W41);
	printf("[C_verifier.cpp]Calculate BatchNormalization42\n\n");
	SW_batch_normalization_16(O41_SW,O42_SW, W42);
	printf("[C_verifier.cpp]Calculate Activation(Relu)43\n\n");
	SW_activation_65(O42_SW,O43_SW);
	printf("[C_verifier.cpp]Calculate Conv2D44\n\n");
	SW_conv2d_14(O40_SW,O44_SW,W44);
	printf("[C_verifier.cpp]Calculate Conv2D45\n\n");
	SW_conv2d_17(O43_SW,O45_SW,W45);
	printf("[C_verifier.cpp]Calculate BatchNormalization46\n\n");
	SW_batch_normalization_14(O44_SW,O46_SW, W46);
	printf("[C_verifier.cpp]Calculate BatchNormalization47\n\n");
	SW_batch_normalization_17(O45_SW,O47_SW, W47);
	printf("[C_verifier.cpp]Calculate Activation(Relu)48\n\n");
	SW_activation_63(O46_SW,O48_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)49\n\n");
	SW_activation_66(O47_SW,O49_SW);
	printf("[C_verifier.cpp]Calculate AveragePooling2D50\n\n");
	SW_average_pooling2d_2(O40_SW,O50_SW);
	printf("[C_verifier.cpp]Calculate Conv2D51\n\n");
	SW_conv2d_13(O40_SW,O51_SW,W51);
	printf("[C_verifier.cpp]Calculate Conv2D52\n\n");
	SW_conv2d_15(O48_SW,O52_SW,W52);
	printf("[C_verifier.cpp]Calculate Conv2D53\n\n");
	SW_conv2d_18(O49_SW,O53_SW,W53);
	printf("[C_verifier.cpp]Calculate Conv2D54\n\n");
	SW_conv2d_19(O50_SW,O54_SW,W54);
	printf("[C_verifier.cpp]Calculate BatchNormalization55\n\n");
	SW_batch_normalization_13(O51_SW,O55_SW, W55);
	printf("[C_verifier.cpp]Calculate BatchNormalization56\n\n");
	SW_batch_normalization_15(O52_SW,O56_SW, W56);
	printf("[C_verifier.cpp]Calculate BatchNormalization57\n\n");
	SW_batch_normalization_18(O53_SW,O57_SW, W57);
	printf("[C_verifier.cpp]Calculate BatchNormalization58\n\n");
	SW_batch_normalization_19(O54_SW,O58_SW, W58);
	printf("[C_verifier.cpp]Calculate Activation(Relu)59\n\n");
	SW_activation_62(O55_SW,O59_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)60\n\n");
	SW_activation_64(O56_SW,O60_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)61\n\n");
	SW_activation_67(O57_SW,O61_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)62\n\n");
	SW_activation_68(O58_SW,O62_SW);
	printf("[C_verifier.cpp]Calculate Concatenate63\n\n");
	SW_mixed1(O59_SW, O60_SW, O61_SW, O62_SW,O63_SW);
	printf("[C_verifier.cpp]Calculate Conv2D64\n\n");
	SW_conv2d_23(O63_SW,O64_SW,W64);
	printf("[C_verifier.cpp]Calculate BatchNormalization65\n\n");
	SW_batch_normalization_23(O64_SW,O65_SW, W65);
	printf("[C_verifier.cpp]Calculate Activation(Relu)66\n\n");
	SW_activation_72(O65_SW,O66_SW);
	printf("[C_verifier.cpp]Calculate Conv2D67\n\n");
	SW_conv2d_21(O63_SW,O67_SW,W67);
	printf("[C_verifier.cpp]Calculate Conv2D68\n\n");
	SW_conv2d_24(O66_SW,O68_SW,W68);
	printf("[C_verifier.cpp]Calculate BatchNormalization69\n\n");
	SW_batch_normalization_21(O67_SW,O69_SW, W69);
	printf("[C_verifier.cpp]Calculate BatchNormalization70\n\n");
	SW_batch_normalization_24(O68_SW,O70_SW, W70);
	printf("[C_verifier.cpp]Calculate Activation(Relu)71\n\n");
	SW_activation_70(O69_SW,O71_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)72\n\n");
	SW_activation_73(O70_SW,O72_SW);
	printf("[C_verifier.cpp]Calculate AveragePooling2D73\n\n");
	SW_average_pooling2d_3(O63_SW,O73_SW);
	printf("[C_verifier.cpp]Calculate Conv2D74\n\n");
	SW_conv2d_20(O63_SW,O74_SW,W74);
	printf("[C_verifier.cpp]Calculate Conv2D75\n\n");
	SW_conv2d_22(O71_SW,O75_SW,W75);
	printf("[C_verifier.cpp]Calculate Conv2D76\n\n");
	SW_conv2d_25(O72_SW,O76_SW,W76);
	printf("[C_verifier.cpp]Calculate Conv2D77\n\n");
	SW_conv2d_26(O73_SW,O77_SW,W77);
	printf("[C_verifier.cpp]Calculate BatchNormalization78\n\n");
	SW_batch_normalization_20(O74_SW,O78_SW, W78);
	printf("[C_verifier.cpp]Calculate BatchNormalization79\n\n");
	SW_batch_normalization_22(O75_SW,O79_SW, W79);
	printf("[C_verifier.cpp]Calculate BatchNormalization80\n\n");
	SW_batch_normalization_25(O76_SW,O80_SW, W80);
	printf("[C_verifier.cpp]Calculate BatchNormalization81\n\n");
	SW_batch_normalization_26(O77_SW,O81_SW, W81);
	printf("[C_verifier.cpp]Calculate Activation(Relu)82\n\n");
	SW_activation_69(O78_SW,O82_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)83\n\n");
	SW_activation_71(O79_SW,O83_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)84\n\n");
	SW_activation_74(O80_SW,O84_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)85\n\n");
	SW_activation_75(O81_SW,O85_SW);
	printf("[C_verifier.cpp]Calculate Concatenate86\n\n");
	SW_mixed2(O82_SW, O83_SW, O84_SW, O85_SW,O86_SW);
	printf("[C_verifier.cpp]Calculate Conv2D87\n\n");
	SW_conv2d_28(O86_SW,O87_SW,W87);
	printf("[C_verifier.cpp]Calculate BatchNormalization88\n\n");
	SW_batch_normalization_28(O87_SW,O88_SW, W88);
	printf("[C_verifier.cpp]Calculate Activation(Relu)89\n\n");
	SW_activation_77(O88_SW,O89_SW);
	printf("[C_verifier.cpp]Calculate Conv2D90\n\n");
	SW_conv2d_29(O89_SW,O90_SW,W90);
	printf("[C_verifier.cpp]Calculate BatchNormalization91\n\n");
	SW_batch_normalization_29(O90_SW,O91_SW, W91);
	printf("[C_verifier.cpp]Calculate Activation(Relu)92\n\n");
	SW_activation_78(O91_SW,O92_SW);
	printf("[C_verifier.cpp]Calculate Conv2D93\n\n");
	SW_conv2d_27(O86_SW,O93_SW,W93);
	printf("[C_verifier.cpp]Calculate Conv2D94\n\n");
	SW_conv2d_30(O92_SW,O94_SW,W94);
	printf("[C_verifier.cpp]Calculate BatchNormalization95\n\n");
	SW_batch_normalization_27(O93_SW,O95_SW, W95);
	printf("[C_verifier.cpp]Calculate BatchNormalization96\n\n");
	SW_batch_normalization_30(O94_SW,O96_SW, W96);
	printf("[C_verifier.cpp]Calculate Activation(Relu)97\n\n");
	SW_activation_76(O95_SW,O97_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)98\n\n");
	SW_activation_79(O96_SW,O98_SW);
	printf("[C_verifier.cpp]Calculate MaxPooling2D99\n\n");
	SW_max_pooling2d_4(O86_SW,O99_SW);
	printf("[C_verifier.cpp]Calculate Concatenate100\n\n");
	SW_mixed3(O97_SW, O98_SW, O99_SW,O100_SW);
	printf("[C_verifier.cpp]Calculate Conv2D101\n\n");
	SW_conv2d_35(O100_SW,O101_SW,W101);
	printf("[C_verifier.cpp]Calculate BatchNormalization102\n\n");
	SW_batch_normalization_35(O101_SW,O102_SW, W102);
	printf("[C_verifier.cpp]Calculate Activation(Relu)103\n\n");
	SW_activation_84(O102_SW,O103_SW);
	printf("[C_verifier.cpp]Calculate Conv2D104\n\n");
	SW_conv2d_36(O103_SW,O104_SW,W104);
	printf("[C_verifier.cpp]Calculate BatchNormalization105\n\n");
	SW_batch_normalization_36(O104_SW,O105_SW, W105);
	printf("[C_verifier.cpp]Calculate Activation(Relu)106\n\n");
	SW_activation_85(O105_SW,O106_SW);
	printf("[C_verifier.cpp]Calculate Conv2D107\n\n");
	SW_conv2d_32(O100_SW,O107_SW,W107);
	printf("[C_verifier.cpp]Calculate Conv2D108\n\n");
	SW_conv2d_37(O106_SW,O108_SW,W108);
	printf("[C_verifier.cpp]Calculate BatchNormalization109\n\n");
	SW_batch_normalization_32(O107_SW,O109_SW, W109);
	printf("[C_verifier.cpp]Calculate BatchNormalization110\n\n");
	SW_batch_normalization_37(O108_SW,O110_SW, W110);
	printf("[C_verifier.cpp]Calculate Activation(Relu)111\n\n");
	SW_activation_81(O109_SW,O111_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)112\n\n");
	SW_activation_86(O110_SW,O112_SW);
	printf("[C_verifier.cpp]Calculate Conv2D113\n\n");
	SW_conv2d_33(O111_SW,O113_SW,W113);
	printf("[C_verifier.cpp]Calculate Conv2D114\n\n");
	SW_conv2d_38(O112_SW,O114_SW,W114);
	printf("[C_verifier.cpp]Calculate BatchNormalization115\n\n");
	SW_batch_normalization_33(O113_SW,O115_SW, W115);
	printf("[C_verifier.cpp]Calculate BatchNormalization116\n\n");
	SW_batch_normalization_38(O114_SW,O116_SW, W116);
	printf("[C_verifier.cpp]Calculate Activation(Relu)117\n\n");
	SW_activation_82(O115_SW,O117_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)118\n\n");
	SW_activation_87(O116_SW,O118_SW);
	printf("[C_verifier.cpp]Calculate AveragePooling2D119\n\n");
	SW_average_pooling2d_4(O100_SW,O119_SW);
	printf("[C_verifier.cpp]Calculate Conv2D120\n\n");
	SW_conv2d_31(O100_SW,O120_SW,W120);
	printf("[C_verifier.cpp]Calculate Conv2D121\n\n");
	SW_conv2d_34(O117_SW,O121_SW,W121);
	printf("[C_verifier.cpp]Calculate Conv2D122\n\n");
	SW_conv2d_39(O118_SW,O122_SW,W122);
	printf("[C_verifier.cpp]Calculate Conv2D123\n\n");
	SW_conv2d_40(O119_SW,O123_SW,W123);
	printf("[C_verifier.cpp]Calculate BatchNormalization124\n\n");
	SW_batch_normalization_31(O120_SW,O124_SW, W124);
	printf("[C_verifier.cpp]Calculate BatchNormalization125\n\n");
	SW_batch_normalization_34(O121_SW,O125_SW, W125);
	printf("[C_verifier.cpp]Calculate BatchNormalization126\n\n");
	SW_batch_normalization_39(O122_SW,O126_SW, W126);
	printf("[C_verifier.cpp]Calculate BatchNormalization127\n\n");
	SW_batch_normalization_40(O123_SW,O127_SW, W127);
	printf("[C_verifier.cpp]Calculate Activation(Relu)128\n\n");
	SW_activation_80(O124_SW,O128_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)129\n\n");
	SW_activation_83(O125_SW,O129_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)130\n\n");
	SW_activation_88(O126_SW,O130_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)131\n\n");
	SW_activation_89(O127_SW,O131_SW);
	printf("[C_verifier.cpp]Calculate Concatenate132\n\n");
	SW_mixed4(O128_SW, O129_SW, O130_SW, O131_SW,O132_SW);
	printf("[C_verifier.cpp]Calculate Conv2D133\n\n");
	SW_conv2d_45(O132_SW,O133_SW,W133);
	printf("[C_verifier.cpp]Calculate BatchNormalization134\n\n");
	SW_batch_normalization_45(O133_SW,O134_SW, W134);
	printf("[C_verifier.cpp]Calculate Activation(Relu)135\n\n");
	SW_activation_94(O134_SW,O135_SW);
	printf("[C_verifier.cpp]Calculate Conv2D136\n\n");
	SW_conv2d_46(O135_SW,O136_SW,W136);
	printf("[C_verifier.cpp]Calculate BatchNormalization137\n\n");
	SW_batch_normalization_46(O136_SW,O137_SW, W137);
	printf("[C_verifier.cpp]Calculate Activation(Relu)138\n\n");
	SW_activation_95(O137_SW,O138_SW);
	printf("[C_verifier.cpp]Calculate Conv2D139\n\n");
	SW_conv2d_42(O132_SW,O139_SW,W139);
	printf("[C_verifier.cpp]Calculate Conv2D140\n\n");
	SW_conv2d_47(O138_SW,O140_SW,W140);
	printf("[C_verifier.cpp]Calculate BatchNormalization141\n\n");
	SW_batch_normalization_42(O139_SW,O141_SW, W141);
	printf("[C_verifier.cpp]Calculate BatchNormalization142\n\n");
	SW_batch_normalization_47(O140_SW,O142_SW, W142);
	printf("[C_verifier.cpp]Calculate Activation(Relu)143\n\n");
	SW_activation_91(O141_SW,O143_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)144\n\n");
	SW_activation_96(O142_SW,O144_SW);
	printf("[C_verifier.cpp]Calculate Conv2D145\n\n");
	SW_conv2d_43(O143_SW,O145_SW,W145);
	printf("[C_verifier.cpp]Calculate Conv2D146\n\n");
	SW_conv2d_48(O144_SW,O146_SW,W146);
	printf("[C_verifier.cpp]Calculate BatchNormalization147\n\n");
	SW_batch_normalization_43(O145_SW,O147_SW, W147);
	printf("[C_verifier.cpp]Calculate BatchNormalization148\n\n");
	SW_batch_normalization_48(O146_SW,O148_SW, W148);
	printf("[C_verifier.cpp]Calculate Activation(Relu)149\n\n");
	SW_activation_92(O147_SW,O149_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)150\n\n");
	SW_activation_97(O148_SW,O150_SW);
	printf("[C_verifier.cpp]Calculate AveragePooling2D151\n\n");
	SW_average_pooling2d_5(O132_SW,O151_SW);
	printf("[C_verifier.cpp]Calculate Conv2D152\n\n");
	SW_conv2d_41(O132_SW,O152_SW,W152);
	printf("[C_verifier.cpp]Calculate Conv2D153\n\n");
	SW_conv2d_44(O149_SW,O153_SW,W153);
	printf("[C_verifier.cpp]Calculate Conv2D154\n\n");
	SW_conv2d_49(O150_SW,O154_SW,W154);
	printf("[C_verifier.cpp]Calculate Conv2D155\n\n");
	SW_conv2d_50(O151_SW,O155_SW,W155);
	printf("[C_verifier.cpp]Calculate BatchNormalization156\n\n");
	SW_batch_normalization_41(O152_SW,O156_SW, W156);
	printf("[C_verifier.cpp]Calculate BatchNormalization157\n\n");
	SW_batch_normalization_44(O153_SW,O157_SW, W157);
	printf("[C_verifier.cpp]Calculate BatchNormalization158\n\n");
	SW_batch_normalization_49(O154_SW,O158_SW, W158);
	printf("[C_verifier.cpp]Calculate BatchNormalization159\n\n");
	SW_batch_normalization_50(O155_SW,O159_SW, W159);
	printf("[C_verifier.cpp]Calculate Activation(Relu)160\n\n");
	SW_activation_90(O156_SW,O160_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)161\n\n");
	SW_activation_93(O157_SW,O161_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)162\n\n");
	SW_activation_98(O158_SW,O162_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)163\n\n");
	SW_activation_99(O159_SW,O163_SW);
	printf("[C_verifier.cpp]Calculate Concatenate164\n\n");
	SW_mixed5(O160_SW, O161_SW, O162_SW, O163_SW,O164_SW);
	printf("[C_verifier.cpp]Calculate Conv2D165\n\n");
	SW_conv2d_55(O164_SW,O165_SW,W165);
	printf("[C_verifier.cpp]Calculate BatchNormalization166\n\n");
	SW_batch_normalization_55(O165_SW,O166_SW, W166);
	printf("[C_verifier.cpp]Calculate Activation(Relu)167\n\n");
	SW_activation_104(O166_SW,O167_SW);
	printf("[C_verifier.cpp]Calculate Conv2D168\n\n");
	SW_conv2d_56(O167_SW,O168_SW,W168);
	printf("[C_verifier.cpp]Calculate BatchNormalization169\n\n");
	SW_batch_normalization_56(O168_SW,O169_SW, W169);
	printf("[C_verifier.cpp]Calculate Activation(Relu)170\n\n");
	SW_activation_105(O169_SW,O170_SW);
	printf("[C_verifier.cpp]Calculate Conv2D171\n\n");
	SW_conv2d_52(O164_SW,O171_SW,W171);
	printf("[C_verifier.cpp]Calculate Conv2D172\n\n");
	SW_conv2d_57(O170_SW,O172_SW,W172);
	printf("[C_verifier.cpp]Calculate BatchNormalization173\n\n");
	SW_batch_normalization_52(O171_SW,O173_SW, W173);
	printf("[C_verifier.cpp]Calculate BatchNormalization174\n\n");
	SW_batch_normalization_57(O172_SW,O174_SW, W174);
	printf("[C_verifier.cpp]Calculate Activation(Relu)175\n\n");
	SW_activation_101(O173_SW,O175_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)176\n\n");
	SW_activation_106(O174_SW,O176_SW);
	printf("[C_verifier.cpp]Calculate Conv2D177\n\n");
	SW_conv2d_53(O175_SW,O177_SW,W177);
	printf("[C_verifier.cpp]Calculate Conv2D178\n\n");
	SW_conv2d_58(O176_SW,O178_SW,W178);
	printf("[C_verifier.cpp]Calculate BatchNormalization179\n\n");
	SW_batch_normalization_53(O177_SW,O179_SW, W179);
	printf("[C_verifier.cpp]Calculate BatchNormalization180\n\n");
	SW_batch_normalization_58(O178_SW,O180_SW, W180);
	printf("[C_verifier.cpp]Calculate Activation(Relu)181\n\n");
	SW_activation_102(O179_SW,O181_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)182\n\n");
	SW_activation_107(O180_SW,O182_SW);
	printf("[C_verifier.cpp]Calculate AveragePooling2D183\n\n");
	SW_average_pooling2d_6(O164_SW,O183_SW);
	printf("[C_verifier.cpp]Calculate Conv2D184\n\n");
	SW_conv2d_51(O164_SW,O184_SW,W184);
	printf("[C_verifier.cpp]Calculate Conv2D185\n\n");
	SW_conv2d_54(O181_SW,O185_SW,W185);
	printf("[C_verifier.cpp]Calculate Conv2D186\n\n");
	SW_conv2d_59(O182_SW,O186_SW,W186);
	printf("[C_verifier.cpp]Calculate Conv2D187\n\n");
	SW_conv2d_60(O183_SW,O187_SW,W187);
	printf("[C_verifier.cpp]Calculate BatchNormalization188\n\n");
	SW_batch_normalization_51(O184_SW,O188_SW, W188);
	printf("[C_verifier.cpp]Calculate BatchNormalization189\n\n");
	SW_batch_normalization_54(O185_SW,O189_SW, W189);
	printf("[C_verifier.cpp]Calculate BatchNormalization190\n\n");
	SW_batch_normalization_59(O186_SW,O190_SW, W190);
	printf("[C_verifier.cpp]Calculate BatchNormalization191\n\n");
	SW_batch_normalization_60(O187_SW,O191_SW, W191);
	printf("[C_verifier.cpp]Calculate Activation(Relu)192\n\n");
	SW_activation_100(O188_SW,O192_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)193\n\n");
	SW_activation_103(O189_SW,O193_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)194\n\n");
	SW_activation_108(O190_SW,O194_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)195\n\n");
	SW_activation_109(O191_SW,O195_SW);
	printf("[C_verifier.cpp]Calculate Concatenate196\n\n");
	SW_mixed6(O192_SW, O193_SW, O194_SW, O195_SW,O196_SW);
	printf("[C_verifier.cpp]Calculate Conv2D197\n\n");
	SW_conv2d_65(O196_SW,O197_SW,W197);
	printf("[C_verifier.cpp]Calculate BatchNormalization198\n\n");
	SW_batch_normalization_65(O197_SW,O198_SW, W198);
	printf("[C_verifier.cpp]Calculate Activation(Relu)199\n\n");
	SW_activation_114(O198_SW,O199_SW);
	printf("[C_verifier.cpp]Calculate Conv2D200\n\n");
	SW_conv2d_66(O199_SW,O200_SW,W200);
	printf("[C_verifier.cpp]Calculate BatchNormalization201\n\n");
	SW_batch_normalization_66(O200_SW,O201_SW, W201);
	printf("[C_verifier.cpp]Calculate Activation(Relu)202\n\n");
	SW_activation_115(O201_SW,O202_SW);
	printf("[C_verifier.cpp]Calculate Conv2D203\n\n");
	SW_conv2d_62(O196_SW,O203_SW,W203);
	printf("[C_verifier.cpp]Calculate Conv2D204\n\n");
	SW_conv2d_67(O202_SW,O204_SW,W204);
	printf("[C_verifier.cpp]Calculate BatchNormalization205\n\n");
	SW_batch_normalization_62(O203_SW,O205_SW, W205);
	printf("[C_verifier.cpp]Calculate BatchNormalization206\n\n");
	SW_batch_normalization_67(O204_SW,O206_SW, W206);
	printf("[C_verifier.cpp]Calculate Activation(Relu)207\n\n");
	SW_activation_111(O205_SW,O207_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)208\n\n");
	SW_activation_116(O206_SW,O208_SW);
	printf("[C_verifier.cpp]Calculate Conv2D209\n\n");
	SW_conv2d_63(O207_SW,O209_SW,W209);
	printf("[C_verifier.cpp]Calculate Conv2D210\n\n");
	SW_conv2d_68(O208_SW,O210_SW,W210);
	printf("[C_verifier.cpp]Calculate BatchNormalization211\n\n");
	SW_batch_normalization_63(O209_SW,O211_SW, W211);
	printf("[C_verifier.cpp]Calculate BatchNormalization212\n\n");
	SW_batch_normalization_68(O210_SW,O212_SW, W212);
	printf("[C_verifier.cpp]Calculate Activation(Relu)213\n\n");
	SW_activation_112(O211_SW,O213_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)214\n\n");
	SW_activation_117(O212_SW,O214_SW);
	printf("[C_verifier.cpp]Calculate AveragePooling2D215\n\n");
	SW_average_pooling2d_7(O196_SW,O215_SW);
	printf("[C_verifier.cpp]Calculate Conv2D216\n\n");
	SW_conv2d_61(O196_SW,O216_SW,W216);
	printf("[C_verifier.cpp]Calculate Conv2D217\n\n");
	SW_conv2d_64(O213_SW,O217_SW,W217);
	printf("[C_verifier.cpp]Calculate Conv2D218\n\n");
	SW_conv2d_69(O214_SW,O218_SW,W218);
	printf("[C_verifier.cpp]Calculate Conv2D219\n\n");
	SW_conv2d_70(O215_SW,O219_SW,W219);
	printf("[C_verifier.cpp]Calculate BatchNormalization220\n\n");
	SW_batch_normalization_61(O216_SW,O220_SW, W220);
	printf("[C_verifier.cpp]Calculate BatchNormalization221\n\n");
	SW_batch_normalization_64(O217_SW,O221_SW, W221);
	printf("[C_verifier.cpp]Calculate BatchNormalization222\n\n");
	SW_batch_normalization_69(O218_SW,O222_SW, W222);
	printf("[C_verifier.cpp]Calculate BatchNormalization223\n\n");
	SW_batch_normalization_70(O219_SW,O223_SW, W223);
	printf("[C_verifier.cpp]Calculate Activation(Relu)224\n\n");
	SW_activation_110(O220_SW,O224_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)225\n\n");
	SW_activation_113(O221_SW,O225_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)226\n\n");
	SW_activation_118(O222_SW,O226_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)227\n\n");
	SW_activation_119(O223_SW,O227_SW);
	printf("[C_verifier.cpp]Calculate Concatenate228\n\n");
	SW_mixed7(O224_SW, O225_SW, O226_SW, O227_SW,O228_SW);
	printf("[C_verifier.cpp]Calculate Conv2D229\n\n");
	SW_conv2d_73(O228_SW,O229_SW,W229);
	printf("[C_verifier.cpp]Calculate BatchNormalization230\n\n");
	SW_batch_normalization_73(O229_SW,O230_SW, W230);
	printf("[C_verifier.cpp]Calculate Activation(Relu)231\n\n");
	SW_activation_122(O230_SW,O231_SW);
	printf("[C_verifier.cpp]Calculate Conv2D232\n\n");
	SW_conv2d_74(O231_SW,O232_SW,W232);
	printf("[C_verifier.cpp]Calculate BatchNormalization233\n\n");
	SW_batch_normalization_74(O232_SW,O233_SW, W233);
	printf("[C_verifier.cpp]Calculate Activation(Relu)234\n\n");
	SW_activation_123(O233_SW,O234_SW);
	printf("[C_verifier.cpp]Calculate Conv2D235\n\n");
	SW_conv2d_71(O228_SW,O235_SW,W235);
	printf("[C_verifier.cpp]Calculate Conv2D236\n\n");
	SW_conv2d_75(O234_SW,O236_SW,W236);
	printf("[C_verifier.cpp]Calculate BatchNormalization237\n\n");
	SW_batch_normalization_71(O235_SW,O237_SW, W237);
	printf("[C_verifier.cpp]Calculate BatchNormalization238\n\n");
	SW_batch_normalization_75(O236_SW,O238_SW, W238);
	printf("[C_verifier.cpp]Calculate Activation(Relu)239\n\n");
	SW_activation_120(O237_SW,O239_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)240\n\n");
	SW_activation_124(O238_SW,O240_SW);
	printf("[C_verifier.cpp]Calculate Conv2D241\n\n");
	SW_conv2d_72(O239_SW,O241_SW,W241);
	printf("[C_verifier.cpp]Calculate Conv2D242\n\n");
	SW_conv2d_76(O240_SW,O242_SW,W242);
	printf("[C_verifier.cpp]Calculate BatchNormalization243\n\n");
	SW_batch_normalization_72(O241_SW,O243_SW, W243);
	printf("[C_verifier.cpp]Calculate BatchNormalization244\n\n");
	SW_batch_normalization_76(O242_SW,O244_SW, W244);
	printf("[C_verifier.cpp]Calculate Activation(Relu)245\n\n");
	SW_activation_121(O243_SW,O245_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)246\n\n");
	SW_activation_125(O244_SW,O246_SW);
	printf("[C_verifier.cpp]Calculate MaxPooling2D247\n\n");
	SW_max_pooling2d_5(O228_SW,O247_SW);
	printf("[C_verifier.cpp]Calculate Concatenate248\n\n");
	SW_mixed8(O245_SW, O246_SW, O247_SW,O248_SW);
	printf("[C_verifier.cpp]Calculate Conv2D249\n\n");
	SW_conv2d_81(O248_SW,O249_SW,W249);
	printf("[C_verifier.cpp]Calculate BatchNormalization250\n\n");
	SW_batch_normalization_81(O249_SW,O250_SW, W250);
	printf("[C_verifier.cpp]Calculate Activation(Relu)251\n\n");
	SW_activation_130(O250_SW,O251_SW);
	printf("[C_verifier.cpp]Calculate Conv2D252\n\n");
	SW_conv2d_78(O248_SW,O252_SW,W252);
	printf("[C_verifier.cpp]Calculate Conv2D253\n\n");
	SW_conv2d_82(O251_SW,O253_SW,W253);
	printf("[C_verifier.cpp]Calculate BatchNormalization254\n\n");
	SW_batch_normalization_78(O252_SW,O254_SW, W254);
	printf("[C_verifier.cpp]Calculate BatchNormalization255\n\n");
	SW_batch_normalization_82(O253_SW,O255_SW, W255);
	printf("[C_verifier.cpp]Calculate Activation(Relu)256\n\n");
	SW_activation_127(O254_SW,O256_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)257\n\n");
	SW_activation_131(O255_SW,O257_SW);
	printf("[C_verifier.cpp]Calculate Conv2D258\n\n");
	SW_conv2d_79(O256_SW,O258_SW,W258);
	printf("[C_verifier.cpp]Calculate Conv2D259\n\n");
	SW_conv2d_80(O256_SW,O259_SW,W259);
	printf("[C_verifier.cpp]Calculate Conv2D260\n\n");
	SW_conv2d_83(O257_SW,O260_SW,W260);
	printf("[C_verifier.cpp]Calculate Conv2D261\n\n");
	SW_conv2d_84(O257_SW,O261_SW,W261);
	printf("[C_verifier.cpp]Calculate AveragePooling2D262\n\n");
	SW_average_pooling2d_8(O248_SW,O262_SW);
	printf("[C_verifier.cpp]Calculate Conv2D263\n\n");
	SW_conv2d_77(O248_SW,O263_SW,W263);
	printf("[C_verifier.cpp]Calculate BatchNormalization264\n\n");
	SW_batch_normalization_79(O258_SW,O264_SW, W264);
	printf("[C_verifier.cpp]Calculate BatchNormalization265\n\n");
	SW_batch_normalization_80(O259_SW,O265_SW, W265);
	printf("[C_verifier.cpp]Calculate BatchNormalization266\n\n");
	SW_batch_normalization_83(O260_SW,O266_SW, W266);
	printf("[C_verifier.cpp]Calculate BatchNormalization267\n\n");
	SW_batch_normalization_84(O261_SW,O267_SW, W267);
	printf("[C_verifier.cpp]Calculate Conv2D268\n\n");
	SW_conv2d_85(O262_SW,O268_SW,W268);
	printf("[C_verifier.cpp]Calculate BatchNormalization269\n\n");
	SW_batch_normalization_77(O263_SW,O269_SW, W269);
	printf("[C_verifier.cpp]Calculate Activation(Relu)270\n\n");
	SW_activation_128(O264_SW,O270_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)271\n\n");
	SW_activation_129(O265_SW,O271_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)272\n\n");
	SW_activation_132(O266_SW,O272_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)273\n\n");
	SW_activation_133(O267_SW,O273_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization274\n\n");
	SW_batch_normalization_85(O268_SW,O274_SW, W274);
	printf("[C_verifier.cpp]Calculate Activation(Relu)275\n\n");
	SW_activation_126(O269_SW,O275_SW);
	printf("[C_verifier.cpp]Calculate Concatenate276\n\n");
	printf("[C_verifier.cpp]Calculate Concatenate277\n\n");
	printf("[C_verifier.cpp]Calculate Activation(Relu)278\n\n");
	SW_activation_134(O274_SW,O278_SW);
	printf("[C_verifier.cpp]Calculate Concatenate279\n\n");
	SW_mixed9(O275_SW, O276_SW, O277_SW, O278_SW,O279_SW);
	printf("[C_verifier.cpp]Calculate Conv2D280\n\n");
	SW_conv2d_90(O279_SW,O280_SW,W280);
	printf("[C_verifier.cpp]Calculate BatchNormalization281\n\n");
	SW_batch_normalization_90(O280_SW,O281_SW, W281);
	printf("[C_verifier.cpp]Calculate Activation(Relu)282\n\n");
	SW_activation_139(O281_SW,O282_SW);
	printf("[C_verifier.cpp]Calculate Conv2D283\n\n");
	SW_conv2d_87(O279_SW,O283_SW,W283);
	printf("[C_verifier.cpp]Calculate Conv2D284\n\n");
	SW_conv2d_91(O282_SW,O284_SW,W284);
	printf("[C_verifier.cpp]Calculate BatchNormalization285\n\n");
	SW_batch_normalization_87(O283_SW,O285_SW, W285);
	printf("[C_verifier.cpp]Calculate BatchNormalization286\n\n");
	SW_batch_normalization_91(O284_SW,O286_SW, W286);
	printf("[C_verifier.cpp]Calculate Activation(Relu)287\n\n");
	SW_activation_136(O285_SW,O287_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)288\n\n");
	SW_activation_140(O286_SW,O288_SW);
	printf("[C_verifier.cpp]Calculate Conv2D289\n\n");
	SW_conv2d_88(O287_SW,O289_SW,W289);
	printf("[C_verifier.cpp]Calculate Conv2D290\n\n");
	SW_conv2d_89(O287_SW,O290_SW,W290);
	printf("[C_verifier.cpp]Calculate Conv2D291\n\n");
	SW_conv2d_92(O288_SW,O291_SW,W291);
	printf("[C_verifier.cpp]Calculate Conv2D292\n\n");
	SW_conv2d_93(O288_SW,O292_SW,W292);
	printf("[C_verifier.cpp]Calculate AveragePooling2D293\n\n");
	SW_average_pooling2d_9(O279_SW,O293_SW);
	printf("[C_verifier.cpp]Calculate Conv2D294\n\n");
	SW_conv2d_86(O279_SW,O294_SW,W294);
	printf("[C_verifier.cpp]Calculate BatchNormalization295\n\n");
	SW_batch_normalization_88(O289_SW,O295_SW, W295);
	printf("[C_verifier.cpp]Calculate BatchNormalization296\n\n");
	SW_batch_normalization_89(O290_SW,O296_SW, W296);
	printf("[C_verifier.cpp]Calculate BatchNormalization297\n\n");
	SW_batch_normalization_92(O291_SW,O297_SW, W297);
	printf("[C_verifier.cpp]Calculate BatchNormalization298\n\n");
	SW_batch_normalization_93(O292_SW,O298_SW, W298);
	printf("[C_verifier.cpp]Calculate Conv2D299\n\n");
	SW_conv2d_94(O293_SW,O299_SW,W299);
	printf("[C_verifier.cpp]Calculate BatchNormalization300\n\n");
	SW_batch_normalization_86(O294_SW,O300_SW, W300);
	printf("[C_verifier.cpp]Calculate Activation(Relu)301\n\n");
	SW_activation_137(O295_SW,O301_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)302\n\n");
	SW_activation_138(O296_SW,O302_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)303\n\n");
	SW_activation_141(O297_SW,O303_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)304\n\n");
	SW_activation_142(O298_SW,O304_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization305\n\n");
	SW_batch_normalization_94(O299_SW,O305_SW, W305);
	printf("[C_verifier.cpp]Calculate Activation(Relu)306\n\n");
	SW_activation_135(O300_SW,O306_SW);
	printf("[C_verifier.cpp]Calculate Concatenate307\n\n");
	printf("[C_verifier.cpp]Calculate Concatenate308\n\n");
	printf("[C_verifier.cpp]Calculate Activation(Relu)309\n\n");
	SW_activation_143(O305_SW,O309_SW);
	printf("[C_verifier.cpp]Calculate Concatenate310\n\n");
	SW_mixed10(O306_SW, O307_SW, O308_SW, O309_SW,O310_SW);
	printf("[C_verifier.cpp]Calculate GlobalAveragePooling2D311\n\n");
	SW_avg_pool(O310_SW,O311_SW);
	printf("[C_verifier.cpp]Calculate Dense312\n\n");
	SW_predictions(O311_SW,O312_SW,W312,B312);
	

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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
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
		for(y = 0; y < 64 ; y++) {
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
		for(y = 0; y < 64 ; y++) {
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
		for(y = 0; y < 48 ; y++) {
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
		for(y = 0; y < 96 ; y++) {
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
		for(y = 0; y < 48 ; y++) {
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
		for(y = 0; y < 96 ; y++) {
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
		for(y = 0; y < 48 ; y++) {
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
		for(y = 0; y < 96 ; y++) {
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


fprintf(o_stream,"%s","AveragePooling2D : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
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
		for(y = 0; y < 64 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 96 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 96 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 96 ; y++) {
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
		for(y = 0; y < 64 ; y++) {
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 288 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 96 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 96 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 96 ; y++) {
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


fprintf(o_stream,"%s","AveragePooling2D : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 288 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 96 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 96 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
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
		for(y = 0; y < 96 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 288 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 25 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 25 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 96 ; y++) {
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
		for(y = 0; y < 96 ; y++) {
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
		for(y = 0; y < 96 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O93_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O93_SW[y][k][x]);
		}
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
		for(y = 0; y < 96 ; y++) {
			fprintf(o_stream,"%.6f ",O94_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O94_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O95_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O95_SW[y][k][x]);
		}
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
		for(y = 0; y < 96 ; y++) {
			fprintf(o_stream,"%.6f ",O96_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O96_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O97_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O97_SW[y][k][x]);
		}
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
		for(y = 0; y < 96 ; y++) {
			fprintf(o_stream,"%.6f ",O98_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O98_SW[y][k][x]);
		}
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
		for(y = 0; y < 288 ; y++) {
			fprintf(o_stream,"%.6f ",O99_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O99_SW[y][k][x]);
		}
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
		for(y = 0; y < 768 ; y++) {
			fprintf(o_stream,"%.6f ",O100_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O100_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O101_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O101_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O102_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O102_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O103_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O103_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O104_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O104_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O105_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O105_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O106_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O106_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O107_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O107_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O108_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O108_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O109_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O109_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O110_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O110_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O111_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O111_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O112_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O112_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O113_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O113_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O114_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O114_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O115_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O115_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O116_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O116_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O117_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O117_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O118_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O118_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","AveragePooling2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 768 ; y++) {
			fprintf(o_stream,"%.6f ",O119_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O119_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O120_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O120_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O121_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O121_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O122_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O122_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O123_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O123_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O124_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O124_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O125_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O125_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O126_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O126_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O127_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O127_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O128_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O128_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O129_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O129_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O130_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O130_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O131_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O131_SW[y][k][x]);
		}
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
		for(y = 0; y < 768 ; y++) {
			fprintf(o_stream,"%.6f ",O132_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O132_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O133_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O133_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O134_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O134_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O135_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O135_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O136_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O136_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O137_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O137_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O138_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O138_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O139_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O139_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O140_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O140_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O141_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O141_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O142_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O142_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O143_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O143_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O144_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O144_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O145_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O145_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O146_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O146_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O147_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O147_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O148_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O148_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O149_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O149_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O150_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O150_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","AveragePooling2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 768 ; y++) {
			fprintf(o_stream,"%.6f ",O151_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O151_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O152_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O152_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O153_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O153_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O154_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O154_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O155_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O155_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O156_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O156_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O157_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O157_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O158_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O158_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O159_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O159_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O160_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O160_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O161_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O161_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O162_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O162_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O163_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O163_SW[y][k][x]);
		}
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
		for(y = 0; y < 768 ; y++) {
			fprintf(o_stream,"%.6f ",O164_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O164_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O165_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O165_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O166_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O166_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O167_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O167_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O168_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O168_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O169_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O169_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O170_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O170_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O171_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O171_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O172_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O172_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O173_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O173_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O174_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O174_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O175_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O175_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O176_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O176_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O177_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O177_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O178_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O178_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O179_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O179_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O180_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O180_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O181_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O181_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O182_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O182_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","AveragePooling2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 768 ; y++) {
			fprintf(o_stream,"%.6f ",O183_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O183_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O184_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O184_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O185_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O185_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O186_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O186_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O187_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O187_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O188_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O188_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O189_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O189_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O190_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O190_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O191_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O191_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O192_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O192_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O193_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O193_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O194_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O194_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O195_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O195_SW[y][k][x]);
		}
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
		for(y = 0; y < 768 ; y++) {
			fprintf(o_stream,"%.6f ",O196_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O196_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O197_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O197_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O198_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O198_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O199_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O199_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O200_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O200_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O201_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O201_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O202_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O202_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O203_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O203_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O204_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O204_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O205_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O205_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O206_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O206_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O207_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O207_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O208_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O208_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O209_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O209_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O210_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O210_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O211_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O211_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O212_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O212_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O213_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O213_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O214_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O214_SW[y][k][x]);
		}
		if(x != 12 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 12 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","AveragePooling2D : [[");
for (k = 0; k < 12 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 12 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 768 ; y++) {
			fprintf(o_stream,"%.6f ",O215_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O215_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O216_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O216_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O217_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O217_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O218_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O218_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O219_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O219_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O220_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O220_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O221_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O221_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O222_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O222_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O223_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O223_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O224_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O224_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O225_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O225_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O226_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O226_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O227_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O227_SW[y][k][x]);
		}
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
		for(y = 0; y < 768 ; y++) {
			fprintf(o_stream,"%.6f ",O228_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O228_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O229_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O229_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O230_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O230_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O231_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O231_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O232_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O232_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O233_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O233_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O234_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O234_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O235_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O235_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O236_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O236_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O237_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O237_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O238_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O238_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O239_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O239_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O240_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O240_SW[y][k][x]);
		}
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
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O241_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O241_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O242_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O242_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O243_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O243_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O244_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O244_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O245_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O245_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O246_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O246_SW[y][k][x]);
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
		for(y = 0; y < 768 ; y++) {
			fprintf(o_stream,"%.6f ",O247_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O247_SW[y][k][x]);
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
		for(y = 0; y < 1280 ; y++) {
			fprintf(o_stream,"%.6f ",O248_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O248_SW[y][k][x]);
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
		for(y = 0; y < 448 ; y++) {
			fprintf(o_stream,"%.6f ",O249_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O249_SW[y][k][x]);
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
		for(y = 0; y < 448 ; y++) {
			fprintf(o_stream,"%.6f ",O250_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O250_SW[y][k][x]);
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
		for(y = 0; y < 448 ; y++) {
			fprintf(o_stream,"%.6f ",O251_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O251_SW[y][k][x]);
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
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O252_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O252_SW[y][k][x]);
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
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O253_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O253_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O254_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O254_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O255_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O255_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O256_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O256_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O257_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O257_SW[y][k][x]);
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
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O258_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O258_SW[y][k][x]);
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
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O259_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O259_SW[y][k][x]);
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
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O260_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O260_SW[y][k][x]);
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
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O261_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O261_SW[y][k][x]);
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


fprintf(o_stream,"%s","AveragePooling2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1280 ; y++) {
			fprintf(o_stream,"%.6f ",O262_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O262_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O263_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O263_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O264_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O264_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O265_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O265_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O266_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O266_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O267_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O267_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O268_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O268_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O269_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O269_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O270_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O270_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O271_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O271_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O272_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O272_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O273_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O273_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O274_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O274_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O275_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O275_SW[y][k][x]);
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
		for(y = 0; y < 768 ; y++) {
			fprintf(o_stream,"%.6f ",O276_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O276_SW[y][k][x]);
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
		for(y = 0; y < 768 ; y++) {
			fprintf(o_stream,"%.6f ",O277_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O277_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O278_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O278_SW[y][k][x]);
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
		for(y = 0; y < 2048 ; y++) {
			fprintf(o_stream,"%.6f ",O279_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O279_SW[y][k][x]);
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
		for(y = 0; y < 448 ; y++) {
			fprintf(o_stream,"%.6f ",O280_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O280_SW[y][k][x]);
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
		for(y = 0; y < 448 ; y++) {
			fprintf(o_stream,"%.6f ",O281_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O281_SW[y][k][x]);
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
		for(y = 0; y < 448 ; y++) {
			fprintf(o_stream,"%.6f ",O282_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O282_SW[y][k][x]);
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
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O283_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O283_SW[y][k][x]);
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
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O284_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O284_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O285_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O285_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O286_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O286_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O287_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O287_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O288_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O288_SW[y][k][x]);
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
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O289_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O289_SW[y][k][x]);
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
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O290_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O290_SW[y][k][x]);
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
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O291_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O291_SW[y][k][x]);
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
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O292_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O292_SW[y][k][x]);
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


fprintf(o_stream,"%s","AveragePooling2D : [[");
for (k = 0; k < 5 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 5 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2048 ; y++) {
			fprintf(o_stream,"%.6f ",O293_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O293_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O294_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O294_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O295_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O295_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O296_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O296_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O297_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O297_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O298_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O298_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O299_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O299_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O300_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O300_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O301_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O301_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O302_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O302_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O303_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O303_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O304_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O304_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O305_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O305_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O306_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O306_SW[y][k][x]);
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
		for(y = 0; y < 768 ; y++) {
			fprintf(o_stream,"%.6f ",O307_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O307_SW[y][k][x]);
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
		for(y = 0; y < 768 ; y++) {
			fprintf(o_stream,"%.6f ",O308_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O308_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O309_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O309_SW[y][k][x]);
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
		for(y = 0; y < 2048 ; y++) {
			fprintf(o_stream,"%.6f ",O310_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O310_SW[y][k][x]);
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
for (k = 0; k <  2048 ; k++) {
	fprintf(o_stream,"%.6f ",O311_SW[k]);
	fprintf(c_num,"%.6f ",O311_SW[k]);
}
fprintf(o_stream,"%s","]]\n\n");


fprintf(o_stream,"%s","Dense : [[");
for (k = 0; k <  1000 ; k++) {
	fprintf(o_stream,"%.6f ",O312_SW[k]);
	fprintf(c_num,"%.6f ",O312_SW[k]);
}
fprintf(o_stream,"%s","]]\n\n");





    fclose(w_stream);
    fclose(i_stream);
    fclose(o_stream);
    fclose(c_num);

    return 0;
}
