#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <math.h>

using namespace std;

typedef float DATA_T;

void SW_zero_padding2d_3(DATA_T I[3][224][224],DATA_T O[3][230][230]) {
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
void SW_conv1_conv(DATA_T I[3][230][230], DATA_T O[64][112][112], DATA_T W[64][3][7][7]) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_conv1_bn(DATA_T I[64][112][112], DATA_T O[64][112][112], DATA_T W[4][64]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

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
void SW_conv1_relu(DATA_T I[64][112][112], DATA_T O[64][112][112]) {
	int m, x, y, i, j, k;
	for (m = 0; m<64; m++) {
		for (x = 0; x<112; x++) {
			for (y = 0; y<112; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_zero_padding2d_4(DATA_T I[64][112][112],DATA_T O[64][114][114]) {
	int m, x, y, i, j;
	for (m = 0; m < 64; m++) {
		for (x = 0; x < 114; x++) {
			for (y = 0; y < 114; y++) {
				O[m][x][y] = 0;
			}
		}
	}
	for (m = 0; m < 64; m++) {
		for (x = 0; x < 112; x++) {
			for (y = 0; y < 112; y++) {
				O[m][x+ 1][y+ 1] = I[m][x][y];
			}
		}
	}
}
void SW_pool1(DATA_T I[64][114][114], DATA_T O[64][56][56])
{
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<64; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
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
void SW_conv2_block1_0_bn(DATA_T I[64][56][56], DATA_T O[64][56][56], DATA_T W[4][64]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

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
void SW_conv2_block1_0_relu(DATA_T I[64][56][56], DATA_T O[64][56][56]) {
	int m, x, y, i, j, k;
	for (m = 0; m<64; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2_block1_1_conv(DATA_T I[64][56][56], DATA_T O[128][56][56], DATA_T W[128][64][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				ofm = 0;
				for (k = 0; k<64; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 56 && y + j <= 56) {
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

void SW_conv2_block1_1_bn(DATA_T I[128][56][56], DATA_T O[128][56][56], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

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
void SW_conv2_block1_1_relu(DATA_T I[128][56][56], DATA_T O[128][56][56]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2_block1_2_conv(DATA_T I[128][56][56], DATA_T O[32][56][56], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(56 - 1) - 56 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
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

void SW_conv2_block1_concat(DATA_T I1[64][56][56], DATA_T I2[32][56][56], DATA_T O[96][56][56]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 56; x++) {
		for(y = 0; y < 56; y++) {
			ch=0;
			for(k = ch; k < ch+64; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=64;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv2_block2_0_bn(DATA_T I[96][56][56], DATA_T O[96][56][56], DATA_T W[4][96]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

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
void SW_conv2_block2_0_relu(DATA_T I[96][56][56], DATA_T O[96][56][56]) {
	int m, x, y, i, j, k;
	for (m = 0; m<96; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2_block2_1_conv(DATA_T I[96][56][56], DATA_T O[128][56][56], DATA_T W[128][96][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				ofm = 0;
				for (k = 0; k<96; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 56 && y + j <= 56) {
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

void SW_conv2_block2_1_bn(DATA_T I[128][56][56], DATA_T O[128][56][56], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

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
void SW_conv2_block2_1_relu(DATA_T I[128][56][56], DATA_T O[128][56][56]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2_block2_2_conv(DATA_T I[128][56][56], DATA_T O[32][56][56], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(56 - 1) - 56 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
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

void SW_conv2_block2_concat(DATA_T I1[96][56][56], DATA_T I2[32][56][56], DATA_T O[128][56][56]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 56; x++) {
		for(y = 0; y < 56; y++) {
			ch=0;
			for(k = ch; k < ch+96; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=96;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv2_block3_0_bn(DATA_T I[128][56][56], DATA_T O[128][56][56], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

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
void SW_conv2_block3_0_relu(DATA_T I[128][56][56], DATA_T O[128][56][56]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2_block3_1_conv(DATA_T I[128][56][56], DATA_T O[128][56][56], DATA_T W[128][128][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 56 && y + j <= 56) {
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

void SW_conv2_block3_1_bn(DATA_T I[128][56][56], DATA_T O[128][56][56], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

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
void SW_conv2_block3_1_relu(DATA_T I[128][56][56], DATA_T O[128][56][56]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2_block3_2_conv(DATA_T I[128][56][56], DATA_T O[32][56][56], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(56 - 1) - 56 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
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

void SW_conv2_block3_concat(DATA_T I1[128][56][56], DATA_T I2[32][56][56], DATA_T O[160][56][56]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 56; x++) {
		for(y = 0; y < 56; y++) {
			ch=0;
			for(k = ch; k < ch+128; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=128;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv2_block4_0_bn(DATA_T I[160][56][56], DATA_T O[160][56][56], DATA_T W[4][160]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 160; m++){
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
void SW_conv2_block4_0_relu(DATA_T I[160][56][56], DATA_T O[160][56][56]) {
	int m, x, y, i, j, k;
	for (m = 0; m<160; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2_block4_1_conv(DATA_T I[160][56][56], DATA_T O[128][56][56], DATA_T W[128][160][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				ofm = 0;
				for (k = 0; k<160; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 56 && y + j <= 56) {
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

void SW_conv2_block4_1_bn(DATA_T I[128][56][56], DATA_T O[128][56][56], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

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
void SW_conv2_block4_1_relu(DATA_T I[128][56][56], DATA_T O[128][56][56]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2_block4_2_conv(DATA_T I[128][56][56], DATA_T O[32][56][56], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(56 - 1) - 56 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
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

void SW_conv2_block4_concat(DATA_T I1[160][56][56], DATA_T I2[32][56][56], DATA_T O[192][56][56]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 56; x++) {
		for(y = 0; y < 56; y++) {
			ch=0;
			for(k = ch; k < ch+160; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=160;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv2_block5_0_bn(DATA_T I[192][56][56], DATA_T O[192][56][56], DATA_T W[4][192]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 192; m++){
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
void SW_conv2_block5_0_relu(DATA_T I[192][56][56], DATA_T O[192][56][56]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2_block5_1_conv(DATA_T I[192][56][56], DATA_T O[128][56][56], DATA_T W[128][192][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				ofm = 0;
				for (k = 0; k<192; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 56 && y + j <= 56) {
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

void SW_conv2_block5_1_bn(DATA_T I[128][56][56], DATA_T O[128][56][56], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

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
void SW_conv2_block5_1_relu(DATA_T I[128][56][56], DATA_T O[128][56][56]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2_block5_2_conv(DATA_T I[128][56][56], DATA_T O[32][56][56], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(56 - 1) - 56 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
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

void SW_conv2_block5_concat(DATA_T I1[192][56][56], DATA_T I2[32][56][56], DATA_T O[224][56][56]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 56; x++) {
		for(y = 0; y < 56; y++) {
			ch=0;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv2_block6_0_bn(DATA_T I[224][56][56], DATA_T O[224][56][56], DATA_T W[4][224]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 224; m++){
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
void SW_conv2_block6_0_relu(DATA_T I[224][56][56], DATA_T O[224][56][56]) {
	int m, x, y, i, j, k;
	for (m = 0; m<224; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2_block6_1_conv(DATA_T I[224][56][56], DATA_T O[128][56][56], DATA_T W[128][224][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				ofm = 0;
				for (k = 0; k<224; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 56 && y + j <= 56) {
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

void SW_conv2_block6_1_bn(DATA_T I[128][56][56], DATA_T O[128][56][56], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

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
void SW_conv2_block6_1_relu(DATA_T I[128][56][56], DATA_T O[128][56][56]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2_block6_2_conv(DATA_T I[128][56][56], DATA_T O[32][56][56], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(56 - 1) - 56 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
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

void SW_conv2_block6_concat(DATA_T I1[224][56][56], DATA_T I2[32][56][56], DATA_T O[256][56][56]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 56; x++) {
		for(y = 0; y < 56; y++) {
			ch=0;
			for(k = ch; k < ch+224; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=224;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_pool2_bn(DATA_T I[256][56][56], DATA_T O[256][56][56], DATA_T W[4][256]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 256; m++){
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
void SW_pool2_relu(DATA_T I[256][56][56], DATA_T O[256][56][56]) {
	int m, x, y, i, j, k;
	for (m = 0; m<256; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_pool2_conv(DATA_T I[256][56][56], DATA_T O[128][56][56], DATA_T W[128][256][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				ofm = 0;
				for (k = 0; k<256; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 56 && y + j <= 56) {
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

void SW_pool2_pool(DATA_T I[128][56][56], DATA_T O[128][28][28])
{
	int m, x, y, i, j;
	DATA_T sum;
	int p = (2*(28-1) - 56 + 2)/2;
    DATA_T div;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
                sum = (DATA_T)0;
                if(x!=0 && x!=28-1 && y!=0 && y!=28-1)
                    div = 9;
                else if((x==0 || x==28-1) && (y==0 || y==28-1))
                    div = 4;
                else
                    div = 6;
				for (i = 0; i<2; i++) {
					for (j = 0; j<2; j++) {

						if (x + i < 56 + p && y + j < 56 + p && x + i -p >= 0 && y + j -p >= 0)
						    sum += I[m][x*2 + i -p][y*2 + j -p];
					}
				}
				O[m][x][y] = sum/div;
			}
		}
	}
}
void SW_conv3_block1_0_bn(DATA_T I[128][28][28], DATA_T O[128][28][28], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

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
void SW_conv3_block1_0_relu(DATA_T I[128][28][28], DATA_T O[128][28][28]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv3_block1_1_conv(DATA_T I[128][28][28], DATA_T O[128][28][28], DATA_T W[128][128][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_conv3_block1_1_bn(DATA_T I[128][28][28], DATA_T O[128][28][28], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

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
void SW_conv3_block1_1_relu(DATA_T I[128][28][28], DATA_T O[128][28][28]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv3_block1_2_conv(DATA_T I[128][28][28], DATA_T O[32][28][28], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 28 + p && y*1 + j < 28 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv3_block1_concat(DATA_T I1[128][28][28], DATA_T I2[32][28][28], DATA_T O[160][28][28]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 28; x++) {
		for(y = 0; y < 28; y++) {
			ch=0;
			for(k = ch; k < ch+128; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=128;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv3_block2_0_bn(DATA_T I[160][28][28], DATA_T O[160][28][28], DATA_T W[4][160]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 160; m++){
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
void SW_conv3_block2_0_relu(DATA_T I[160][28][28], DATA_T O[160][28][28]) {
	int m, x, y, i, j, k;
	for (m = 0; m<160; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv3_block2_1_conv(DATA_T I[160][28][28], DATA_T O[128][28][28], DATA_T W[128][160][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<160; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 28 && y + j <= 28) {
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

void SW_conv3_block2_1_bn(DATA_T I[128][28][28], DATA_T O[128][28][28], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

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
void SW_conv3_block2_1_relu(DATA_T I[128][28][28], DATA_T O[128][28][28]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv3_block2_2_conv(DATA_T I[128][28][28], DATA_T O[32][28][28], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 28 + p && y*1 + j < 28 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv3_block2_concat(DATA_T I1[160][28][28], DATA_T I2[32][28][28], DATA_T O[192][28][28]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 28; x++) {
		for(y = 0; y < 28; y++) {
			ch=0;
			for(k = ch; k < ch+160; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=160;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv3_block3_0_bn(DATA_T I[192][28][28], DATA_T O[192][28][28], DATA_T W[4][192]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 192; m++){
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
void SW_conv3_block3_0_relu(DATA_T I[192][28][28], DATA_T O[192][28][28]) {
	int m, x, y, i, j, k;
	for (m = 0; m<192; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv3_block3_1_conv(DATA_T I[192][28][28], DATA_T O[128][28][28], DATA_T W[128][192][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<192; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 28 && y + j <= 28) {
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

void SW_conv3_block3_1_bn(DATA_T I[128][28][28], DATA_T O[128][28][28], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

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
void SW_conv3_block3_1_relu(DATA_T I[128][28][28], DATA_T O[128][28][28]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv3_block3_2_conv(DATA_T I[128][28][28], DATA_T O[32][28][28], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 28 + p && y*1 + j < 28 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv3_block3_concat(DATA_T I1[192][28][28], DATA_T I2[32][28][28], DATA_T O[224][28][28]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 28; x++) {
		for(y = 0; y < 28; y++) {
			ch=0;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv3_block4_0_bn(DATA_T I[224][28][28], DATA_T O[224][28][28], DATA_T W[4][224]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 224; m++){
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
void SW_conv3_block4_0_relu(DATA_T I[224][28][28], DATA_T O[224][28][28]) {
	int m, x, y, i, j, k;
	for (m = 0; m<224; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv3_block4_1_conv(DATA_T I[224][28][28], DATA_T O[128][28][28], DATA_T W[128][224][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<224; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 28 && y + j <= 28) {
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

void SW_conv3_block4_1_bn(DATA_T I[128][28][28], DATA_T O[128][28][28], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

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
void SW_conv3_block4_1_relu(DATA_T I[128][28][28], DATA_T O[128][28][28]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv3_block4_2_conv(DATA_T I[128][28][28], DATA_T O[32][28][28], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 28 + p && y*1 + j < 28 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv3_block4_concat(DATA_T I1[224][28][28], DATA_T I2[32][28][28], DATA_T O[256][28][28]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 28; x++) {
		for(y = 0; y < 28; y++) {
			ch=0;
			for(k = ch; k < ch+224; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=224;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv3_block5_0_bn(DATA_T I[256][28][28], DATA_T O[256][28][28], DATA_T W[4][256]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 256; m++){
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
void SW_conv3_block5_0_relu(DATA_T I[256][28][28], DATA_T O[256][28][28]) {
	int m, x, y, i, j, k;
	for (m = 0; m<256; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv3_block5_1_conv(DATA_T I[256][28][28], DATA_T O[128][28][28], DATA_T W[128][256][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<256; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 28 && y + j <= 28) {
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

void SW_conv3_block5_1_bn(DATA_T I[128][28][28], DATA_T O[128][28][28], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

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
void SW_conv3_block5_1_relu(DATA_T I[128][28][28], DATA_T O[128][28][28]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv3_block5_2_conv(DATA_T I[128][28][28], DATA_T O[32][28][28], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 28 + p && y*1 + j < 28 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv3_block5_concat(DATA_T I1[256][28][28], DATA_T I2[32][28][28], DATA_T O[288][28][28]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 28; x++) {
		for(y = 0; y < 28; y++) {
			ch=0;
			for(k = ch; k < ch+256; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=256;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv3_block6_0_bn(DATA_T I[288][28][28], DATA_T O[288][28][28], DATA_T W[4][288]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 288; m++){
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
void SW_conv3_block6_0_relu(DATA_T I[288][28][28], DATA_T O[288][28][28]) {
	int m, x, y, i, j, k;
	for (m = 0; m<288; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv3_block6_1_conv(DATA_T I[288][28][28], DATA_T O[128][28][28], DATA_T W[128][288][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<288; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 28 && y + j <= 28) {
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

void SW_conv3_block6_1_bn(DATA_T I[128][28][28], DATA_T O[128][28][28], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

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
void SW_conv3_block6_1_relu(DATA_T I[128][28][28], DATA_T O[128][28][28]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv3_block6_2_conv(DATA_T I[128][28][28], DATA_T O[32][28][28], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 28 + p && y*1 + j < 28 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv3_block6_concat(DATA_T I1[288][28][28], DATA_T I2[32][28][28], DATA_T O[320][28][28]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 28; x++) {
		for(y = 0; y < 28; y++) {
			ch=0;
			for(k = ch; k < ch+288; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=288;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv3_block7_0_bn(DATA_T I[320][28][28], DATA_T O[320][28][28], DATA_T W[4][320]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 320; m++){
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
void SW_conv3_block7_0_relu(DATA_T I[320][28][28], DATA_T O[320][28][28]) {
	int m, x, y, i, j, k;
	for (m = 0; m<320; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv3_block7_1_conv(DATA_T I[320][28][28], DATA_T O[128][28][28], DATA_T W[128][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 28 && y + j <= 28) {
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

void SW_conv3_block7_1_bn(DATA_T I[128][28][28], DATA_T O[128][28][28], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

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
void SW_conv3_block7_1_relu(DATA_T I[128][28][28], DATA_T O[128][28][28]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv3_block7_2_conv(DATA_T I[128][28][28], DATA_T O[32][28][28], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 28 + p && y*1 + j < 28 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv3_block7_concat(DATA_T I1[320][28][28], DATA_T I2[32][28][28], DATA_T O[352][28][28]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 28; x++) {
		for(y = 0; y < 28; y++) {
			ch=0;
			for(k = ch; k < ch+320; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=320;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv3_block8_0_bn(DATA_T I[352][28][28], DATA_T O[352][28][28], DATA_T W[4][352]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 352; m++){
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
void SW_conv3_block8_0_relu(DATA_T I[352][28][28], DATA_T O[352][28][28]) {
	int m, x, y, i, j, k;
	for (m = 0; m<352; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv3_block8_1_conv(DATA_T I[352][28][28], DATA_T O[128][28][28], DATA_T W[128][352][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<352; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 28 && y + j <= 28) {
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

void SW_conv3_block8_1_bn(DATA_T I[128][28][28], DATA_T O[128][28][28], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

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
void SW_conv3_block8_1_relu(DATA_T I[128][28][28], DATA_T O[128][28][28]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv3_block8_2_conv(DATA_T I[128][28][28], DATA_T O[32][28][28], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 28 + p && y*1 + j < 28 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv3_block8_concat(DATA_T I1[352][28][28], DATA_T I2[32][28][28], DATA_T O[384][28][28]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 28; x++) {
		for(y = 0; y < 28; y++) {
			ch=0;
			for(k = ch; k < ch+352; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=352;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv3_block9_0_bn(DATA_T I[384][28][28], DATA_T O[384][28][28], DATA_T W[4][384]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 384; m++){
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
void SW_conv3_block9_0_relu(DATA_T I[384][28][28], DATA_T O[384][28][28]) {
	int m, x, y, i, j, k;
	for (m = 0; m<384; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv3_block9_1_conv(DATA_T I[384][28][28], DATA_T O[128][28][28], DATA_T W[128][384][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 28 && y + j <= 28) {
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

void SW_conv3_block9_1_bn(DATA_T I[128][28][28], DATA_T O[128][28][28], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

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
void SW_conv3_block9_1_relu(DATA_T I[128][28][28], DATA_T O[128][28][28]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv3_block9_2_conv(DATA_T I[128][28][28], DATA_T O[32][28][28], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 28 + p && y*1 + j < 28 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv3_block9_concat(DATA_T I1[384][28][28], DATA_T I2[32][28][28], DATA_T O[416][28][28]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 28; x++) {
		for(y = 0; y < 28; y++) {
			ch=0;
			for(k = ch; k < ch+384; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=384;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv3_block10_0_bn(DATA_T I[416][28][28], DATA_T O[416][28][28], DATA_T W[4][416]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 416; m++){
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
void SW_conv3_block10_0_relu(DATA_T I[416][28][28], DATA_T O[416][28][28]) {
	int m, x, y, i, j, k;
	for (m = 0; m<416; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv3_block10_1_conv(DATA_T I[416][28][28], DATA_T O[128][28][28], DATA_T W[128][416][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<416; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 28 && y + j <= 28) {
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

void SW_conv3_block10_1_bn(DATA_T I[128][28][28], DATA_T O[128][28][28], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

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
void SW_conv3_block10_1_relu(DATA_T I[128][28][28], DATA_T O[128][28][28]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv3_block10_2_conv(DATA_T I[128][28][28], DATA_T O[32][28][28], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 28 + p && y*1 + j < 28 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv3_block10_concat(DATA_T I1[416][28][28], DATA_T I2[32][28][28], DATA_T O[448][28][28]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 28; x++) {
		for(y = 0; y < 28; y++) {
			ch=0;
			for(k = ch; k < ch+416; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=416;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv3_block11_0_bn(DATA_T I[448][28][28], DATA_T O[448][28][28], DATA_T W[4][448]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 448; m++){
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
void SW_conv3_block11_0_relu(DATA_T I[448][28][28], DATA_T O[448][28][28]) {
	int m, x, y, i, j, k;
	for (m = 0; m<448; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv3_block11_1_conv(DATA_T I[448][28][28], DATA_T O[128][28][28], DATA_T W[128][448][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<448; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 28 && y + j <= 28) {
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

void SW_conv3_block11_1_bn(DATA_T I[128][28][28], DATA_T O[128][28][28], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

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
void SW_conv3_block11_1_relu(DATA_T I[128][28][28], DATA_T O[128][28][28]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv3_block11_2_conv(DATA_T I[128][28][28], DATA_T O[32][28][28], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 28 + p && y*1 + j < 28 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv3_block11_concat(DATA_T I1[448][28][28], DATA_T I2[32][28][28], DATA_T O[480][28][28]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 28; x++) {
		for(y = 0; y < 28; y++) {
			ch=0;
			for(k = ch; k < ch+448; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=448;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv3_block12_0_bn(DATA_T I[480][28][28], DATA_T O[480][28][28], DATA_T W[4][480]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 480; m++){
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
void SW_conv3_block12_0_relu(DATA_T I[480][28][28], DATA_T O[480][28][28]) {
	int m, x, y, i, j, k;
	for (m = 0; m<480; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv3_block12_1_conv(DATA_T I[480][28][28], DATA_T O[128][28][28], DATA_T W[128][480][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<480; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 28 && y + j <= 28) {
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

void SW_conv3_block12_1_bn(DATA_T I[128][28][28], DATA_T O[128][28][28], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

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
void SW_conv3_block12_1_relu(DATA_T I[128][28][28], DATA_T O[128][28][28]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv3_block12_2_conv(DATA_T I[128][28][28], DATA_T O[32][28][28], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 28 + p && y*1 + j < 28 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv3_block12_concat(DATA_T I1[480][28][28], DATA_T I2[32][28][28], DATA_T O[512][28][28]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 28; x++) {
		for(y = 0; y < 28; y++) {
			ch=0;
			for(k = ch; k < ch+480; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=480;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_pool3_bn(DATA_T I[512][28][28], DATA_T O[512][28][28], DATA_T W[4][512]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

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
void SW_pool3_relu(DATA_T I[512][28][28], DATA_T O[512][28][28]) {
	int m, x, y, i, j, k;
	for (m = 0; m<512; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_pool3_conv(DATA_T I[512][28][28], DATA_T O[256][28][28], DATA_T W[256][512][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<256; m++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_pool3_pool(DATA_T I[256][28][28], DATA_T O[256][14][14])
{
	int m, x, y, i, j;
	DATA_T sum;
	int p = (2*(14-1) - 28 + 2)/2;
    DATA_T div;
	for (m = 0; m<256; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
                sum = (DATA_T)0;
                if(x!=0 && x!=14-1 && y!=0 && y!=14-1)
                    div = 9;
                else if((x==0 || x==14-1) && (y==0 || y==14-1))
                    div = 4;
                else
                    div = 6;
				for (i = 0; i<2; i++) {
					for (j = 0; j<2; j++) {

						if (x + i < 28 + p && y + j < 28 + p && x + i -p >= 0 && y + j -p >= 0)
						    sum += I[m][x*2 + i -p][y*2 + j -p];
					}
				}
				O[m][x][y] = sum/div;
			}
		}
	}
}
void SW_conv4_block1_0_bn(DATA_T I[256][14][14], DATA_T O[256][14][14], DATA_T W[4][256]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

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
void SW_conv4_block1_0_relu(DATA_T I[256][14][14], DATA_T O[256][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<256; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block1_1_conv(DATA_T I[256][14][14], DATA_T O[128][14][14], DATA_T W[128][256][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_conv4_block1_1_bn(DATA_T I[128][14][14], DATA_T O[128][14][14], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv4_block1_1_relu(DATA_T I[128][14][14], DATA_T O[128][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block1_2_conv(DATA_T I[128][14][14], DATA_T O[32][14][14], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv4_block1_concat(DATA_T I1[256][14][14], DATA_T I2[32][14][14], DATA_T O[288][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+256; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=256;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv4_block2_0_bn(DATA_T I[288][14][14], DATA_T O[288][14][14], DATA_T W[4][288]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 288; m++){
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
void SW_conv4_block2_0_relu(DATA_T I[288][14][14], DATA_T O[288][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<288; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block2_1_conv(DATA_T I[288][14][14], DATA_T O[128][14][14], DATA_T W[128][288][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<288; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_conv4_block2_1_bn(DATA_T I[128][14][14], DATA_T O[128][14][14], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv4_block2_1_relu(DATA_T I[128][14][14], DATA_T O[128][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block2_2_conv(DATA_T I[128][14][14], DATA_T O[32][14][14], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv4_block2_concat(DATA_T I1[288][14][14], DATA_T I2[32][14][14], DATA_T O[320][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+288; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=288;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv4_block3_0_bn(DATA_T I[320][14][14], DATA_T O[320][14][14], DATA_T W[4][320]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 320; m++){
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
void SW_conv4_block3_0_relu(DATA_T I[320][14][14], DATA_T O[320][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<320; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block3_1_conv(DATA_T I[320][14][14], DATA_T O[128][14][14], DATA_T W[128][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_conv4_block3_1_bn(DATA_T I[128][14][14], DATA_T O[128][14][14], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv4_block3_1_relu(DATA_T I[128][14][14], DATA_T O[128][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block3_2_conv(DATA_T I[128][14][14], DATA_T O[32][14][14], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv4_block3_concat(DATA_T I1[320][14][14], DATA_T I2[32][14][14], DATA_T O[352][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+320; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=320;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv4_block4_0_bn(DATA_T I[352][14][14], DATA_T O[352][14][14], DATA_T W[4][352]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 352; m++){
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
void SW_conv4_block4_0_relu(DATA_T I[352][14][14], DATA_T O[352][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<352; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block4_1_conv(DATA_T I[352][14][14], DATA_T O[128][14][14], DATA_T W[128][352][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<352; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_conv4_block4_1_bn(DATA_T I[128][14][14], DATA_T O[128][14][14], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv4_block4_1_relu(DATA_T I[128][14][14], DATA_T O[128][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block4_2_conv(DATA_T I[128][14][14], DATA_T O[32][14][14], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv4_block4_concat(DATA_T I1[352][14][14], DATA_T I2[32][14][14], DATA_T O[384][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+352; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=352;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv4_block5_0_bn(DATA_T I[384][14][14], DATA_T O[384][14][14], DATA_T W[4][384]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 384; m++){
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
void SW_conv4_block5_0_relu(DATA_T I[384][14][14], DATA_T O[384][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<384; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block5_1_conv(DATA_T I[384][14][14], DATA_T O[128][14][14], DATA_T W[128][384][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_conv4_block5_1_bn(DATA_T I[128][14][14], DATA_T O[128][14][14], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv4_block5_1_relu(DATA_T I[128][14][14], DATA_T O[128][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block5_2_conv(DATA_T I[128][14][14], DATA_T O[32][14][14], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv4_block5_concat(DATA_T I1[384][14][14], DATA_T I2[32][14][14], DATA_T O[416][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+384; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=384;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv4_block6_0_bn(DATA_T I[416][14][14], DATA_T O[416][14][14], DATA_T W[4][416]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 416; m++){
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
void SW_conv4_block6_0_relu(DATA_T I[416][14][14], DATA_T O[416][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<416; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block6_1_conv(DATA_T I[416][14][14], DATA_T O[128][14][14], DATA_T W[128][416][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<416; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_conv4_block6_1_bn(DATA_T I[128][14][14], DATA_T O[128][14][14], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv4_block6_1_relu(DATA_T I[128][14][14], DATA_T O[128][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block6_2_conv(DATA_T I[128][14][14], DATA_T O[32][14][14], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv4_block6_concat(DATA_T I1[416][14][14], DATA_T I2[32][14][14], DATA_T O[448][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+416; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=416;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv4_block7_0_bn(DATA_T I[448][14][14], DATA_T O[448][14][14], DATA_T W[4][448]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 448; m++){
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
void SW_conv4_block7_0_relu(DATA_T I[448][14][14], DATA_T O[448][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<448; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block7_1_conv(DATA_T I[448][14][14], DATA_T O[128][14][14], DATA_T W[128][448][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<448; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_conv4_block7_1_bn(DATA_T I[128][14][14], DATA_T O[128][14][14], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv4_block7_1_relu(DATA_T I[128][14][14], DATA_T O[128][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block7_2_conv(DATA_T I[128][14][14], DATA_T O[32][14][14], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv4_block7_concat(DATA_T I1[448][14][14], DATA_T I2[32][14][14], DATA_T O[480][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+448; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=448;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv4_block8_0_bn(DATA_T I[480][14][14], DATA_T O[480][14][14], DATA_T W[4][480]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 480; m++){
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
void SW_conv4_block8_0_relu(DATA_T I[480][14][14], DATA_T O[480][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<480; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block8_1_conv(DATA_T I[480][14][14], DATA_T O[128][14][14], DATA_T W[128][480][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<480; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_conv4_block8_1_bn(DATA_T I[128][14][14], DATA_T O[128][14][14], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv4_block8_1_relu(DATA_T I[128][14][14], DATA_T O[128][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block8_2_conv(DATA_T I[128][14][14], DATA_T O[32][14][14], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv4_block8_concat(DATA_T I1[480][14][14], DATA_T I2[32][14][14], DATA_T O[512][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+480; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=480;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv4_block9_0_bn(DATA_T I[512][14][14], DATA_T O[512][14][14], DATA_T W[4][512]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 512; m++){
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
void SW_conv4_block9_0_relu(DATA_T I[512][14][14], DATA_T O[512][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<512; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block9_1_conv(DATA_T I[512][14][14], DATA_T O[128][14][14], DATA_T W[128][512][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<512; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_conv4_block9_1_bn(DATA_T I[128][14][14], DATA_T O[128][14][14], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv4_block9_1_relu(DATA_T I[128][14][14], DATA_T O[128][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block9_2_conv(DATA_T I[128][14][14], DATA_T O[32][14][14], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv4_block9_concat(DATA_T I1[512][14][14], DATA_T I2[32][14][14], DATA_T O[544][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+512; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=512;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv4_block10_0_bn(DATA_T I[544][14][14], DATA_T O[544][14][14], DATA_T W[4][544]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 544; m++){
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
void SW_conv4_block10_0_relu(DATA_T I[544][14][14], DATA_T O[544][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<544; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block10_1_conv(DATA_T I[544][14][14], DATA_T O[128][14][14], DATA_T W[128][544][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<544; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_conv4_block10_1_bn(DATA_T I[128][14][14], DATA_T O[128][14][14], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv4_block10_1_relu(DATA_T I[128][14][14], DATA_T O[128][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block10_2_conv(DATA_T I[128][14][14], DATA_T O[32][14][14], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv4_block10_concat(DATA_T I1[544][14][14], DATA_T I2[32][14][14], DATA_T O[576][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+544; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=544;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv4_block11_0_bn(DATA_T I[576][14][14], DATA_T O[576][14][14], DATA_T W[4][576]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 576; m++){
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
void SW_conv4_block11_0_relu(DATA_T I[576][14][14], DATA_T O[576][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<576; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block11_1_conv(DATA_T I[576][14][14], DATA_T O[128][14][14], DATA_T W[128][576][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<576; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_conv4_block11_1_bn(DATA_T I[128][14][14], DATA_T O[128][14][14], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv4_block11_1_relu(DATA_T I[128][14][14], DATA_T O[128][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block11_2_conv(DATA_T I[128][14][14], DATA_T O[32][14][14], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv4_block11_concat(DATA_T I1[576][14][14], DATA_T I2[32][14][14], DATA_T O[608][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+576; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=576;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv4_block12_0_bn(DATA_T I[608][14][14], DATA_T O[608][14][14], DATA_T W[4][608]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 608; m++){
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
void SW_conv4_block12_0_relu(DATA_T I[608][14][14], DATA_T O[608][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<608; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block12_1_conv(DATA_T I[608][14][14], DATA_T O[128][14][14], DATA_T W[128][608][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<608; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_conv4_block12_1_bn(DATA_T I[128][14][14], DATA_T O[128][14][14], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv4_block12_1_relu(DATA_T I[128][14][14], DATA_T O[128][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block12_2_conv(DATA_T I[128][14][14], DATA_T O[32][14][14], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv4_block12_concat(DATA_T I1[608][14][14], DATA_T I2[32][14][14], DATA_T O[640][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+608; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=608;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv4_block13_0_bn(DATA_T I[640][14][14], DATA_T O[640][14][14], DATA_T W[4][640]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 640; m++){
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
void SW_conv4_block13_0_relu(DATA_T I[640][14][14], DATA_T O[640][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<640; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block13_1_conv(DATA_T I[640][14][14], DATA_T O[128][14][14], DATA_T W[128][640][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<640; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_conv4_block13_1_bn(DATA_T I[128][14][14], DATA_T O[128][14][14], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv4_block13_1_relu(DATA_T I[128][14][14], DATA_T O[128][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block13_2_conv(DATA_T I[128][14][14], DATA_T O[32][14][14], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv4_block13_concat(DATA_T I1[640][14][14], DATA_T I2[32][14][14], DATA_T O[672][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+640; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=640;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv4_block14_0_bn(DATA_T I[672][14][14], DATA_T O[672][14][14], DATA_T W[4][672]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 672; m++){
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
void SW_conv4_block14_0_relu(DATA_T I[672][14][14], DATA_T O[672][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<672; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block14_1_conv(DATA_T I[672][14][14], DATA_T O[128][14][14], DATA_T W[128][672][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<672; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_conv4_block14_1_bn(DATA_T I[128][14][14], DATA_T O[128][14][14], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv4_block14_1_relu(DATA_T I[128][14][14], DATA_T O[128][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block14_2_conv(DATA_T I[128][14][14], DATA_T O[32][14][14], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv4_block14_concat(DATA_T I1[672][14][14], DATA_T I2[32][14][14], DATA_T O[704][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+672; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=672;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv4_block15_0_bn(DATA_T I[704][14][14], DATA_T O[704][14][14], DATA_T W[4][704]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 704; m++){
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
void SW_conv4_block15_0_relu(DATA_T I[704][14][14], DATA_T O[704][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<704; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block15_1_conv(DATA_T I[704][14][14], DATA_T O[128][14][14], DATA_T W[128][704][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<704; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_conv4_block15_1_bn(DATA_T I[128][14][14], DATA_T O[128][14][14], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv4_block15_1_relu(DATA_T I[128][14][14], DATA_T O[128][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block15_2_conv(DATA_T I[128][14][14], DATA_T O[32][14][14], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv4_block15_concat(DATA_T I1[704][14][14], DATA_T I2[32][14][14], DATA_T O[736][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+704; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=704;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv4_block16_0_bn(DATA_T I[736][14][14], DATA_T O[736][14][14], DATA_T W[4][736]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 736; m++){
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
void SW_conv4_block16_0_relu(DATA_T I[736][14][14], DATA_T O[736][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<736; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block16_1_conv(DATA_T I[736][14][14], DATA_T O[128][14][14], DATA_T W[128][736][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<736; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_conv4_block16_1_bn(DATA_T I[128][14][14], DATA_T O[128][14][14], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv4_block16_1_relu(DATA_T I[128][14][14], DATA_T O[128][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block16_2_conv(DATA_T I[128][14][14], DATA_T O[32][14][14], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv4_block16_concat(DATA_T I1[736][14][14], DATA_T I2[32][14][14], DATA_T O[768][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+736; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=736;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv4_block17_0_bn(DATA_T I[768][14][14], DATA_T O[768][14][14], DATA_T W[4][768]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 768; m++){
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
void SW_conv4_block17_0_relu(DATA_T I[768][14][14], DATA_T O[768][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<768; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block17_1_conv(DATA_T I[768][14][14], DATA_T O[128][14][14], DATA_T W[128][768][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<768; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_conv4_block17_1_bn(DATA_T I[128][14][14], DATA_T O[128][14][14], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv4_block17_1_relu(DATA_T I[128][14][14], DATA_T O[128][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block17_2_conv(DATA_T I[128][14][14], DATA_T O[32][14][14], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv4_block17_concat(DATA_T I1[768][14][14], DATA_T I2[32][14][14], DATA_T O[800][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+768; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=768;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv4_block18_0_bn(DATA_T I[800][14][14], DATA_T O[800][14][14], DATA_T W[4][800]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 800; m++){
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
void SW_conv4_block18_0_relu(DATA_T I[800][14][14], DATA_T O[800][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<800; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block18_1_conv(DATA_T I[800][14][14], DATA_T O[128][14][14], DATA_T W[128][800][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<800; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_conv4_block18_1_bn(DATA_T I[128][14][14], DATA_T O[128][14][14], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv4_block18_1_relu(DATA_T I[128][14][14], DATA_T O[128][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block18_2_conv(DATA_T I[128][14][14], DATA_T O[32][14][14], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv4_block18_concat(DATA_T I1[800][14][14], DATA_T I2[32][14][14], DATA_T O[832][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+800; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=800;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv4_block19_0_bn(DATA_T I[832][14][14], DATA_T O[832][14][14], DATA_T W[4][832]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 832; m++){
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
void SW_conv4_block19_0_relu(DATA_T I[832][14][14], DATA_T O[832][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<832; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block19_1_conv(DATA_T I[832][14][14], DATA_T O[128][14][14], DATA_T W[128][832][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<832; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_conv4_block19_1_bn(DATA_T I[128][14][14], DATA_T O[128][14][14], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv4_block19_1_relu(DATA_T I[128][14][14], DATA_T O[128][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block19_2_conv(DATA_T I[128][14][14], DATA_T O[32][14][14], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv4_block19_concat(DATA_T I1[832][14][14], DATA_T I2[32][14][14], DATA_T O[864][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+832; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=832;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv4_block20_0_bn(DATA_T I[864][14][14], DATA_T O[864][14][14], DATA_T W[4][864]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 864; m++){
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
void SW_conv4_block20_0_relu(DATA_T I[864][14][14], DATA_T O[864][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<864; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block20_1_conv(DATA_T I[864][14][14], DATA_T O[128][14][14], DATA_T W[128][864][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<864; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_conv4_block20_1_bn(DATA_T I[128][14][14], DATA_T O[128][14][14], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv4_block20_1_relu(DATA_T I[128][14][14], DATA_T O[128][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block20_2_conv(DATA_T I[128][14][14], DATA_T O[32][14][14], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv4_block20_concat(DATA_T I1[864][14][14], DATA_T I2[32][14][14], DATA_T O[896][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+864; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=864;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv4_block21_0_bn(DATA_T I[896][14][14], DATA_T O[896][14][14], DATA_T W[4][896]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 896; m++){
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
void SW_conv4_block21_0_relu(DATA_T I[896][14][14], DATA_T O[896][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<896; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block21_1_conv(DATA_T I[896][14][14], DATA_T O[128][14][14], DATA_T W[128][896][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<896; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_conv4_block21_1_bn(DATA_T I[128][14][14], DATA_T O[128][14][14], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv4_block21_1_relu(DATA_T I[128][14][14], DATA_T O[128][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block21_2_conv(DATA_T I[128][14][14], DATA_T O[32][14][14], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv4_block21_concat(DATA_T I1[896][14][14], DATA_T I2[32][14][14], DATA_T O[928][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+896; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=896;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv4_block22_0_bn(DATA_T I[928][14][14], DATA_T O[928][14][14], DATA_T W[4][928]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 928; m++){
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
void SW_conv4_block22_0_relu(DATA_T I[928][14][14], DATA_T O[928][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<928; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block22_1_conv(DATA_T I[928][14][14], DATA_T O[128][14][14], DATA_T W[128][928][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<928; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_conv4_block22_1_bn(DATA_T I[128][14][14], DATA_T O[128][14][14], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv4_block22_1_relu(DATA_T I[128][14][14], DATA_T O[128][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block22_2_conv(DATA_T I[128][14][14], DATA_T O[32][14][14], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv4_block22_concat(DATA_T I1[928][14][14], DATA_T I2[32][14][14], DATA_T O[960][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+928; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=928;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv4_block23_0_bn(DATA_T I[960][14][14], DATA_T O[960][14][14], DATA_T W[4][960]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 960; m++){
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
void SW_conv4_block23_0_relu(DATA_T I[960][14][14], DATA_T O[960][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<960; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block23_1_conv(DATA_T I[960][14][14], DATA_T O[128][14][14], DATA_T W[128][960][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<960; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_conv4_block23_1_bn(DATA_T I[128][14][14], DATA_T O[128][14][14], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv4_block23_1_relu(DATA_T I[128][14][14], DATA_T O[128][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block23_2_conv(DATA_T I[128][14][14], DATA_T O[32][14][14], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv4_block23_concat(DATA_T I1[960][14][14], DATA_T I2[32][14][14], DATA_T O[992][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+960; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=960;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv4_block24_0_bn(DATA_T I[992][14][14], DATA_T O[992][14][14], DATA_T W[4][992]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 992; m++){
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
void SW_conv4_block24_0_relu(DATA_T I[992][14][14], DATA_T O[992][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<992; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block24_1_conv(DATA_T I[992][14][14], DATA_T O[128][14][14], DATA_T W[128][992][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<992; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_conv4_block24_1_bn(DATA_T I[128][14][14], DATA_T O[128][14][14], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv4_block24_1_relu(DATA_T I[128][14][14], DATA_T O[128][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block24_2_conv(DATA_T I[128][14][14], DATA_T O[32][14][14], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv4_block24_concat(DATA_T I1[992][14][14], DATA_T I2[32][14][14], DATA_T O[1024][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+992; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=992;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv4_block25_0_bn(DATA_T I[1024][14][14], DATA_T O[1024][14][14], DATA_T W[4][1024]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

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
void SW_conv4_block25_0_relu(DATA_T I[1024][14][14], DATA_T O[1024][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1024; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block25_1_conv(DATA_T I[1024][14][14], DATA_T O[128][14][14], DATA_T W[128][1024][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_conv4_block25_1_bn(DATA_T I[128][14][14], DATA_T O[128][14][14], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv4_block25_1_relu(DATA_T I[128][14][14], DATA_T O[128][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block25_2_conv(DATA_T I[128][14][14], DATA_T O[32][14][14], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv4_block25_concat(DATA_T I1[1024][14][14], DATA_T I2[32][14][14], DATA_T O[1056][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+1024; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=1024;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv4_block26_0_bn(DATA_T I[1056][14][14], DATA_T O[1056][14][14], DATA_T W[4][1056]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 1056; m++){
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
void SW_conv4_block26_0_relu(DATA_T I[1056][14][14], DATA_T O[1056][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1056; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block26_1_conv(DATA_T I[1056][14][14], DATA_T O[128][14][14], DATA_T W[128][1056][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<1056; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_conv4_block26_1_bn(DATA_T I[128][14][14], DATA_T O[128][14][14], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv4_block26_1_relu(DATA_T I[128][14][14], DATA_T O[128][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block26_2_conv(DATA_T I[128][14][14], DATA_T O[32][14][14], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv4_block26_concat(DATA_T I1[1056][14][14], DATA_T I2[32][14][14], DATA_T O[1088][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+1056; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=1056;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv4_block27_0_bn(DATA_T I[1088][14][14], DATA_T O[1088][14][14], DATA_T W[4][1088]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 1088; m++){
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
void SW_conv4_block27_0_relu(DATA_T I[1088][14][14], DATA_T O[1088][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block27_1_conv(DATA_T I[1088][14][14], DATA_T O[128][14][14], DATA_T W[128][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_conv4_block27_1_bn(DATA_T I[128][14][14], DATA_T O[128][14][14], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv4_block27_1_relu(DATA_T I[128][14][14], DATA_T O[128][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block27_2_conv(DATA_T I[128][14][14], DATA_T O[32][14][14], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv4_block27_concat(DATA_T I1[1088][14][14], DATA_T I2[32][14][14], DATA_T O[1120][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+1088; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=1088;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv4_block28_0_bn(DATA_T I[1120][14][14], DATA_T O[1120][14][14], DATA_T W[4][1120]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 1120; m++){
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
void SW_conv4_block28_0_relu(DATA_T I[1120][14][14], DATA_T O[1120][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1120; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block28_1_conv(DATA_T I[1120][14][14], DATA_T O[128][14][14], DATA_T W[128][1120][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<1120; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_conv4_block28_1_bn(DATA_T I[128][14][14], DATA_T O[128][14][14], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv4_block28_1_relu(DATA_T I[128][14][14], DATA_T O[128][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block28_2_conv(DATA_T I[128][14][14], DATA_T O[32][14][14], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv4_block28_concat(DATA_T I1[1120][14][14], DATA_T I2[32][14][14], DATA_T O[1152][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+1120; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=1120;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv4_block29_0_bn(DATA_T I[1152][14][14], DATA_T O[1152][14][14], DATA_T W[4][1152]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 1152; m++){
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
void SW_conv4_block29_0_relu(DATA_T I[1152][14][14], DATA_T O[1152][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1152; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block29_1_conv(DATA_T I[1152][14][14], DATA_T O[128][14][14], DATA_T W[128][1152][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<1152; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_conv4_block29_1_bn(DATA_T I[128][14][14], DATA_T O[128][14][14], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv4_block29_1_relu(DATA_T I[128][14][14], DATA_T O[128][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block29_2_conv(DATA_T I[128][14][14], DATA_T O[32][14][14], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv4_block29_concat(DATA_T I1[1152][14][14], DATA_T I2[32][14][14], DATA_T O[1184][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+1152; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=1152;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv4_block30_0_bn(DATA_T I[1184][14][14], DATA_T O[1184][14][14], DATA_T W[4][1184]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 1184; m++){
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
void SW_conv4_block30_0_relu(DATA_T I[1184][14][14], DATA_T O[1184][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1184; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block30_1_conv(DATA_T I[1184][14][14], DATA_T O[128][14][14], DATA_T W[128][1184][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<1184; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_conv4_block30_1_bn(DATA_T I[128][14][14], DATA_T O[128][14][14], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv4_block30_1_relu(DATA_T I[128][14][14], DATA_T O[128][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block30_2_conv(DATA_T I[128][14][14], DATA_T O[32][14][14], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv4_block30_concat(DATA_T I1[1184][14][14], DATA_T I2[32][14][14], DATA_T O[1216][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+1184; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=1184;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv4_block31_0_bn(DATA_T I[1216][14][14], DATA_T O[1216][14][14], DATA_T W[4][1216]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 1216; m++){
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
void SW_conv4_block31_0_relu(DATA_T I[1216][14][14], DATA_T O[1216][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1216; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block31_1_conv(DATA_T I[1216][14][14], DATA_T O[128][14][14], DATA_T W[128][1216][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<1216; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_conv4_block31_1_bn(DATA_T I[128][14][14], DATA_T O[128][14][14], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv4_block31_1_relu(DATA_T I[128][14][14], DATA_T O[128][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block31_2_conv(DATA_T I[128][14][14], DATA_T O[32][14][14], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv4_block31_concat(DATA_T I1[1216][14][14], DATA_T I2[32][14][14], DATA_T O[1248][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+1216; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=1216;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv4_block32_0_bn(DATA_T I[1248][14][14], DATA_T O[1248][14][14], DATA_T W[4][1248]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 1248; m++){
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
void SW_conv4_block32_0_relu(DATA_T I[1248][14][14], DATA_T O[1248][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1248; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block32_1_conv(DATA_T I[1248][14][14], DATA_T O[128][14][14], DATA_T W[128][1248][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<1248; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_conv4_block32_1_bn(DATA_T I[128][14][14], DATA_T O[128][14][14], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv4_block32_1_relu(DATA_T I[128][14][14], DATA_T O[128][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv4_block32_2_conv(DATA_T I[128][14][14], DATA_T O[32][14][14], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv4_block32_concat(DATA_T I1[1248][14][14], DATA_T I2[32][14][14], DATA_T O[1280][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+1248; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=1248;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_pool4_bn(DATA_T I[1280][14][14], DATA_T O[1280][14][14], DATA_T W[4][1280]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 1280; m++){
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
void SW_pool4_relu(DATA_T I[1280][14][14], DATA_T O[1280][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1280; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_pool4_conv(DATA_T I[1280][14][14], DATA_T O[640][14][14], DATA_T W[640][1280][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<640; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<1280; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 14 && y + j <= 14) {
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

void SW_pool4_pool(DATA_T I[640][14][14], DATA_T O[640][7][7])
{
	int m, x, y, i, j;
	DATA_T sum;
	int p = (2*(7-1) - 14 + 2)/2;
    DATA_T div;
	for (m = 0; m<640; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
                sum = (DATA_T)0;
                if(x!=0 && x!=7-1 && y!=0 && y!=7-1)
                    div = 9;
                else if((x==0 || x==7-1) && (y==0 || y==7-1))
                    div = 4;
                else
                    div = 6;
				for (i = 0; i<2; i++) {
					for (j = 0; j<2; j++) {

						if (x + i < 14 + p && y + j < 14 + p && x + i -p >= 0 && y + j -p >= 0)
						    sum += I[m][x*2 + i -p][y*2 + j -p];
					}
				}
				O[m][x][y] = sum/div;
			}
		}
	}
}
void SW_conv5_block1_0_bn(DATA_T I[640][7][7], DATA_T O[640][7][7], DATA_T W[4][640]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 640; m++){
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
void SW_conv5_block1_0_relu(DATA_T I[640][7][7], DATA_T O[640][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<640; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block1_1_conv(DATA_T I[640][7][7], DATA_T O[128][7][7], DATA_T W[128][640][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<640; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_conv5_block1_1_bn(DATA_T I[128][7][7], DATA_T O[128][7][7], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv5_block1_1_relu(DATA_T I[128][7][7], DATA_T O[128][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block1_2_conv(DATA_T I[128][7][7], DATA_T O[32][7][7], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 7 + p && y*1 + j < 7 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv5_block1_concat(DATA_T I1[640][7][7], DATA_T I2[32][7][7], DATA_T O[672][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+640; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=640;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv5_block2_0_bn(DATA_T I[672][7][7], DATA_T O[672][7][7], DATA_T W[4][672]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 672; m++){
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
void SW_conv5_block2_0_relu(DATA_T I[672][7][7], DATA_T O[672][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<672; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block2_1_conv(DATA_T I[672][7][7], DATA_T O[128][7][7], DATA_T W[128][672][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<672; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_conv5_block2_1_bn(DATA_T I[128][7][7], DATA_T O[128][7][7], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv5_block2_1_relu(DATA_T I[128][7][7], DATA_T O[128][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block2_2_conv(DATA_T I[128][7][7], DATA_T O[32][7][7], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 7 + p && y*1 + j < 7 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv5_block2_concat(DATA_T I1[672][7][7], DATA_T I2[32][7][7], DATA_T O[704][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+672; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=672;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv5_block3_0_bn(DATA_T I[704][7][7], DATA_T O[704][7][7], DATA_T W[4][704]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 704; m++){
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
void SW_conv5_block3_0_relu(DATA_T I[704][7][7], DATA_T O[704][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<704; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block3_1_conv(DATA_T I[704][7][7], DATA_T O[128][7][7], DATA_T W[128][704][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<704; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_conv5_block3_1_bn(DATA_T I[128][7][7], DATA_T O[128][7][7], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv5_block3_1_relu(DATA_T I[128][7][7], DATA_T O[128][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block3_2_conv(DATA_T I[128][7][7], DATA_T O[32][7][7], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 7 + p && y*1 + j < 7 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv5_block3_concat(DATA_T I1[704][7][7], DATA_T I2[32][7][7], DATA_T O[736][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+704; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=704;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv5_block4_0_bn(DATA_T I[736][7][7], DATA_T O[736][7][7], DATA_T W[4][736]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 736; m++){
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
void SW_conv5_block4_0_relu(DATA_T I[736][7][7], DATA_T O[736][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<736; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block4_1_conv(DATA_T I[736][7][7], DATA_T O[128][7][7], DATA_T W[128][736][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<736; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_conv5_block4_1_bn(DATA_T I[128][7][7], DATA_T O[128][7][7], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv5_block4_1_relu(DATA_T I[128][7][7], DATA_T O[128][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block4_2_conv(DATA_T I[128][7][7], DATA_T O[32][7][7], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 7 + p && y*1 + j < 7 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv5_block4_concat(DATA_T I1[736][7][7], DATA_T I2[32][7][7], DATA_T O[768][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+736; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=736;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv5_block5_0_bn(DATA_T I[768][7][7], DATA_T O[768][7][7], DATA_T W[4][768]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 768; m++){
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
void SW_conv5_block5_0_relu(DATA_T I[768][7][7], DATA_T O[768][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<768; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block5_1_conv(DATA_T I[768][7][7], DATA_T O[128][7][7], DATA_T W[128][768][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<768; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_conv5_block5_1_bn(DATA_T I[128][7][7], DATA_T O[128][7][7], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv5_block5_1_relu(DATA_T I[128][7][7], DATA_T O[128][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block5_2_conv(DATA_T I[128][7][7], DATA_T O[32][7][7], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 7 + p && y*1 + j < 7 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv5_block5_concat(DATA_T I1[768][7][7], DATA_T I2[32][7][7], DATA_T O[800][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+768; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=768;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv5_block6_0_bn(DATA_T I[800][7][7], DATA_T O[800][7][7], DATA_T W[4][800]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 800; m++){
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
void SW_conv5_block6_0_relu(DATA_T I[800][7][7], DATA_T O[800][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<800; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block6_1_conv(DATA_T I[800][7][7], DATA_T O[128][7][7], DATA_T W[128][800][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<800; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_conv5_block6_1_bn(DATA_T I[128][7][7], DATA_T O[128][7][7], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv5_block6_1_relu(DATA_T I[128][7][7], DATA_T O[128][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block6_2_conv(DATA_T I[128][7][7], DATA_T O[32][7][7], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 7 + p && y*1 + j < 7 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv5_block6_concat(DATA_T I1[800][7][7], DATA_T I2[32][7][7], DATA_T O[832][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+800; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=800;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv5_block7_0_bn(DATA_T I[832][7][7], DATA_T O[832][7][7], DATA_T W[4][832]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 832; m++){
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
void SW_conv5_block7_0_relu(DATA_T I[832][7][7], DATA_T O[832][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<832; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block7_1_conv(DATA_T I[832][7][7], DATA_T O[128][7][7], DATA_T W[128][832][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<832; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_conv5_block7_1_bn(DATA_T I[128][7][7], DATA_T O[128][7][7], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv5_block7_1_relu(DATA_T I[128][7][7], DATA_T O[128][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block7_2_conv(DATA_T I[128][7][7], DATA_T O[32][7][7], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 7 + p && y*1 + j < 7 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv5_block7_concat(DATA_T I1[832][7][7], DATA_T I2[32][7][7], DATA_T O[864][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+832; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=832;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv5_block8_0_bn(DATA_T I[864][7][7], DATA_T O[864][7][7], DATA_T W[4][864]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 864; m++){
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
void SW_conv5_block8_0_relu(DATA_T I[864][7][7], DATA_T O[864][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<864; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block8_1_conv(DATA_T I[864][7][7], DATA_T O[128][7][7], DATA_T W[128][864][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<864; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_conv5_block8_1_bn(DATA_T I[128][7][7], DATA_T O[128][7][7], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv5_block8_1_relu(DATA_T I[128][7][7], DATA_T O[128][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block8_2_conv(DATA_T I[128][7][7], DATA_T O[32][7][7], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 7 + p && y*1 + j < 7 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv5_block8_concat(DATA_T I1[864][7][7], DATA_T I2[32][7][7], DATA_T O[896][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+864; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=864;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv5_block9_0_bn(DATA_T I[896][7][7], DATA_T O[896][7][7], DATA_T W[4][896]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 896; m++){
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
void SW_conv5_block9_0_relu(DATA_T I[896][7][7], DATA_T O[896][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<896; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block9_1_conv(DATA_T I[896][7][7], DATA_T O[128][7][7], DATA_T W[128][896][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<896; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_conv5_block9_1_bn(DATA_T I[128][7][7], DATA_T O[128][7][7], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv5_block9_1_relu(DATA_T I[128][7][7], DATA_T O[128][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block9_2_conv(DATA_T I[128][7][7], DATA_T O[32][7][7], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 7 + p && y*1 + j < 7 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv5_block9_concat(DATA_T I1[896][7][7], DATA_T I2[32][7][7], DATA_T O[928][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+896; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=896;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv5_block10_0_bn(DATA_T I[928][7][7], DATA_T O[928][7][7], DATA_T W[4][928]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 928; m++){
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
void SW_conv5_block10_0_relu(DATA_T I[928][7][7], DATA_T O[928][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<928; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block10_1_conv(DATA_T I[928][7][7], DATA_T O[128][7][7], DATA_T W[128][928][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<928; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_conv5_block10_1_bn(DATA_T I[128][7][7], DATA_T O[128][7][7], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv5_block10_1_relu(DATA_T I[128][7][7], DATA_T O[128][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block10_2_conv(DATA_T I[128][7][7], DATA_T O[32][7][7], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 7 + p && y*1 + j < 7 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv5_block10_concat(DATA_T I1[928][7][7], DATA_T I2[32][7][7], DATA_T O[960][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+928; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=928;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv5_block11_0_bn(DATA_T I[960][7][7], DATA_T O[960][7][7], DATA_T W[4][960]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 960; m++){
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
void SW_conv5_block11_0_relu(DATA_T I[960][7][7], DATA_T O[960][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<960; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block11_1_conv(DATA_T I[960][7][7], DATA_T O[128][7][7], DATA_T W[128][960][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<960; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_conv5_block11_1_bn(DATA_T I[128][7][7], DATA_T O[128][7][7], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv5_block11_1_relu(DATA_T I[128][7][7], DATA_T O[128][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block11_2_conv(DATA_T I[128][7][7], DATA_T O[32][7][7], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 7 + p && y*1 + j < 7 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv5_block11_concat(DATA_T I1[960][7][7], DATA_T I2[32][7][7], DATA_T O[992][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+960; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=960;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv5_block12_0_bn(DATA_T I[992][7][7], DATA_T O[992][7][7], DATA_T W[4][992]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 992; m++){
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
void SW_conv5_block12_0_relu(DATA_T I[992][7][7], DATA_T O[992][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<992; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block12_1_conv(DATA_T I[992][7][7], DATA_T O[128][7][7], DATA_T W[128][992][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<992; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_conv5_block12_1_bn(DATA_T I[128][7][7], DATA_T O[128][7][7], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv5_block12_1_relu(DATA_T I[128][7][7], DATA_T O[128][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block12_2_conv(DATA_T I[128][7][7], DATA_T O[32][7][7], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 7 + p && y*1 + j < 7 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv5_block12_concat(DATA_T I1[992][7][7], DATA_T I2[32][7][7], DATA_T O[1024][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+992; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=992;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv5_block13_0_bn(DATA_T I[1024][7][7], DATA_T O[1024][7][7], DATA_T W[4][1024]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 1024; m++){
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
void SW_conv5_block13_0_relu(DATA_T I[1024][7][7], DATA_T O[1024][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1024; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block13_1_conv(DATA_T I[1024][7][7], DATA_T O[128][7][7], DATA_T W[128][1024][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<1024; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_conv5_block13_1_bn(DATA_T I[128][7][7], DATA_T O[128][7][7], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv5_block13_1_relu(DATA_T I[128][7][7], DATA_T O[128][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block13_2_conv(DATA_T I[128][7][7], DATA_T O[32][7][7], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 7 + p && y*1 + j < 7 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv5_block13_concat(DATA_T I1[1024][7][7], DATA_T I2[32][7][7], DATA_T O[1056][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+1024; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=1024;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv5_block14_0_bn(DATA_T I[1056][7][7], DATA_T O[1056][7][7], DATA_T W[4][1056]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 1056; m++){
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
void SW_conv5_block14_0_relu(DATA_T I[1056][7][7], DATA_T O[1056][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1056; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block14_1_conv(DATA_T I[1056][7][7], DATA_T O[128][7][7], DATA_T W[128][1056][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<1056; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_conv5_block14_1_bn(DATA_T I[128][7][7], DATA_T O[128][7][7], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv5_block14_1_relu(DATA_T I[128][7][7], DATA_T O[128][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block14_2_conv(DATA_T I[128][7][7], DATA_T O[32][7][7], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 7 + p && y*1 + j < 7 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv5_block14_concat(DATA_T I1[1056][7][7], DATA_T I2[32][7][7], DATA_T O[1088][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+1056; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=1056;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv5_block15_0_bn(DATA_T I[1088][7][7], DATA_T O[1088][7][7], DATA_T W[4][1088]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 1088; m++){
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
void SW_conv5_block15_0_relu(DATA_T I[1088][7][7], DATA_T O[1088][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1088; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block15_1_conv(DATA_T I[1088][7][7], DATA_T O[128][7][7], DATA_T W[128][1088][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<1088; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_conv5_block15_1_bn(DATA_T I[128][7][7], DATA_T O[128][7][7], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv5_block15_1_relu(DATA_T I[128][7][7], DATA_T O[128][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block15_2_conv(DATA_T I[128][7][7], DATA_T O[32][7][7], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 7 + p && y*1 + j < 7 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv5_block15_concat(DATA_T I1[1088][7][7], DATA_T I2[32][7][7], DATA_T O[1120][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+1088; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=1088;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv5_block16_0_bn(DATA_T I[1120][7][7], DATA_T O[1120][7][7], DATA_T W[4][1120]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 1120; m++){
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
void SW_conv5_block16_0_relu(DATA_T I[1120][7][7], DATA_T O[1120][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1120; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block16_1_conv(DATA_T I[1120][7][7], DATA_T O[128][7][7], DATA_T W[128][1120][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<1120; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_conv5_block16_1_bn(DATA_T I[128][7][7], DATA_T O[128][7][7], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv5_block16_1_relu(DATA_T I[128][7][7], DATA_T O[128][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block16_2_conv(DATA_T I[128][7][7], DATA_T O[32][7][7], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 7 + p && y*1 + j < 7 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv5_block16_concat(DATA_T I1[1120][7][7], DATA_T I2[32][7][7], DATA_T O[1152][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+1120; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=1120;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv5_block17_0_bn(DATA_T I[1152][7][7], DATA_T O[1152][7][7], DATA_T W[4][1152]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 1152; m++){
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
void SW_conv5_block17_0_relu(DATA_T I[1152][7][7], DATA_T O[1152][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1152; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block17_1_conv(DATA_T I[1152][7][7], DATA_T O[128][7][7], DATA_T W[128][1152][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<1152; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_conv5_block17_1_bn(DATA_T I[128][7][7], DATA_T O[128][7][7], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv5_block17_1_relu(DATA_T I[128][7][7], DATA_T O[128][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block17_2_conv(DATA_T I[128][7][7], DATA_T O[32][7][7], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 7 + p && y*1 + j < 7 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv5_block17_concat(DATA_T I1[1152][7][7], DATA_T I2[32][7][7], DATA_T O[1184][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+1152; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=1152;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv5_block18_0_bn(DATA_T I[1184][7][7], DATA_T O[1184][7][7], DATA_T W[4][1184]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 1184; m++){
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
void SW_conv5_block18_0_relu(DATA_T I[1184][7][7], DATA_T O[1184][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1184; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block18_1_conv(DATA_T I[1184][7][7], DATA_T O[128][7][7], DATA_T W[128][1184][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<1184; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_conv5_block18_1_bn(DATA_T I[128][7][7], DATA_T O[128][7][7], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv5_block18_1_relu(DATA_T I[128][7][7], DATA_T O[128][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block18_2_conv(DATA_T I[128][7][7], DATA_T O[32][7][7], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 7 + p && y*1 + j < 7 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv5_block18_concat(DATA_T I1[1184][7][7], DATA_T I2[32][7][7], DATA_T O[1216][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+1184; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=1184;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv5_block19_0_bn(DATA_T I[1216][7][7], DATA_T O[1216][7][7], DATA_T W[4][1216]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 1216; m++){
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
void SW_conv5_block19_0_relu(DATA_T I[1216][7][7], DATA_T O[1216][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1216; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block19_1_conv(DATA_T I[1216][7][7], DATA_T O[128][7][7], DATA_T W[128][1216][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<1216; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_conv5_block19_1_bn(DATA_T I[128][7][7], DATA_T O[128][7][7], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv5_block19_1_relu(DATA_T I[128][7][7], DATA_T O[128][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block19_2_conv(DATA_T I[128][7][7], DATA_T O[32][7][7], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 7 + p && y*1 + j < 7 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv5_block19_concat(DATA_T I1[1216][7][7], DATA_T I2[32][7][7], DATA_T O[1248][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+1216; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=1216;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv5_block20_0_bn(DATA_T I[1248][7][7], DATA_T O[1248][7][7], DATA_T W[4][1248]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 1248; m++){
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
void SW_conv5_block20_0_relu(DATA_T I[1248][7][7], DATA_T O[1248][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1248; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block20_1_conv(DATA_T I[1248][7][7], DATA_T O[128][7][7], DATA_T W[128][1248][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<1248; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_conv5_block20_1_bn(DATA_T I[128][7][7], DATA_T O[128][7][7], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv5_block20_1_relu(DATA_T I[128][7][7], DATA_T O[128][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block20_2_conv(DATA_T I[128][7][7], DATA_T O[32][7][7], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 7 + p && y*1 + j < 7 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv5_block20_concat(DATA_T I1[1248][7][7], DATA_T I2[32][7][7], DATA_T O[1280][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+1248; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=1248;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv5_block21_0_bn(DATA_T I[1280][7][7], DATA_T O[1280][7][7], DATA_T W[4][1280]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 1280; m++){
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
void SW_conv5_block21_0_relu(DATA_T I[1280][7][7], DATA_T O[1280][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1280; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block21_1_conv(DATA_T I[1280][7][7], DATA_T O[128][7][7], DATA_T W[128][1280][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<1280; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_conv5_block21_1_bn(DATA_T I[128][7][7], DATA_T O[128][7][7], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv5_block21_1_relu(DATA_T I[128][7][7], DATA_T O[128][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block21_2_conv(DATA_T I[128][7][7], DATA_T O[32][7][7], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 7 + p && y*1 + j < 7 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv5_block21_concat(DATA_T I1[1280][7][7], DATA_T I2[32][7][7], DATA_T O[1312][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+1280; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=1280;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv5_block22_0_bn(DATA_T I[1312][7][7], DATA_T O[1312][7][7], DATA_T W[4][1312]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 1312; m++){
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
void SW_conv5_block22_0_relu(DATA_T I[1312][7][7], DATA_T O[1312][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1312; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block22_1_conv(DATA_T I[1312][7][7], DATA_T O[128][7][7], DATA_T W[128][1312][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<1312; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_conv5_block22_1_bn(DATA_T I[128][7][7], DATA_T O[128][7][7], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv5_block22_1_relu(DATA_T I[128][7][7], DATA_T O[128][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block22_2_conv(DATA_T I[128][7][7], DATA_T O[32][7][7], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 7 + p && y*1 + j < 7 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv5_block22_concat(DATA_T I1[1312][7][7], DATA_T I2[32][7][7], DATA_T O[1344][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+1312; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=1312;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv5_block23_0_bn(DATA_T I[1344][7][7], DATA_T O[1344][7][7], DATA_T W[4][1344]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 1344; m++){
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
void SW_conv5_block23_0_relu(DATA_T I[1344][7][7], DATA_T O[1344][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1344; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block23_1_conv(DATA_T I[1344][7][7], DATA_T O[128][7][7], DATA_T W[128][1344][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<1344; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_conv5_block23_1_bn(DATA_T I[128][7][7], DATA_T O[128][7][7], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv5_block23_1_relu(DATA_T I[128][7][7], DATA_T O[128][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block23_2_conv(DATA_T I[128][7][7], DATA_T O[32][7][7], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 7 + p && y*1 + j < 7 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv5_block23_concat(DATA_T I1[1344][7][7], DATA_T I2[32][7][7], DATA_T O[1376][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+1344; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=1344;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv5_block24_0_bn(DATA_T I[1376][7][7], DATA_T O[1376][7][7], DATA_T W[4][1376]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 1376; m++){
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
void SW_conv5_block24_0_relu(DATA_T I[1376][7][7], DATA_T O[1376][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1376; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block24_1_conv(DATA_T I[1376][7][7], DATA_T O[128][7][7], DATA_T W[128][1376][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<1376; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_conv5_block24_1_bn(DATA_T I[128][7][7], DATA_T O[128][7][7], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv5_block24_1_relu(DATA_T I[128][7][7], DATA_T O[128][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block24_2_conv(DATA_T I[128][7][7], DATA_T O[32][7][7], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 7 + p && y*1 + j < 7 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv5_block24_concat(DATA_T I1[1376][7][7], DATA_T I2[32][7][7], DATA_T O[1408][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+1376; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=1376;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv5_block25_0_bn(DATA_T I[1408][7][7], DATA_T O[1408][7][7], DATA_T W[4][1408]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 1408; m++){
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
void SW_conv5_block25_0_relu(DATA_T I[1408][7][7], DATA_T O[1408][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1408; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block25_1_conv(DATA_T I[1408][7][7], DATA_T O[128][7][7], DATA_T W[128][1408][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<1408; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_conv5_block25_1_bn(DATA_T I[128][7][7], DATA_T O[128][7][7], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv5_block25_1_relu(DATA_T I[128][7][7], DATA_T O[128][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block25_2_conv(DATA_T I[128][7][7], DATA_T O[32][7][7], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 7 + p && y*1 + j < 7 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv5_block25_concat(DATA_T I1[1408][7][7], DATA_T I2[32][7][7], DATA_T O[1440][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+1408; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=1408;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv5_block26_0_bn(DATA_T I[1440][7][7], DATA_T O[1440][7][7], DATA_T W[4][1440]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 1440; m++){
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
void SW_conv5_block26_0_relu(DATA_T I[1440][7][7], DATA_T O[1440][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1440; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block26_1_conv(DATA_T I[1440][7][7], DATA_T O[128][7][7], DATA_T W[128][1440][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<1440; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_conv5_block26_1_bn(DATA_T I[128][7][7], DATA_T O[128][7][7], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv5_block26_1_relu(DATA_T I[128][7][7], DATA_T O[128][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block26_2_conv(DATA_T I[128][7][7], DATA_T O[32][7][7], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 7 + p && y*1 + j < 7 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv5_block26_concat(DATA_T I1[1440][7][7], DATA_T I2[32][7][7], DATA_T O[1472][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+1440; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=1440;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv5_block27_0_bn(DATA_T I[1472][7][7], DATA_T O[1472][7][7], DATA_T W[4][1472]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 1472; m++){
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
void SW_conv5_block27_0_relu(DATA_T I[1472][7][7], DATA_T O[1472][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1472; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block27_1_conv(DATA_T I[1472][7][7], DATA_T O[128][7][7], DATA_T W[128][1472][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<1472; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_conv5_block27_1_bn(DATA_T I[128][7][7], DATA_T O[128][7][7], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv5_block27_1_relu(DATA_T I[128][7][7], DATA_T O[128][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block27_2_conv(DATA_T I[128][7][7], DATA_T O[32][7][7], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 7 + p && y*1 + j < 7 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv5_block27_concat(DATA_T I1[1472][7][7], DATA_T I2[32][7][7], DATA_T O[1504][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+1472; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=1472;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv5_block28_0_bn(DATA_T I[1504][7][7], DATA_T O[1504][7][7], DATA_T W[4][1504]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 1504; m++){
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
void SW_conv5_block28_0_relu(DATA_T I[1504][7][7], DATA_T O[1504][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1504; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block28_1_conv(DATA_T I[1504][7][7], DATA_T O[128][7][7], DATA_T W[128][1504][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<1504; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_conv5_block28_1_bn(DATA_T I[128][7][7], DATA_T O[128][7][7], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv5_block28_1_relu(DATA_T I[128][7][7], DATA_T O[128][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block28_2_conv(DATA_T I[128][7][7], DATA_T O[32][7][7], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 7 + p && y*1 + j < 7 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv5_block28_concat(DATA_T I1[1504][7][7], DATA_T I2[32][7][7], DATA_T O[1536][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+1504; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=1504;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv5_block29_0_bn(DATA_T I[1536][7][7], DATA_T O[1536][7][7], DATA_T W[4][1536]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 1536; m++){
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
void SW_conv5_block29_0_relu(DATA_T I[1536][7][7], DATA_T O[1536][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1536; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block29_1_conv(DATA_T I[1536][7][7], DATA_T O[128][7][7], DATA_T W[128][1536][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<1536; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_conv5_block29_1_bn(DATA_T I[128][7][7], DATA_T O[128][7][7], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv5_block29_1_relu(DATA_T I[128][7][7], DATA_T O[128][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block29_2_conv(DATA_T I[128][7][7], DATA_T O[32][7][7], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 7 + p && y*1 + j < 7 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv5_block29_concat(DATA_T I1[1536][7][7], DATA_T I2[32][7][7], DATA_T O[1568][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+1536; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=1536;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv5_block30_0_bn(DATA_T I[1568][7][7], DATA_T O[1568][7][7], DATA_T W[4][1568]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 1568; m++){
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
void SW_conv5_block30_0_relu(DATA_T I[1568][7][7], DATA_T O[1568][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1568; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block30_1_conv(DATA_T I[1568][7][7], DATA_T O[128][7][7], DATA_T W[128][1568][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<1568; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_conv5_block30_1_bn(DATA_T I[128][7][7], DATA_T O[128][7][7], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv5_block30_1_relu(DATA_T I[128][7][7], DATA_T O[128][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block30_2_conv(DATA_T I[128][7][7], DATA_T O[32][7][7], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 7 + p && y*1 + j < 7 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv5_block30_concat(DATA_T I1[1568][7][7], DATA_T I2[32][7][7], DATA_T O[1600][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+1568; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=1568;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv5_block31_0_bn(DATA_T I[1600][7][7], DATA_T O[1600][7][7], DATA_T W[4][1600]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 1600; m++){
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
void SW_conv5_block31_0_relu(DATA_T I[1600][7][7], DATA_T O[1600][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1600; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block31_1_conv(DATA_T I[1600][7][7], DATA_T O[128][7][7], DATA_T W[128][1600][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<1600; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_conv5_block31_1_bn(DATA_T I[128][7][7], DATA_T O[128][7][7], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv5_block31_1_relu(DATA_T I[128][7][7], DATA_T O[128][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block31_2_conv(DATA_T I[128][7][7], DATA_T O[32][7][7], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 7 + p && y*1 + j < 7 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv5_block31_concat(DATA_T I1[1600][7][7], DATA_T I2[32][7][7], DATA_T O[1632][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+1600; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=1600;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv5_block32_0_bn(DATA_T I[1632][7][7], DATA_T O[1632][7][7], DATA_T W[4][1632]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 1632; m++){
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
void SW_conv5_block32_0_relu(DATA_T I[1632][7][7], DATA_T O[1632][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<1632; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block32_1_conv(DATA_T I[1632][7][7], DATA_T O[128][7][7], DATA_T W[128][1632][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<1632; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 7 && y + j <= 7) {
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

void SW_conv5_block32_1_bn(DATA_T I[128][7][7], DATA_T O[128][7][7], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 128; m++){
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
void SW_conv5_block32_1_relu(DATA_T I[128][7][7], DATA_T O[128][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv5_block32_2_conv(DATA_T I[128][7][7], DATA_T O[32][7][7], DATA_T W[32][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 7 + p && y*1 + j < 7 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv5_block32_concat(DATA_T I1[1632][7][7], DATA_T I2[32][7][7], DATA_T O[1664][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+1632; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=1632;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_bn(DATA_T I[1664][7][7], DATA_T O[1664][7][7], DATA_T W[4][1664]) {
	int m, x, y;
	DATA_T epsilon = 1.001E-05;

    for (m = 0; m < 1664; m++){
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
void SW_avg_pool(DATA_T I[1664][7][7], DATA_T O[1664]) {
	int m, x, y;
	double avg;
	int div = 7 * 7;
	for (m = 0; m < 1664; m++){
		avg = 0;
		for (x = 0; x < 7; x++) {
			for (y = 0; y < 7; y++) {
				avg += I[m][x][y];
			}
		}
		O[m] = avg/div;
	}
}
void SW_fc1000(DATA_T I[1664], DATA_T O[1000], DATA_T W[1000][1664], DATA_T B[1000])
{
    //Dense
	int m, c;
	for(m=0; m<1000; m++){
        O[m] = 0;
		for (c = 0; c < 1664; c++){
            O[m] += W[m][c] * I[c];
        }
        O[m] += B[m];
        if (O[m] < 0) //Relu
            O[m] = 0;
     }
}


//argv[1] = init_weight.txt, argv[2] = init_input.txt
int main(int argc, char *argv[]){

    DATA_T temp;
    int m, x, y, i, j, k, l;
    int trash;

    static DATA_T I[3][224][224];
	static DATA_T W2[64][3][7][7];
	static DATA_T W3[4][64];
	static DATA_T W7[4][64];
	static DATA_T W9[128][64][1][1];
	static DATA_T W10[4][128];
	static DATA_T W12[32][128][3][3];
	static DATA_T W14[4][96];
	static DATA_T W16[128][96][1][1];
	static DATA_T W17[4][128];
	static DATA_T W19[32][128][3][3];
	static DATA_T W21[4][128];
	static DATA_T W23[128][128][1][1];
	static DATA_T W24[4][128];
	static DATA_T W26[32][128][3][3];
	static DATA_T W28[4][160];
	static DATA_T W30[128][160][1][1];
	static DATA_T W31[4][128];
	static DATA_T W33[32][128][3][3];
	static DATA_T W35[4][192];
	static DATA_T W37[128][192][1][1];
	static DATA_T W38[4][128];
	static DATA_T W40[32][128][3][3];
	static DATA_T W42[4][224];
	static DATA_T W44[128][224][1][1];
	static DATA_T W45[4][128];
	static DATA_T W47[32][128][3][3];
	static DATA_T W49[4][256];
	static DATA_T W51[128][256][1][1];
	static DATA_T W53[4][128];
	static DATA_T W55[128][128][1][1];
	static DATA_T W56[4][128];
	static DATA_T W58[32][128][3][3];
	static DATA_T W60[4][160];
	static DATA_T W62[128][160][1][1];
	static DATA_T W63[4][128];
	static DATA_T W65[32][128][3][3];
	static DATA_T W67[4][192];
	static DATA_T W69[128][192][1][1];
	static DATA_T W70[4][128];
	static DATA_T W72[32][128][3][3];
	static DATA_T W74[4][224];
	static DATA_T W76[128][224][1][1];
	static DATA_T W77[4][128];
	static DATA_T W79[32][128][3][3];
	static DATA_T W81[4][256];
	static DATA_T W83[128][256][1][1];
	static DATA_T W84[4][128];
	static DATA_T W86[32][128][3][3];
	static DATA_T W88[4][288];
	static DATA_T W90[128][288][1][1];
	static DATA_T W91[4][128];
	static DATA_T W93[32][128][3][3];
	static DATA_T W95[4][320];
	static DATA_T W97[128][320][1][1];
	static DATA_T W98[4][128];
	static DATA_T W100[32][128][3][3];
	static DATA_T W102[4][352];
	static DATA_T W104[128][352][1][1];
	static DATA_T W105[4][128];
	static DATA_T W107[32][128][3][3];
	static DATA_T W109[4][384];
	static DATA_T W111[128][384][1][1];
	static DATA_T W112[4][128];
	static DATA_T W114[32][128][3][3];
	static DATA_T W116[4][416];
	static DATA_T W118[128][416][1][1];
	static DATA_T W119[4][128];
	static DATA_T W121[32][128][3][3];
	static DATA_T W123[4][448];
	static DATA_T W125[128][448][1][1];
	static DATA_T W126[4][128];
	static DATA_T W128[32][128][3][3];
	static DATA_T W130[4][480];
	static DATA_T W132[128][480][1][1];
	static DATA_T W133[4][128];
	static DATA_T W135[32][128][3][3];
	static DATA_T W137[4][512];
	static DATA_T W139[256][512][1][1];
	static DATA_T W141[4][256];
	static DATA_T W143[128][256][1][1];
	static DATA_T W144[4][128];
	static DATA_T W146[32][128][3][3];
	static DATA_T W148[4][288];
	static DATA_T W150[128][288][1][1];
	static DATA_T W151[4][128];
	static DATA_T W153[32][128][3][3];
	static DATA_T W155[4][320];
	static DATA_T W157[128][320][1][1];
	static DATA_T W158[4][128];
	static DATA_T W160[32][128][3][3];
	static DATA_T W162[4][352];
	static DATA_T W164[128][352][1][1];
	static DATA_T W165[4][128];
	static DATA_T W167[32][128][3][3];
	static DATA_T W169[4][384];
	static DATA_T W171[128][384][1][1];
	static DATA_T W172[4][128];
	static DATA_T W174[32][128][3][3];
	static DATA_T W176[4][416];
	static DATA_T W178[128][416][1][1];
	static DATA_T W179[4][128];
	static DATA_T W181[32][128][3][3];
	static DATA_T W183[4][448];
	static DATA_T W185[128][448][1][1];
	static DATA_T W186[4][128];
	static DATA_T W188[32][128][3][3];
	static DATA_T W190[4][480];
	static DATA_T W192[128][480][1][1];
	static DATA_T W193[4][128];
	static DATA_T W195[32][128][3][3];
	static DATA_T W197[4][512];
	static DATA_T W199[128][512][1][1];
	static DATA_T W200[4][128];
	static DATA_T W202[32][128][3][3];
	static DATA_T W204[4][544];
	static DATA_T W206[128][544][1][1];
	static DATA_T W207[4][128];
	static DATA_T W209[32][128][3][3];
	static DATA_T W211[4][576];
	static DATA_T W213[128][576][1][1];
	static DATA_T W214[4][128];
	static DATA_T W216[32][128][3][3];
	static DATA_T W218[4][608];
	static DATA_T W220[128][608][1][1];
	static DATA_T W221[4][128];
	static DATA_T W223[32][128][3][3];
	static DATA_T W225[4][640];
	static DATA_T W227[128][640][1][1];
	static DATA_T W228[4][128];
	static DATA_T W230[32][128][3][3];
	static DATA_T W232[4][672];
	static DATA_T W234[128][672][1][1];
	static DATA_T W235[4][128];
	static DATA_T W237[32][128][3][3];
	static DATA_T W239[4][704];
	static DATA_T W241[128][704][1][1];
	static DATA_T W242[4][128];
	static DATA_T W244[32][128][3][3];
	static DATA_T W246[4][736];
	static DATA_T W248[128][736][1][1];
	static DATA_T W249[4][128];
	static DATA_T W251[32][128][3][3];
	static DATA_T W253[4][768];
	static DATA_T W255[128][768][1][1];
	static DATA_T W256[4][128];
	static DATA_T W258[32][128][3][3];
	static DATA_T W260[4][800];
	static DATA_T W262[128][800][1][1];
	static DATA_T W263[4][128];
	static DATA_T W265[32][128][3][3];
	static DATA_T W267[4][832];
	static DATA_T W269[128][832][1][1];
	static DATA_T W270[4][128];
	static DATA_T W272[32][128][3][3];
	static DATA_T W274[4][864];
	static DATA_T W276[128][864][1][1];
	static DATA_T W277[4][128];
	static DATA_T W279[32][128][3][3];
	static DATA_T W281[4][896];
	static DATA_T W283[128][896][1][1];
	static DATA_T W284[4][128];
	static DATA_T W286[32][128][3][3];
	static DATA_T W288[4][928];
	static DATA_T W290[128][928][1][1];
	static DATA_T W291[4][128];
	static DATA_T W293[32][128][3][3];
	static DATA_T W295[4][960];
	static DATA_T W297[128][960][1][1];
	static DATA_T W298[4][128];
	static DATA_T W300[32][128][3][3];
	static DATA_T W302[4][992];
	static DATA_T W304[128][992][1][1];
	static DATA_T W305[4][128];
	static DATA_T W307[32][128][3][3];
	static DATA_T W309[4][1024];
	static DATA_T W311[128][1024][1][1];
	static DATA_T W312[4][128];
	static DATA_T W314[32][128][3][3];
	static DATA_T W316[4][1056];
	static DATA_T W318[128][1056][1][1];
	static DATA_T W319[4][128];
	static DATA_T W321[32][128][3][3];
	static DATA_T W323[4][1088];
	static DATA_T W325[128][1088][1][1];
	static DATA_T W326[4][128];
	static DATA_T W328[32][128][3][3];
	static DATA_T W330[4][1120];
	static DATA_T W332[128][1120][1][1];
	static DATA_T W333[4][128];
	static DATA_T W335[32][128][3][3];
	static DATA_T W337[4][1152];
	static DATA_T W339[128][1152][1][1];
	static DATA_T W340[4][128];
	static DATA_T W342[32][128][3][3];
	static DATA_T W344[4][1184];
	static DATA_T W346[128][1184][1][1];
	static DATA_T W347[4][128];
	static DATA_T W349[32][128][3][3];
	static DATA_T W351[4][1216];
	static DATA_T W353[128][1216][1][1];
	static DATA_T W354[4][128];
	static DATA_T W356[32][128][3][3];
	static DATA_T W358[4][1248];
	static DATA_T W360[128][1248][1][1];
	static DATA_T W361[4][128];
	static DATA_T W363[32][128][3][3];
	static DATA_T W365[4][1280];
	static DATA_T W367[640][1280][1][1];
	static DATA_T W369[4][640];
	static DATA_T W371[128][640][1][1];
	static DATA_T W372[4][128];
	static DATA_T W374[32][128][3][3];
	static DATA_T W376[4][672];
	static DATA_T W378[128][672][1][1];
	static DATA_T W379[4][128];
	static DATA_T W381[32][128][3][3];
	static DATA_T W383[4][704];
	static DATA_T W385[128][704][1][1];
	static DATA_T W386[4][128];
	static DATA_T W388[32][128][3][3];
	static DATA_T W390[4][736];
	static DATA_T W392[128][736][1][1];
	static DATA_T W393[4][128];
	static DATA_T W395[32][128][3][3];
	static DATA_T W397[4][768];
	static DATA_T W399[128][768][1][1];
	static DATA_T W400[4][128];
	static DATA_T W402[32][128][3][3];
	static DATA_T W404[4][800];
	static DATA_T W406[128][800][1][1];
	static DATA_T W407[4][128];
	static DATA_T W409[32][128][3][3];
	static DATA_T W411[4][832];
	static DATA_T W413[128][832][1][1];
	static DATA_T W414[4][128];
	static DATA_T W416[32][128][3][3];
	static DATA_T W418[4][864];
	static DATA_T W420[128][864][1][1];
	static DATA_T W421[4][128];
	static DATA_T W423[32][128][3][3];
	static DATA_T W425[4][896];
	static DATA_T W427[128][896][1][1];
	static DATA_T W428[4][128];
	static DATA_T W430[32][128][3][3];
	static DATA_T W432[4][928];
	static DATA_T W434[128][928][1][1];
	static DATA_T W435[4][128];
	static DATA_T W437[32][128][3][3];
	static DATA_T W439[4][960];
	static DATA_T W441[128][960][1][1];
	static DATA_T W442[4][128];
	static DATA_T W444[32][128][3][3];
	static DATA_T W446[4][992];
	static DATA_T W448[128][992][1][1];
	static DATA_T W449[4][128];
	static DATA_T W451[32][128][3][3];
	static DATA_T W453[4][1024];
	static DATA_T W455[128][1024][1][1];
	static DATA_T W456[4][128];
	static DATA_T W458[32][128][3][3];
	static DATA_T W460[4][1056];
	static DATA_T W462[128][1056][1][1];
	static DATA_T W463[4][128];
	static DATA_T W465[32][128][3][3];
	static DATA_T W467[4][1088];
	static DATA_T W469[128][1088][1][1];
	static DATA_T W470[4][128];
	static DATA_T W472[32][128][3][3];
	static DATA_T W474[4][1120];
	static DATA_T W476[128][1120][1][1];
	static DATA_T W477[4][128];
	static DATA_T W479[32][128][3][3];
	static DATA_T W481[4][1152];
	static DATA_T W483[128][1152][1][1];
	static DATA_T W484[4][128];
	static DATA_T W486[32][128][3][3];
	static DATA_T W488[4][1184];
	static DATA_T W490[128][1184][1][1];
	static DATA_T W491[4][128];
	static DATA_T W493[32][128][3][3];
	static DATA_T W495[4][1216];
	static DATA_T W497[128][1216][1][1];
	static DATA_T W498[4][128];
	static DATA_T W500[32][128][3][3];
	static DATA_T W502[4][1248];
	static DATA_T W504[128][1248][1][1];
	static DATA_T W505[4][128];
	static DATA_T W507[32][128][3][3];
	static DATA_T W509[4][1280];
	static DATA_T W511[128][1280][1][1];
	static DATA_T W512[4][128];
	static DATA_T W514[32][128][3][3];
	static DATA_T W516[4][1312];
	static DATA_T W518[128][1312][1][1];
	static DATA_T W519[4][128];
	static DATA_T W521[32][128][3][3];
	static DATA_T W523[4][1344];
	static DATA_T W525[128][1344][1][1];
	static DATA_T W526[4][128];
	static DATA_T W528[32][128][3][3];
	static DATA_T W530[4][1376];
	static DATA_T W532[128][1376][1][1];
	static DATA_T W533[4][128];
	static DATA_T W535[32][128][3][3];
	static DATA_T W537[4][1408];
	static DATA_T W539[128][1408][1][1];
	static DATA_T W540[4][128];
	static DATA_T W542[32][128][3][3];
	static DATA_T W544[4][1440];
	static DATA_T W546[128][1440][1][1];
	static DATA_T W547[4][128];
	static DATA_T W549[32][128][3][3];
	static DATA_T W551[4][1472];
	static DATA_T W553[128][1472][1][1];
	static DATA_T W554[4][128];
	static DATA_T W556[32][128][3][3];
	static DATA_T W558[4][1504];
	static DATA_T W560[128][1504][1][1];
	static DATA_T W561[4][128];
	static DATA_T W563[32][128][3][3];
	static DATA_T W565[4][1536];
	static DATA_T W567[128][1536][1][1];
	static DATA_T W568[4][128];
	static DATA_T W570[32][128][3][3];
	static DATA_T W572[4][1568];
	static DATA_T W574[128][1568][1][1];
	static DATA_T W575[4][128];
	static DATA_T W577[32][128][3][3];
	static DATA_T W579[4][1600];
	static DATA_T W581[128][1600][1][1];
	static DATA_T W582[4][128];
	static DATA_T W584[32][128][3][3];
	static DATA_T W586[4][1632];
	static DATA_T W588[128][1632][1][1];
	static DATA_T W589[4][128];
	static DATA_T W591[32][128][3][3];
	static DATA_T W593[4][1664];
	static DATA_T B595[1000];
	static DATA_T W595[1000][1664];
	

    static DATA_T O0_SW[3][224][224];
	static DATA_T O1_SW[3][230][230];
	static DATA_T O2_SW[64][112][112];
	static DATA_T O3_SW[64][112][112];
	static DATA_T O4_SW[64][112][112];
	static DATA_T O5_SW[64][114][114];
	static DATA_T O6_SW[64][56][56];
	static DATA_T O7_SW[64][56][56];
	static DATA_T O8_SW[64][56][56];
	static DATA_T O9_SW[128][56][56];
	static DATA_T O10_SW[128][56][56];
	static DATA_T O11_SW[128][56][56];
	static DATA_T O12_SW[32][56][56];
	static DATA_T O13_SW[96][56][56];
	static DATA_T O14_SW[96][56][56];
	static DATA_T O15_SW[96][56][56];
	static DATA_T O16_SW[128][56][56];
	static DATA_T O17_SW[128][56][56];
	static DATA_T O18_SW[128][56][56];
	static DATA_T O19_SW[32][56][56];
	static DATA_T O20_SW[128][56][56];
	static DATA_T O21_SW[128][56][56];
	static DATA_T O22_SW[128][56][56];
	static DATA_T O23_SW[128][56][56];
	static DATA_T O24_SW[128][56][56];
	static DATA_T O25_SW[128][56][56];
	static DATA_T O26_SW[32][56][56];
	static DATA_T O27_SW[160][56][56];
	static DATA_T O28_SW[160][56][56];
	static DATA_T O29_SW[160][56][56];
	static DATA_T O30_SW[128][56][56];
	static DATA_T O31_SW[128][56][56];
	static DATA_T O32_SW[128][56][56];
	static DATA_T O33_SW[32][56][56];
	static DATA_T O34_SW[192][56][56];
	static DATA_T O35_SW[192][56][56];
	static DATA_T O36_SW[192][56][56];
	static DATA_T O37_SW[128][56][56];
	static DATA_T O38_SW[128][56][56];
	static DATA_T O39_SW[128][56][56];
	static DATA_T O40_SW[32][56][56];
	static DATA_T O41_SW[224][56][56];
	static DATA_T O42_SW[224][56][56];
	static DATA_T O43_SW[224][56][56];
	static DATA_T O44_SW[128][56][56];
	static DATA_T O45_SW[128][56][56];
	static DATA_T O46_SW[128][56][56];
	static DATA_T O47_SW[32][56][56];
	static DATA_T O48_SW[256][56][56];
	static DATA_T O49_SW[256][56][56];
	static DATA_T O50_SW[256][56][56];
	static DATA_T O51_SW[128][56][56];
	static DATA_T O52_SW[128][28][28];
	static DATA_T O53_SW[128][28][28];
	static DATA_T O54_SW[128][28][28];
	static DATA_T O55_SW[128][28][28];
	static DATA_T O56_SW[128][28][28];
	static DATA_T O57_SW[128][28][28];
	static DATA_T O58_SW[32][28][28];
	static DATA_T O59_SW[160][28][28];
	static DATA_T O60_SW[160][28][28];
	static DATA_T O61_SW[160][28][28];
	static DATA_T O62_SW[128][28][28];
	static DATA_T O63_SW[128][28][28];
	static DATA_T O64_SW[128][28][28];
	static DATA_T O65_SW[32][28][28];
	static DATA_T O66_SW[192][28][28];
	static DATA_T O67_SW[192][28][28];
	static DATA_T O68_SW[192][28][28];
	static DATA_T O69_SW[128][28][28];
	static DATA_T O70_SW[128][28][28];
	static DATA_T O71_SW[128][28][28];
	static DATA_T O72_SW[32][28][28];
	static DATA_T O73_SW[224][28][28];
	static DATA_T O74_SW[224][28][28];
	static DATA_T O75_SW[224][28][28];
	static DATA_T O76_SW[128][28][28];
	static DATA_T O77_SW[128][28][28];
	static DATA_T O78_SW[128][28][28];
	static DATA_T O79_SW[32][28][28];
	static DATA_T O80_SW[256][28][28];
	static DATA_T O81_SW[256][28][28];
	static DATA_T O82_SW[256][28][28];
	static DATA_T O83_SW[128][28][28];
	static DATA_T O84_SW[128][28][28];
	static DATA_T O85_SW[128][28][28];
	static DATA_T O86_SW[32][28][28];
	static DATA_T O87_SW[288][28][28];
	static DATA_T O88_SW[288][28][28];
	static DATA_T O89_SW[288][28][28];
	static DATA_T O90_SW[128][28][28];
	static DATA_T O91_SW[128][28][28];
	static DATA_T O92_SW[128][28][28];
	static DATA_T O93_SW[32][28][28];
	static DATA_T O94_SW[320][28][28];
	static DATA_T O95_SW[320][28][28];
	static DATA_T O96_SW[320][28][28];
	static DATA_T O97_SW[128][28][28];
	static DATA_T O98_SW[128][28][28];
	static DATA_T O99_SW[128][28][28];
	static DATA_T O100_SW[32][28][28];
	static DATA_T O101_SW[352][28][28];
	static DATA_T O102_SW[352][28][28];
	static DATA_T O103_SW[352][28][28];
	static DATA_T O104_SW[128][28][28];
	static DATA_T O105_SW[128][28][28];
	static DATA_T O106_SW[128][28][28];
	static DATA_T O107_SW[32][28][28];
	static DATA_T O108_SW[384][28][28];
	static DATA_T O109_SW[384][28][28];
	static DATA_T O110_SW[384][28][28];
	static DATA_T O111_SW[128][28][28];
	static DATA_T O112_SW[128][28][28];
	static DATA_T O113_SW[128][28][28];
	static DATA_T O114_SW[32][28][28];
	static DATA_T O115_SW[416][28][28];
	static DATA_T O116_SW[416][28][28];
	static DATA_T O117_SW[416][28][28];
	static DATA_T O118_SW[128][28][28];
	static DATA_T O119_SW[128][28][28];
	static DATA_T O120_SW[128][28][28];
	static DATA_T O121_SW[32][28][28];
	static DATA_T O122_SW[448][28][28];
	static DATA_T O123_SW[448][28][28];
	static DATA_T O124_SW[448][28][28];
	static DATA_T O125_SW[128][28][28];
	static DATA_T O126_SW[128][28][28];
	static DATA_T O127_SW[128][28][28];
	static DATA_T O128_SW[32][28][28];
	static DATA_T O129_SW[480][28][28];
	static DATA_T O130_SW[480][28][28];
	static DATA_T O131_SW[480][28][28];
	static DATA_T O132_SW[128][28][28];
	static DATA_T O133_SW[128][28][28];
	static DATA_T O134_SW[128][28][28];
	static DATA_T O135_SW[32][28][28];
	static DATA_T O136_SW[512][28][28];
	static DATA_T O137_SW[512][28][28];
	static DATA_T O138_SW[512][28][28];
	static DATA_T O139_SW[256][28][28];
	static DATA_T O140_SW[256][14][14];
	static DATA_T O141_SW[256][14][14];
	static DATA_T O142_SW[256][14][14];
	static DATA_T O143_SW[128][14][14];
	static DATA_T O144_SW[128][14][14];
	static DATA_T O145_SW[128][14][14];
	static DATA_T O146_SW[32][14][14];
	static DATA_T O147_SW[288][14][14];
	static DATA_T O148_SW[288][14][14];
	static DATA_T O149_SW[288][14][14];
	static DATA_T O150_SW[128][14][14];
	static DATA_T O151_SW[128][14][14];
	static DATA_T O152_SW[128][14][14];
	static DATA_T O153_SW[32][14][14];
	static DATA_T O154_SW[320][14][14];
	static DATA_T O155_SW[320][14][14];
	static DATA_T O156_SW[320][14][14];
	static DATA_T O157_SW[128][14][14];
	static DATA_T O158_SW[128][14][14];
	static DATA_T O159_SW[128][14][14];
	static DATA_T O160_SW[32][14][14];
	static DATA_T O161_SW[352][14][14];
	static DATA_T O162_SW[352][14][14];
	static DATA_T O163_SW[352][14][14];
	static DATA_T O164_SW[128][14][14];
	static DATA_T O165_SW[128][14][14];
	static DATA_T O166_SW[128][14][14];
	static DATA_T O167_SW[32][14][14];
	static DATA_T O168_SW[384][14][14];
	static DATA_T O169_SW[384][14][14];
	static DATA_T O170_SW[384][14][14];
	static DATA_T O171_SW[128][14][14];
	static DATA_T O172_SW[128][14][14];
	static DATA_T O173_SW[128][14][14];
	static DATA_T O174_SW[32][14][14];
	static DATA_T O175_SW[416][14][14];
	static DATA_T O176_SW[416][14][14];
	static DATA_T O177_SW[416][14][14];
	static DATA_T O178_SW[128][14][14];
	static DATA_T O179_SW[128][14][14];
	static DATA_T O180_SW[128][14][14];
	static DATA_T O181_SW[32][14][14];
	static DATA_T O182_SW[448][14][14];
	static DATA_T O183_SW[448][14][14];
	static DATA_T O184_SW[448][14][14];
	static DATA_T O185_SW[128][14][14];
	static DATA_T O186_SW[128][14][14];
	static DATA_T O187_SW[128][14][14];
	static DATA_T O188_SW[32][14][14];
	static DATA_T O189_SW[480][14][14];
	static DATA_T O190_SW[480][14][14];
	static DATA_T O191_SW[480][14][14];
	static DATA_T O192_SW[128][14][14];
	static DATA_T O193_SW[128][14][14];
	static DATA_T O194_SW[128][14][14];
	static DATA_T O195_SW[32][14][14];
	static DATA_T O196_SW[512][14][14];
	static DATA_T O197_SW[512][14][14];
	static DATA_T O198_SW[512][14][14];
	static DATA_T O199_SW[128][14][14];
	static DATA_T O200_SW[128][14][14];
	static DATA_T O201_SW[128][14][14];
	static DATA_T O202_SW[32][14][14];
	static DATA_T O203_SW[544][14][14];
	static DATA_T O204_SW[544][14][14];
	static DATA_T O205_SW[544][14][14];
	static DATA_T O206_SW[128][14][14];
	static DATA_T O207_SW[128][14][14];
	static DATA_T O208_SW[128][14][14];
	static DATA_T O209_SW[32][14][14];
	static DATA_T O210_SW[576][14][14];
	static DATA_T O211_SW[576][14][14];
	static DATA_T O212_SW[576][14][14];
	static DATA_T O213_SW[128][14][14];
	static DATA_T O214_SW[128][14][14];
	static DATA_T O215_SW[128][14][14];
	static DATA_T O216_SW[32][14][14];
	static DATA_T O217_SW[608][14][14];
	static DATA_T O218_SW[608][14][14];
	static DATA_T O219_SW[608][14][14];
	static DATA_T O220_SW[128][14][14];
	static DATA_T O221_SW[128][14][14];
	static DATA_T O222_SW[128][14][14];
	static DATA_T O223_SW[32][14][14];
	static DATA_T O224_SW[640][14][14];
	static DATA_T O225_SW[640][14][14];
	static DATA_T O226_SW[640][14][14];
	static DATA_T O227_SW[128][14][14];
	static DATA_T O228_SW[128][14][14];
	static DATA_T O229_SW[128][14][14];
	static DATA_T O230_SW[32][14][14];
	static DATA_T O231_SW[672][14][14];
	static DATA_T O232_SW[672][14][14];
	static DATA_T O233_SW[672][14][14];
	static DATA_T O234_SW[128][14][14];
	static DATA_T O235_SW[128][14][14];
	static DATA_T O236_SW[128][14][14];
	static DATA_T O237_SW[32][14][14];
	static DATA_T O238_SW[704][14][14];
	static DATA_T O239_SW[704][14][14];
	static DATA_T O240_SW[704][14][14];
	static DATA_T O241_SW[128][14][14];
	static DATA_T O242_SW[128][14][14];
	static DATA_T O243_SW[128][14][14];
	static DATA_T O244_SW[32][14][14];
	static DATA_T O245_SW[736][14][14];
	static DATA_T O246_SW[736][14][14];
	static DATA_T O247_SW[736][14][14];
	static DATA_T O248_SW[128][14][14];
	static DATA_T O249_SW[128][14][14];
	static DATA_T O250_SW[128][14][14];
	static DATA_T O251_SW[32][14][14];
	static DATA_T O252_SW[768][14][14];
	static DATA_T O253_SW[768][14][14];
	static DATA_T O254_SW[768][14][14];
	static DATA_T O255_SW[128][14][14];
	static DATA_T O256_SW[128][14][14];
	static DATA_T O257_SW[128][14][14];
	static DATA_T O258_SW[32][14][14];
	static DATA_T O259_SW[800][14][14];
	static DATA_T O260_SW[800][14][14];
	static DATA_T O261_SW[800][14][14];
	static DATA_T O262_SW[128][14][14];
	static DATA_T O263_SW[128][14][14];
	static DATA_T O264_SW[128][14][14];
	static DATA_T O265_SW[32][14][14];
	static DATA_T O266_SW[832][14][14];
	static DATA_T O267_SW[832][14][14];
	static DATA_T O268_SW[832][14][14];
	static DATA_T O269_SW[128][14][14];
	static DATA_T O270_SW[128][14][14];
	static DATA_T O271_SW[128][14][14];
	static DATA_T O272_SW[32][14][14];
	static DATA_T O273_SW[864][14][14];
	static DATA_T O274_SW[864][14][14];
	static DATA_T O275_SW[864][14][14];
	static DATA_T O276_SW[128][14][14];
	static DATA_T O277_SW[128][14][14];
	static DATA_T O278_SW[128][14][14];
	static DATA_T O279_SW[32][14][14];
	static DATA_T O280_SW[896][14][14];
	static DATA_T O281_SW[896][14][14];
	static DATA_T O282_SW[896][14][14];
	static DATA_T O283_SW[128][14][14];
	static DATA_T O284_SW[128][14][14];
	static DATA_T O285_SW[128][14][14];
	static DATA_T O286_SW[32][14][14];
	static DATA_T O287_SW[928][14][14];
	static DATA_T O288_SW[928][14][14];
	static DATA_T O289_SW[928][14][14];
	static DATA_T O290_SW[128][14][14];
	static DATA_T O291_SW[128][14][14];
	static DATA_T O292_SW[128][14][14];
	static DATA_T O293_SW[32][14][14];
	static DATA_T O294_SW[960][14][14];
	static DATA_T O295_SW[960][14][14];
	static DATA_T O296_SW[960][14][14];
	static DATA_T O297_SW[128][14][14];
	static DATA_T O298_SW[128][14][14];
	static DATA_T O299_SW[128][14][14];
	static DATA_T O300_SW[32][14][14];
	static DATA_T O301_SW[992][14][14];
	static DATA_T O302_SW[992][14][14];
	static DATA_T O303_SW[992][14][14];
	static DATA_T O304_SW[128][14][14];
	static DATA_T O305_SW[128][14][14];
	static DATA_T O306_SW[128][14][14];
	static DATA_T O307_SW[32][14][14];
	static DATA_T O308_SW[1024][14][14];
	static DATA_T O309_SW[1024][14][14];
	static DATA_T O310_SW[1024][14][14];
	static DATA_T O311_SW[128][14][14];
	static DATA_T O312_SW[128][14][14];
	static DATA_T O313_SW[128][14][14];
	static DATA_T O314_SW[32][14][14];
	static DATA_T O315_SW[1056][14][14];
	static DATA_T O316_SW[1056][14][14];
	static DATA_T O317_SW[1056][14][14];
	static DATA_T O318_SW[128][14][14];
	static DATA_T O319_SW[128][14][14];
	static DATA_T O320_SW[128][14][14];
	static DATA_T O321_SW[32][14][14];
	static DATA_T O322_SW[1088][14][14];
	static DATA_T O323_SW[1088][14][14];
	static DATA_T O324_SW[1088][14][14];
	static DATA_T O325_SW[128][14][14];
	static DATA_T O326_SW[128][14][14];
	static DATA_T O327_SW[128][14][14];
	static DATA_T O328_SW[32][14][14];
	static DATA_T O329_SW[1120][14][14];
	static DATA_T O330_SW[1120][14][14];
	static DATA_T O331_SW[1120][14][14];
	static DATA_T O332_SW[128][14][14];
	static DATA_T O333_SW[128][14][14];
	static DATA_T O334_SW[128][14][14];
	static DATA_T O335_SW[32][14][14];
	static DATA_T O336_SW[1152][14][14];
	static DATA_T O337_SW[1152][14][14];
	static DATA_T O338_SW[1152][14][14];
	static DATA_T O339_SW[128][14][14];
	static DATA_T O340_SW[128][14][14];
	static DATA_T O341_SW[128][14][14];
	static DATA_T O342_SW[32][14][14];
	static DATA_T O343_SW[1184][14][14];
	static DATA_T O344_SW[1184][14][14];
	static DATA_T O345_SW[1184][14][14];
	static DATA_T O346_SW[128][14][14];
	static DATA_T O347_SW[128][14][14];
	static DATA_T O348_SW[128][14][14];
	static DATA_T O349_SW[32][14][14];
	static DATA_T O350_SW[1216][14][14];
	static DATA_T O351_SW[1216][14][14];
	static DATA_T O352_SW[1216][14][14];
	static DATA_T O353_SW[128][14][14];
	static DATA_T O354_SW[128][14][14];
	static DATA_T O355_SW[128][14][14];
	static DATA_T O356_SW[32][14][14];
	static DATA_T O357_SW[1248][14][14];
	static DATA_T O358_SW[1248][14][14];
	static DATA_T O359_SW[1248][14][14];
	static DATA_T O360_SW[128][14][14];
	static DATA_T O361_SW[128][14][14];
	static DATA_T O362_SW[128][14][14];
	static DATA_T O363_SW[32][14][14];
	static DATA_T O364_SW[1280][14][14];
	static DATA_T O365_SW[1280][14][14];
	static DATA_T O366_SW[1280][14][14];
	static DATA_T O367_SW[640][14][14];
	static DATA_T O368_SW[640][7][7];
	static DATA_T O369_SW[640][7][7];
	static DATA_T O370_SW[640][7][7];
	static DATA_T O371_SW[128][7][7];
	static DATA_T O372_SW[128][7][7];
	static DATA_T O373_SW[128][7][7];
	static DATA_T O374_SW[32][7][7];
	static DATA_T O375_SW[672][7][7];
	static DATA_T O376_SW[672][7][7];
	static DATA_T O377_SW[672][7][7];
	static DATA_T O378_SW[128][7][7];
	static DATA_T O379_SW[128][7][7];
	static DATA_T O380_SW[128][7][7];
	static DATA_T O381_SW[32][7][7];
	static DATA_T O382_SW[704][7][7];
	static DATA_T O383_SW[704][7][7];
	static DATA_T O384_SW[704][7][7];
	static DATA_T O385_SW[128][7][7];
	static DATA_T O386_SW[128][7][7];
	static DATA_T O387_SW[128][7][7];
	static DATA_T O388_SW[32][7][7];
	static DATA_T O389_SW[736][7][7];
	static DATA_T O390_SW[736][7][7];
	static DATA_T O391_SW[736][7][7];
	static DATA_T O392_SW[128][7][7];
	static DATA_T O393_SW[128][7][7];
	static DATA_T O394_SW[128][7][7];
	static DATA_T O395_SW[32][7][7];
	static DATA_T O396_SW[768][7][7];
	static DATA_T O397_SW[768][7][7];
	static DATA_T O398_SW[768][7][7];
	static DATA_T O399_SW[128][7][7];
	static DATA_T O400_SW[128][7][7];
	static DATA_T O401_SW[128][7][7];
	static DATA_T O402_SW[32][7][7];
	static DATA_T O403_SW[800][7][7];
	static DATA_T O404_SW[800][7][7];
	static DATA_T O405_SW[800][7][7];
	static DATA_T O406_SW[128][7][7];
	static DATA_T O407_SW[128][7][7];
	static DATA_T O408_SW[128][7][7];
	static DATA_T O409_SW[32][7][7];
	static DATA_T O410_SW[832][7][7];
	static DATA_T O411_SW[832][7][7];
	static DATA_T O412_SW[832][7][7];
	static DATA_T O413_SW[128][7][7];
	static DATA_T O414_SW[128][7][7];
	static DATA_T O415_SW[128][7][7];
	static DATA_T O416_SW[32][7][7];
	static DATA_T O417_SW[864][7][7];
	static DATA_T O418_SW[864][7][7];
	static DATA_T O419_SW[864][7][7];
	static DATA_T O420_SW[128][7][7];
	static DATA_T O421_SW[128][7][7];
	static DATA_T O422_SW[128][7][7];
	static DATA_T O423_SW[32][7][7];
	static DATA_T O424_SW[896][7][7];
	static DATA_T O425_SW[896][7][7];
	static DATA_T O426_SW[896][7][7];
	static DATA_T O427_SW[128][7][7];
	static DATA_T O428_SW[128][7][7];
	static DATA_T O429_SW[128][7][7];
	static DATA_T O430_SW[32][7][7];
	static DATA_T O431_SW[928][7][7];
	static DATA_T O432_SW[928][7][7];
	static DATA_T O433_SW[928][7][7];
	static DATA_T O434_SW[128][7][7];
	static DATA_T O435_SW[128][7][7];
	static DATA_T O436_SW[128][7][7];
	static DATA_T O437_SW[32][7][7];
	static DATA_T O438_SW[960][7][7];
	static DATA_T O439_SW[960][7][7];
	static DATA_T O440_SW[960][7][7];
	static DATA_T O441_SW[128][7][7];
	static DATA_T O442_SW[128][7][7];
	static DATA_T O443_SW[128][7][7];
	static DATA_T O444_SW[32][7][7];
	static DATA_T O445_SW[992][7][7];
	static DATA_T O446_SW[992][7][7];
	static DATA_T O447_SW[992][7][7];
	static DATA_T O448_SW[128][7][7];
	static DATA_T O449_SW[128][7][7];
	static DATA_T O450_SW[128][7][7];
	static DATA_T O451_SW[32][7][7];
	static DATA_T O452_SW[1024][7][7];
	static DATA_T O453_SW[1024][7][7];
	static DATA_T O454_SW[1024][7][7];
	static DATA_T O455_SW[128][7][7];
	static DATA_T O456_SW[128][7][7];
	static DATA_T O457_SW[128][7][7];
	static DATA_T O458_SW[32][7][7];
	static DATA_T O459_SW[1056][7][7];
	static DATA_T O460_SW[1056][7][7];
	static DATA_T O461_SW[1056][7][7];
	static DATA_T O462_SW[128][7][7];
	static DATA_T O463_SW[128][7][7];
	static DATA_T O464_SW[128][7][7];
	static DATA_T O465_SW[32][7][7];
	static DATA_T O466_SW[1088][7][7];
	static DATA_T O467_SW[1088][7][7];
	static DATA_T O468_SW[1088][7][7];
	static DATA_T O469_SW[128][7][7];
	static DATA_T O470_SW[128][7][7];
	static DATA_T O471_SW[128][7][7];
	static DATA_T O472_SW[32][7][7];
	static DATA_T O473_SW[1120][7][7];
	static DATA_T O474_SW[1120][7][7];
	static DATA_T O475_SW[1120][7][7];
	static DATA_T O476_SW[128][7][7];
	static DATA_T O477_SW[128][7][7];
	static DATA_T O478_SW[128][7][7];
	static DATA_T O479_SW[32][7][7];
	static DATA_T O480_SW[1152][7][7];
	static DATA_T O481_SW[1152][7][7];
	static DATA_T O482_SW[1152][7][7];
	static DATA_T O483_SW[128][7][7];
	static DATA_T O484_SW[128][7][7];
	static DATA_T O485_SW[128][7][7];
	static DATA_T O486_SW[32][7][7];
	static DATA_T O487_SW[1184][7][7];
	static DATA_T O488_SW[1184][7][7];
	static DATA_T O489_SW[1184][7][7];
	static DATA_T O490_SW[128][7][7];
	static DATA_T O491_SW[128][7][7];
	static DATA_T O492_SW[128][7][7];
	static DATA_T O493_SW[32][7][7];
	static DATA_T O494_SW[1216][7][7];
	static DATA_T O495_SW[1216][7][7];
	static DATA_T O496_SW[1216][7][7];
	static DATA_T O497_SW[128][7][7];
	static DATA_T O498_SW[128][7][7];
	static DATA_T O499_SW[128][7][7];
	static DATA_T O500_SW[32][7][7];
	static DATA_T O501_SW[1248][7][7];
	static DATA_T O502_SW[1248][7][7];
	static DATA_T O503_SW[1248][7][7];
	static DATA_T O504_SW[128][7][7];
	static DATA_T O505_SW[128][7][7];
	static DATA_T O506_SW[128][7][7];
	static DATA_T O507_SW[32][7][7];
	static DATA_T O508_SW[1280][7][7];
	static DATA_T O509_SW[1280][7][7];
	static DATA_T O510_SW[1280][7][7];
	static DATA_T O511_SW[128][7][7];
	static DATA_T O512_SW[128][7][7];
	static DATA_T O513_SW[128][7][7];
	static DATA_T O514_SW[32][7][7];
	static DATA_T O515_SW[1312][7][7];
	static DATA_T O516_SW[1312][7][7];
	static DATA_T O517_SW[1312][7][7];
	static DATA_T O518_SW[128][7][7];
	static DATA_T O519_SW[128][7][7];
	static DATA_T O520_SW[128][7][7];
	static DATA_T O521_SW[32][7][7];
	static DATA_T O522_SW[1344][7][7];
	static DATA_T O523_SW[1344][7][7];
	static DATA_T O524_SW[1344][7][7];
	static DATA_T O525_SW[128][7][7];
	static DATA_T O526_SW[128][7][7];
	static DATA_T O527_SW[128][7][7];
	static DATA_T O528_SW[32][7][7];
	static DATA_T O529_SW[1376][7][7];
	static DATA_T O530_SW[1376][7][7];
	static DATA_T O531_SW[1376][7][7];
	static DATA_T O532_SW[128][7][7];
	static DATA_T O533_SW[128][7][7];
	static DATA_T O534_SW[128][7][7];
	static DATA_T O535_SW[32][7][7];
	static DATA_T O536_SW[1408][7][7];
	static DATA_T O537_SW[1408][7][7];
	static DATA_T O538_SW[1408][7][7];
	static DATA_T O539_SW[128][7][7];
	static DATA_T O540_SW[128][7][7];
	static DATA_T O541_SW[128][7][7];
	static DATA_T O542_SW[32][7][7];
	static DATA_T O543_SW[1440][7][7];
	static DATA_T O544_SW[1440][7][7];
	static DATA_T O545_SW[1440][7][7];
	static DATA_T O546_SW[128][7][7];
	static DATA_T O547_SW[128][7][7];
	static DATA_T O548_SW[128][7][7];
	static DATA_T O549_SW[32][7][7];
	static DATA_T O550_SW[1472][7][7];
	static DATA_T O551_SW[1472][7][7];
	static DATA_T O552_SW[1472][7][7];
	static DATA_T O553_SW[128][7][7];
	static DATA_T O554_SW[128][7][7];
	static DATA_T O555_SW[128][7][7];
	static DATA_T O556_SW[32][7][7];
	static DATA_T O557_SW[1504][7][7];
	static DATA_T O558_SW[1504][7][7];
	static DATA_T O559_SW[1504][7][7];
	static DATA_T O560_SW[128][7][7];
	static DATA_T O561_SW[128][7][7];
	static DATA_T O562_SW[128][7][7];
	static DATA_T O563_SW[32][7][7];
	static DATA_T O564_SW[1536][7][7];
	static DATA_T O565_SW[1536][7][7];
	static DATA_T O566_SW[1536][7][7];
	static DATA_T O567_SW[128][7][7];
	static DATA_T O568_SW[128][7][7];
	static DATA_T O569_SW[128][7][7];
	static DATA_T O570_SW[32][7][7];
	static DATA_T O571_SW[1568][7][7];
	static DATA_T O572_SW[1568][7][7];
	static DATA_T O573_SW[1568][7][7];
	static DATA_T O574_SW[128][7][7];
	static DATA_T O575_SW[128][7][7];
	static DATA_T O576_SW[128][7][7];
	static DATA_T O577_SW[32][7][7];
	static DATA_T O578_SW[1600][7][7];
	static DATA_T O579_SW[1600][7][7];
	static DATA_T O580_SW[1600][7][7];
	static DATA_T O581_SW[128][7][7];
	static DATA_T O582_SW[128][7][7];
	static DATA_T O583_SW[128][7][7];
	static DATA_T O584_SW[32][7][7];
	static DATA_T O585_SW[1632][7][7];
	static DATA_T O586_SW[1632][7][7];
	static DATA_T O587_SW[1632][7][7];
	static DATA_T O588_SW[128][7][7];
	static DATA_T O589_SW[128][7][7];
	static DATA_T O590_SW[128][7][7];
	static DATA_T O591_SW[32][7][7];
	static DATA_T O592_SW[1664][7][7];
	static DATA_T O593_SW[1664][7][7];
	static DATA_T O594_SW[1664];
	static DATA_T O595_SW[1000];
	

    FILE *w_stream = fopen(argv[1], "rb");
    if (w_stream == NULL) printf("weight file was not opened");
    FILE *i_stream = fopen(argv[2], "rb");
    if (i_stream == NULL) printf("input file was not opened");
    FILE *o_stream = fopen("Produced_code/densenet169/Output/c_output.txt", "w");
    if (o_stream == NULL) printf("Output file was not opened");
    FILE *c_num = fopen("Produced_code/densenet169/Output/c_output_num.txt", "w");
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

/*
for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B2[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W3[x][y] = (DATA_T) trash;
    }
}
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W7[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W9[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B9[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W10[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W12[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B12[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 96 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W14[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 96 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W16[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B16[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W17[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W19[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B19[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W21[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W23[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B23[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W24[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W26[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B26[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W28[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W30[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B30[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W31[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W33[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B33[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W35[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 192 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W37[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B37[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W38[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W40[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B40[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 224 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W42[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 224 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W44[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B44[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W45[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W47[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B47[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W49[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W51[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B51[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W53[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W55[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B55[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W56[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W58[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B58[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W60[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W62[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B62[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W63[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W65[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B65[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W67[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 192 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W69[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B69[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W70[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
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

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 224 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W74[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 224 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W76[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B76[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W77[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W79[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B79[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W81[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W83[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B83[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W84[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W86[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B86[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 288 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W88[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 288 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W90[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B90[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W91[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W93[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B93[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 320 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W95[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W97[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B97[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W98[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W100[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B100[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 352 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W102[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 352 ; i++) {
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

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W105[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
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

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 384 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W109[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W111[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B111[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W112[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W114[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B114[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 416 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W116[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 416 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W118[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B118[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W119[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W121[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B121[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 448 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W123[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 448 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W125[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B125[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W126[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W128[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B128[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 480 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W130[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 480 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W132[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B132[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W133[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W135[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B135[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 512 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W137[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W139[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B139[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W141[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W143[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B143[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W144[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W146[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B146[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 288 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W148[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 288 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W150[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B150[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W151[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W153[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B153[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 320 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W155[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W157[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B157[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W158[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
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

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 352 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W162[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 352 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W164[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B164[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W165[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W167[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B167[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 384 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W169[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W171[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B171[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W172[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W174[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B174[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 416 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W176[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 416 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W178[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B178[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W179[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W181[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B181[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 448 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W183[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 448 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W185[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B185[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W186[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W188[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B188[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 480 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W190[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 480 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W192[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B192[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W193[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
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

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 512 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W197[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W199[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B199[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W200[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W202[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B202[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 544 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W204[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 544 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W206[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B206[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W207[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W209[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B209[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 576 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W211[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 576 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W213[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B213[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W214[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W216[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B216[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 608 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W218[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 608 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W220[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B220[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W221[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W223[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B223[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 640 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W225[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 640 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W227[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B227[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W228[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W230[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B230[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 672 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W232[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 672 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W234[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B234[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W235[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W237[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B237[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 704 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W239[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 704 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W241[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B241[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W242[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W244[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B244[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 736 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W246[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 736 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W248[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B248[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W249[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W251[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B251[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 768 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W253[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 768 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W255[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B255[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W256[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W258[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B258[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 800 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W260[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 800 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W262[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B262[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W263[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W265[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B265[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 832 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W267[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 832 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W269[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B269[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W270[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W272[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B272[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 864 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W274[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 864 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W276[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B276[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W277[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W279[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B279[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 896 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W281[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 896 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W283[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B283[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W284[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W286[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B286[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 928 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W288[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 928 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W290[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B290[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W291[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W293[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B293[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 960 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W295[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 960 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W297[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B297[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W298[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W300[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B300[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 992 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W302[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 992 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W304[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B304[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W305[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W307[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B307[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1024 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W309[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1024 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W311[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B311[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W312[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W314[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B314[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1056 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W316[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1056 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W318[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B318[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W319[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W321[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B321[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1088 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W323[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W325[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B325[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W326[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W328[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B328[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1120 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W330[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1120 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W332[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B332[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W333[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W335[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B335[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1152 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W337[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1152 ; i++) {
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

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W340[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W342[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B342[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1184 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W344[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1184 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W346[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B346[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W347[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W349[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B349[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1216 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W351[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1216 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W353[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B353[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W354[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W356[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B356[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1248 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W358[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1248 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W360[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B360[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W361[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W363[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B363[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1280 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W365[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1280 ; i++) {
			for (j = 0; j < 640 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W367[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 640 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B367[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 640 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W369[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 640 ; i++) {
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

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W372[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W374[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B374[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 672 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W376[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 672 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W378[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B378[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W379[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W381[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B381[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 704 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W383[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 704 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W385[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B385[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W386[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W388[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B388[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 736 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W390[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 736 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W392[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B392[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W393[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W395[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B395[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 768 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W397[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 768 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W399[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B399[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W400[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W402[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B402[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 800 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W404[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 800 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W406[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B406[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W407[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W409[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B409[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 832 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W411[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 832 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W413[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B413[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W414[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W416[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B416[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 864 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W418[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 864 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W420[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B420[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W421[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W423[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B423[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 896 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W425[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 896 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W427[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B427[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W428[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W430[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B430[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 928 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W432[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 928 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W434[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B434[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W435[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W437[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B437[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 960 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W439[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 960 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W441[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B441[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W442[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W444[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B444[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 992 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W446[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 992 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W448[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B448[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W449[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W451[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B451[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1024 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W453[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1024 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W455[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B455[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W456[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W458[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B458[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1056 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W460[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1056 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W462[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B462[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W463[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W465[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B465[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1088 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W467[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1088 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W469[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B469[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W470[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W472[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B472[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1120 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W474[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1120 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W476[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B476[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W477[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W479[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B479[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1152 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W481[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1152 ; i++) {
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

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W484[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W486[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B486[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1184 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W488[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1184 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W490[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B490[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W491[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W493[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B493[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1216 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W495[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1216 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W497[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B497[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W498[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W500[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B500[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1248 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W502[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1248 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W504[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B504[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W505[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W507[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B507[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1280 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W509[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1280 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W511[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B511[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W512[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W514[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B514[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1312 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W516[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1312 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W518[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B518[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W519[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W521[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B521[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1344 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W523[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1344 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W525[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B525[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W526[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W528[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B528[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1376 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W530[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1376 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W532[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B532[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W533[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W535[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B535[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1408 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W537[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1408 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W539[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B539[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W540[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W542[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B542[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1440 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W544[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1440 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W546[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B546[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W547[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W549[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B549[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1472 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W551[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1472 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W553[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B553[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W554[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W556[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B556[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1504 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W558[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1504 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W560[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B560[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W561[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W563[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B563[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1536 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W565[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1536 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W567[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B567[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W568[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W570[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B570[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1568 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W572[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1568 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W574[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B574[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W575[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W577[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B577[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1600 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W579[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1600 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W581[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B581[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W582[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W584[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B584[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1632 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W586[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1632 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W588[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B588[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W589[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W591[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B591[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1664 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W593[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1664 ; m++) {
	for (k = 0; k < 1000 ; k++) {
		fread(&trash, sizeof(int), 1, w_stream);
        W595[k][m] = (DATA_T) trash;
	}
}


for (m = 0; m < 1000 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B595[m] = (DATA_T) trash;
}

	
    printf("[C_verifier.cpp]Finish Initialization");

    printf("[C_verifier.cpp]InputLayer\n\n");
	printf("[C_verifier.cpp]Calculate ZeroPadding2D1\n\n");
	SW_zero_padding2d_3(O0_SW,O1_SW);
	printf("[C_verifier.cpp]Calculate Conv2D2\n\n");
	SW_conv1_conv(O1_SW,O2_SW,W2);
	printf("[C_verifier.cpp]Calculate BatchNormalization3\n\n");
	SW_conv1_bn(O2_SW,O3_SW, W3);
	printf("[C_verifier.cpp]Calculate Activation(Relu)4\n\n");
	SW_conv1_relu(O3_SW,O4_SW);
	printf("[C_verifier.cpp]Calculate ZeroPadding2D5\n\n");
	SW_zero_padding2d_4(O4_SW,O5_SW);
	printf("[C_verifier.cpp]Calculate MaxPooling2D6\n\n");
	SW_pool1(O5_SW,O6_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization7\n\n");
	SW_conv2_block1_0_bn(O6_SW,O7_SW, W7);
	printf("[C_verifier.cpp]Calculate Activation(Relu)8\n\n");
	SW_conv2_block1_0_relu(O7_SW,O8_SW);
	printf("[C_verifier.cpp]Calculate Conv2D9\n\n");
	SW_conv2_block1_1_conv(O8_SW,O9_SW,W9);
	printf("[C_verifier.cpp]Calculate BatchNormalization10\n\n");
	SW_conv2_block1_1_bn(O9_SW,O10_SW, W10);
	printf("[C_verifier.cpp]Calculate Activation(Relu)11\n\n");
	SW_conv2_block1_1_relu(O10_SW,O11_SW);
	printf("[C_verifier.cpp]Calculate Conv2D12\n\n");
	SW_conv2_block1_2_conv(O11_SW,O12_SW,W12);
	printf("[C_verifier.cpp]Calculate Concatenate13\n\n");
	SW_conv2_block1_concat(O6_SW, O12_SW, O13_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization14\n\n");
	SW_conv2_block2_0_bn(O13_SW,O14_SW, W14);
	printf("[C_verifier.cpp]Calculate Activation(Relu)15\n\n");
	SW_conv2_block2_0_relu(O14_SW,O15_SW);
	printf("[C_verifier.cpp]Calculate Conv2D16\n\n");
	SW_conv2_block2_1_conv(O15_SW,O16_SW,W16);
	printf("[C_verifier.cpp]Calculate BatchNormalization17\n\n");
	SW_conv2_block2_1_bn(O16_SW,O17_SW, W17);
	printf("[C_verifier.cpp]Calculate Activation(Relu)18\n\n");
	SW_conv2_block2_1_relu(O17_SW,O18_SW);
	printf("[C_verifier.cpp]Calculate Conv2D19\n\n");
	SW_conv2_block2_2_conv(O18_SW,O19_SW,W19);
	printf("[C_verifier.cpp]Calculate Concatenate20\n\n");
	SW_conv2_block2_concat(O13_SW, O19_SW, O20_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization21\n\n");
	SW_conv2_block3_0_bn(O20_SW,O21_SW, W21);
	printf("[C_verifier.cpp]Calculate Activation(Relu)22\n\n");
	SW_conv2_block3_0_relu(O21_SW,O22_SW);
	printf("[C_verifier.cpp]Calculate Conv2D23\n\n");
	SW_conv2_block3_1_conv(O22_SW,O23_SW,W23);
	printf("[C_verifier.cpp]Calculate BatchNormalization24\n\n");
	SW_conv2_block3_1_bn(O23_SW,O24_SW, W24);
	printf("[C_verifier.cpp]Calculate Activation(Relu)25\n\n");
	SW_conv2_block3_1_relu(O24_SW,O25_SW);
	printf("[C_verifier.cpp]Calculate Conv2D26\n\n");
	SW_conv2_block3_2_conv(O25_SW,O26_SW,W26);
	printf("[C_verifier.cpp]Calculate Concatenate27\n\n");
	SW_conv2_block3_concat(O20_SW, O26_SW, O27_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization28\n\n");
	SW_conv2_block4_0_bn(O27_SW,O28_SW, W28);
	printf("[C_verifier.cpp]Calculate Activation(Relu)29\n\n");
	SW_conv2_block4_0_relu(O28_SW,O29_SW);
	printf("[C_verifier.cpp]Calculate Conv2D30\n\n");
	SW_conv2_block4_1_conv(O29_SW,O30_SW,W30);
	printf("[C_verifier.cpp]Calculate BatchNormalization31\n\n");
	SW_conv2_block4_1_bn(O30_SW,O31_SW, W31);
	printf("[C_verifier.cpp]Calculate Activation(Relu)32\n\n");
	SW_conv2_block4_1_relu(O31_SW,O32_SW);
	printf("[C_verifier.cpp]Calculate Conv2D33\n\n");
	SW_conv2_block4_2_conv(O32_SW,O33_SW,W33);
	printf("[C_verifier.cpp]Calculate Concatenate34\n\n");
	SW_conv2_block4_concat(O27_SW, O33_SW, O34_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization35\n\n");
	SW_conv2_block5_0_bn(O34_SW,O35_SW, W35);
	printf("[C_verifier.cpp]Calculate Activation(Relu)36\n\n");
	SW_conv2_block5_0_relu(O35_SW,O36_SW);
	printf("[C_verifier.cpp]Calculate Conv2D37\n\n");
	SW_conv2_block5_1_conv(O36_SW,O37_SW,W37);
	printf("[C_verifier.cpp]Calculate BatchNormalization38\n\n");
	SW_conv2_block5_1_bn(O37_SW,O38_SW, W38);
	printf("[C_verifier.cpp]Calculate Activation(Relu)39\n\n");
	SW_conv2_block5_1_relu(O38_SW,O39_SW);
	printf("[C_verifier.cpp]Calculate Conv2D40\n\n");
	SW_conv2_block5_2_conv(O39_SW,O40_SW,W40);
	printf("[C_verifier.cpp]Calculate Concatenate41\n\n");
	SW_conv2_block5_concat(O34_SW, O40_SW, O41_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization42\n\n");
	SW_conv2_block6_0_bn(O41_SW,O42_SW, W42);
	printf("[C_verifier.cpp]Calculate Activation(Relu)43\n\n");
	SW_conv2_block6_0_relu(O42_SW,O43_SW);
	printf("[C_verifier.cpp]Calculate Conv2D44\n\n");
	SW_conv2_block6_1_conv(O43_SW,O44_SW,W44);
	printf("[C_verifier.cpp]Calculate BatchNormalization45\n\n");
	SW_conv2_block6_1_bn(O44_SW,O45_SW, W45);
	printf("[C_verifier.cpp]Calculate Activation(Relu)46\n\n");
	SW_conv2_block6_1_relu(O45_SW,O46_SW);
	printf("[C_verifier.cpp]Calculate Conv2D47\n\n");
	SW_conv2_block6_2_conv(O46_SW,O47_SW,W47);
	printf("[C_verifier.cpp]Calculate Concatenate48\n\n");
	SW_conv2_block6_concat(O41_SW, O47_SW, O48_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization49\n\n");
	SW_pool2_bn(O48_SW,O49_SW, W49);
	printf("[C_verifier.cpp]Calculate Activation(Relu)50\n\n");
	SW_pool2_relu(O49_SW,O50_SW);
	printf("[C_verifier.cpp]Calculate Conv2D51\n\n");
	SW_pool2_conv(O50_SW,O51_SW,W51);
	printf("[C_verifier.cpp]Calculate AveragePooling2D52\n\n");
	SW_pool2_pool(O51_SW,O52_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization53\n\n");
	SW_conv3_block1_0_bn(O52_SW,O53_SW, W53);
	printf("[C_verifier.cpp]Calculate Activation(Relu)54\n\n");
	SW_conv3_block1_0_relu(O53_SW,O54_SW);
	printf("[C_verifier.cpp]Calculate Conv2D55\n\n");
	SW_conv3_block1_1_conv(O54_SW,O55_SW,W55);
	printf("[C_verifier.cpp]Calculate BatchNormalization56\n\n");
	SW_conv3_block1_1_bn(O55_SW,O56_SW, W56);
	printf("[C_verifier.cpp]Calculate Activation(Relu)57\n\n");
	SW_conv3_block1_1_relu(O56_SW,O57_SW);
	printf("[C_verifier.cpp]Calculate Conv2D58\n\n");
	SW_conv3_block1_2_conv(O57_SW,O58_SW,W58);
	printf("[C_verifier.cpp]Calculate Concatenate59\n\n");
	SW_conv3_block1_concat(O52_SW, O58_SW, O59_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization60\n\n");
	SW_conv3_block2_0_bn(O59_SW,O60_SW, W60);
	printf("[C_verifier.cpp]Calculate Activation(Relu)61\n\n");
	SW_conv3_block2_0_relu(O60_SW,O61_SW);
	printf("[C_verifier.cpp]Calculate Conv2D62\n\n");
	SW_conv3_block2_1_conv(O61_SW,O62_SW,W62);
	printf("[C_verifier.cpp]Calculate BatchNormalization63\n\n");
	SW_conv3_block2_1_bn(O62_SW,O63_SW, W63);
	printf("[C_verifier.cpp]Calculate Activation(Relu)64\n\n");
	SW_conv3_block2_1_relu(O63_SW,O64_SW);
	printf("[C_verifier.cpp]Calculate Conv2D65\n\n");
	SW_conv3_block2_2_conv(O64_SW,O65_SW,W65);
	printf("[C_verifier.cpp]Calculate Concatenate66\n\n");
	SW_conv3_block2_concat(O59_SW, O65_SW, O66_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization67\n\n");
	SW_conv3_block3_0_bn(O66_SW,O67_SW, W67);
	printf("[C_verifier.cpp]Calculate Activation(Relu)68\n\n");
	SW_conv3_block3_0_relu(O67_SW,O68_SW);
	printf("[C_verifier.cpp]Calculate Conv2D69\n\n");
	SW_conv3_block3_1_conv(O68_SW,O69_SW,W69);
	printf("[C_verifier.cpp]Calculate BatchNormalization70\n\n");
	SW_conv3_block3_1_bn(O69_SW,O70_SW, W70);
	printf("[C_verifier.cpp]Calculate Activation(Relu)71\n\n");
	SW_conv3_block3_1_relu(O70_SW,O71_SW);
	printf("[C_verifier.cpp]Calculate Conv2D72\n\n");
	SW_conv3_block3_2_conv(O71_SW,O72_SW,W72);
	printf("[C_verifier.cpp]Calculate Concatenate73\n\n");
	SW_conv3_block3_concat(O66_SW, O72_SW, O73_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization74\n\n");
	SW_conv3_block4_0_bn(O73_SW,O74_SW, W74);
	printf("[C_verifier.cpp]Calculate Activation(Relu)75\n\n");
	SW_conv3_block4_0_relu(O74_SW,O75_SW);
	printf("[C_verifier.cpp]Calculate Conv2D76\n\n");
	SW_conv3_block4_1_conv(O75_SW,O76_SW,W76);
	printf("[C_verifier.cpp]Calculate BatchNormalization77\n\n");
	SW_conv3_block4_1_bn(O76_SW,O77_SW, W77);
	printf("[C_verifier.cpp]Calculate Activation(Relu)78\n\n");
	SW_conv3_block4_1_relu(O77_SW,O78_SW);
	printf("[C_verifier.cpp]Calculate Conv2D79\n\n");
	SW_conv3_block4_2_conv(O78_SW,O79_SW,W79);
	printf("[C_verifier.cpp]Calculate Concatenate80\n\n");
	SW_conv3_block4_concat(O73_SW, O79_SW, O80_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization81\n\n");
	SW_conv3_block5_0_bn(O80_SW,O81_SW, W81);
	printf("[C_verifier.cpp]Calculate Activation(Relu)82\n\n");
	SW_conv3_block5_0_relu(O81_SW,O82_SW);
	printf("[C_verifier.cpp]Calculate Conv2D83\n\n");
	SW_conv3_block5_1_conv(O82_SW,O83_SW,W83);
	printf("[C_verifier.cpp]Calculate BatchNormalization84\n\n");
	SW_conv3_block5_1_bn(O83_SW,O84_SW, W84);
	printf("[C_verifier.cpp]Calculate Activation(Relu)85\n\n");
	SW_conv3_block5_1_relu(O84_SW,O85_SW);
	printf("[C_verifier.cpp]Calculate Conv2D86\n\n");
	SW_conv3_block5_2_conv(O85_SW,O86_SW,W86);
	printf("[C_verifier.cpp]Calculate Concatenate87\n\n");
	SW_conv3_block5_concat(O80_SW, O86_SW, O87_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization88\n\n");
	SW_conv3_block6_0_bn(O87_SW,O88_SW, W88);
	printf("[C_verifier.cpp]Calculate Activation(Relu)89\n\n");
	SW_conv3_block6_0_relu(O88_SW,O89_SW);
	printf("[C_verifier.cpp]Calculate Conv2D90\n\n");
	SW_conv3_block6_1_conv(O89_SW,O90_SW,W90);
	printf("[C_verifier.cpp]Calculate BatchNormalization91\n\n");
	SW_conv3_block6_1_bn(O90_SW,O91_SW, W91);
	printf("[C_verifier.cpp]Calculate Activation(Relu)92\n\n");
	SW_conv3_block6_1_relu(O91_SW,O92_SW);
	printf("[C_verifier.cpp]Calculate Conv2D93\n\n");
	SW_conv3_block6_2_conv(O92_SW,O93_SW,W93);
	printf("[C_verifier.cpp]Calculate Concatenate94\n\n");
	SW_conv3_block6_concat(O87_SW, O93_SW, O94_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization95\n\n");
	SW_conv3_block7_0_bn(O94_SW,O95_SW, W95);
	printf("[C_verifier.cpp]Calculate Activation(Relu)96\n\n");
	SW_conv3_block7_0_relu(O95_SW,O96_SW);
	printf("[C_verifier.cpp]Calculate Conv2D97\n\n");
	SW_conv3_block7_1_conv(O96_SW,O97_SW,W97);
	printf("[C_verifier.cpp]Calculate BatchNormalization98\n\n");
	SW_conv3_block7_1_bn(O97_SW,O98_SW, W98);
	printf("[C_verifier.cpp]Calculate Activation(Relu)99\n\n");
	SW_conv3_block7_1_relu(O98_SW,O99_SW);
	printf("[C_verifier.cpp]Calculate Conv2D100\n\n");
	SW_conv3_block7_2_conv(O99_SW,O100_SW,W100);
	printf("[C_verifier.cpp]Calculate Concatenate101\n\n");
	SW_conv3_block7_concat(O94_SW, O100_SW, O101_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization102\n\n");
	SW_conv3_block8_0_bn(O101_SW,O102_SW, W102);
	printf("[C_verifier.cpp]Calculate Activation(Relu)103\n\n");
	SW_conv3_block8_0_relu(O102_SW,O103_SW);
	printf("[C_verifier.cpp]Calculate Conv2D104\n\n");
	SW_conv3_block8_1_conv(O103_SW,O104_SW,W104);
	printf("[C_verifier.cpp]Calculate BatchNormalization105\n\n");
	SW_conv3_block8_1_bn(O104_SW,O105_SW, W105);
	printf("[C_verifier.cpp]Calculate Activation(Relu)106\n\n");
	SW_conv3_block8_1_relu(O105_SW,O106_SW);
	printf("[C_verifier.cpp]Calculate Conv2D107\n\n");
	SW_conv3_block8_2_conv(O106_SW,O107_SW,W107);
	printf("[C_verifier.cpp]Calculate Concatenate108\n\n");
	SW_conv3_block8_concat(O101_SW, O107_SW, O108_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization109\n\n");
	SW_conv3_block9_0_bn(O108_SW,O109_SW, W109);
	printf("[C_verifier.cpp]Calculate Activation(Relu)110\n\n");
	SW_conv3_block9_0_relu(O109_SW,O110_SW);
	printf("[C_verifier.cpp]Calculate Conv2D111\n\n");
	SW_conv3_block9_1_conv(O110_SW,O111_SW,W111);
	printf("[C_verifier.cpp]Calculate BatchNormalization112\n\n");
	SW_conv3_block9_1_bn(O111_SW,O112_SW, W112);
	printf("[C_verifier.cpp]Calculate Activation(Relu)113\n\n");
	SW_conv3_block9_1_relu(O112_SW,O113_SW);
	printf("[C_verifier.cpp]Calculate Conv2D114\n\n");
	SW_conv3_block9_2_conv(O113_SW,O114_SW,W114);
	printf("[C_verifier.cpp]Calculate Concatenate115\n\n");
	SW_conv3_block9_concat(O108_SW, O114_SW, O115_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization116\n\n");
	SW_conv3_block10_0_bn(O115_SW,O116_SW, W116);
	printf("[C_verifier.cpp]Calculate Activation(Relu)117\n\n");
	SW_conv3_block10_0_relu(O116_SW,O117_SW);
	printf("[C_verifier.cpp]Calculate Conv2D118\n\n");
	SW_conv3_block10_1_conv(O117_SW,O118_SW,W118);
	printf("[C_verifier.cpp]Calculate BatchNormalization119\n\n");
	SW_conv3_block10_1_bn(O118_SW,O119_SW, W119);
	printf("[C_verifier.cpp]Calculate Activation(Relu)120\n\n");
	SW_conv3_block10_1_relu(O119_SW,O120_SW);
	printf("[C_verifier.cpp]Calculate Conv2D121\n\n");
	SW_conv3_block10_2_conv(O120_SW,O121_SW,W121);
	printf("[C_verifier.cpp]Calculate Concatenate122\n\n");
	SW_conv3_block10_concat(O115_SW, O121_SW, O122_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization123\n\n");
	SW_conv3_block11_0_bn(O122_SW,O123_SW, W123);
	printf("[C_verifier.cpp]Calculate Activation(Relu)124\n\n");
	SW_conv3_block11_0_relu(O123_SW,O124_SW);
	printf("[C_verifier.cpp]Calculate Conv2D125\n\n");
	SW_conv3_block11_1_conv(O124_SW,O125_SW,W125);
	printf("[C_verifier.cpp]Calculate BatchNormalization126\n\n");
	SW_conv3_block11_1_bn(O125_SW,O126_SW, W126);
	printf("[C_verifier.cpp]Calculate Activation(Relu)127\n\n");
	SW_conv3_block11_1_relu(O126_SW,O127_SW);
	printf("[C_verifier.cpp]Calculate Conv2D128\n\n");
	SW_conv3_block11_2_conv(O127_SW,O128_SW,W128);
	printf("[C_verifier.cpp]Calculate Concatenate129\n\n");
	SW_conv3_block11_concat(O122_SW, O128_SW, O129_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization130\n\n");
	SW_conv3_block12_0_bn(O129_SW,O130_SW, W130);
	printf("[C_verifier.cpp]Calculate Activation(Relu)131\n\n");
	SW_conv3_block12_0_relu(O130_SW,O131_SW);
	printf("[C_verifier.cpp]Calculate Conv2D132\n\n");
	SW_conv3_block12_1_conv(O131_SW,O132_SW,W132);
	printf("[C_verifier.cpp]Calculate BatchNormalization133\n\n");
	SW_conv3_block12_1_bn(O132_SW,O133_SW, W133);
	printf("[C_verifier.cpp]Calculate Activation(Relu)134\n\n");
	SW_conv3_block12_1_relu(O133_SW,O134_SW);
	printf("[C_verifier.cpp]Calculate Conv2D135\n\n");
	SW_conv3_block12_2_conv(O134_SW,O135_SW,W135);
	printf("[C_verifier.cpp]Calculate Concatenate136\n\n");
	SW_conv3_block12_concat(O129_SW, O135_SW, O136_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization137\n\n");
	SW_pool3_bn(O136_SW,O137_SW, W137);
	printf("[C_verifier.cpp]Calculate Activation(Relu)138\n\n");
	SW_pool3_relu(O137_SW,O138_SW);
	printf("[C_verifier.cpp]Calculate Conv2D139\n\n");
	SW_pool3_conv(O138_SW,O139_SW,W139);
	printf("[C_verifier.cpp]Calculate AveragePooling2D140\n\n");
	SW_pool3_pool(O139_SW,O140_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization141\n\n");
	SW_conv4_block1_0_bn(O140_SW,O141_SW, W141);
	printf("[C_verifier.cpp]Calculate Activation(Relu)142\n\n");
	SW_conv4_block1_0_relu(O141_SW,O142_SW);
	printf("[C_verifier.cpp]Calculate Conv2D143\n\n");
	SW_conv4_block1_1_conv(O142_SW,O143_SW,W143);
	printf("[C_verifier.cpp]Calculate BatchNormalization144\n\n");
	SW_conv4_block1_1_bn(O143_SW,O144_SW, W144);
	printf("[C_verifier.cpp]Calculate Activation(Relu)145\n\n");
	SW_conv4_block1_1_relu(O144_SW,O145_SW);
	printf("[C_verifier.cpp]Calculate Conv2D146\n\n");
	SW_conv4_block1_2_conv(O145_SW,O146_SW,W146);
	printf("[C_verifier.cpp]Calculate Concatenate147\n\n");
	SW_conv4_block1_concat(O140_SW, O146_SW, O147_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization148\n\n");
	SW_conv4_block2_0_bn(O147_SW,O148_SW, W148);
	printf("[C_verifier.cpp]Calculate Activation(Relu)149\n\n");
	SW_conv4_block2_0_relu(O148_SW,O149_SW);
	printf("[C_verifier.cpp]Calculate Conv2D150\n\n");
	SW_conv4_block2_1_conv(O149_SW,O150_SW,W150);
	printf("[C_verifier.cpp]Calculate BatchNormalization151\n\n");
	SW_conv4_block2_1_bn(O150_SW,O151_SW, W151);
	printf("[C_verifier.cpp]Calculate Activation(Relu)152\n\n");
	SW_conv4_block2_1_relu(O151_SW,O152_SW);
	printf("[C_verifier.cpp]Calculate Conv2D153\n\n");
	SW_conv4_block2_2_conv(O152_SW,O153_SW,W153);
	printf("[C_verifier.cpp]Calculate Concatenate154\n\n");
	SW_conv4_block2_concat(O147_SW, O153_SW, O154_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization155\n\n");
	SW_conv4_block3_0_bn(O154_SW,O155_SW, W155);
	printf("[C_verifier.cpp]Calculate Activation(Relu)156\n\n");
	SW_conv4_block3_0_relu(O155_SW,O156_SW);
	printf("[C_verifier.cpp]Calculate Conv2D157\n\n");
	SW_conv4_block3_1_conv(O156_SW,O157_SW,W157);
	printf("[C_verifier.cpp]Calculate BatchNormalization158\n\n");
	SW_conv4_block3_1_bn(O157_SW,O158_SW, W158);
	printf("[C_verifier.cpp]Calculate Activation(Relu)159\n\n");
	SW_conv4_block3_1_relu(O158_SW,O159_SW);
	printf("[C_verifier.cpp]Calculate Conv2D160\n\n");
	SW_conv4_block3_2_conv(O159_SW,O160_SW,W160);
	printf("[C_verifier.cpp]Calculate Concatenate161\n\n");
	SW_conv4_block3_concat(O154_SW, O160_SW, O161_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization162\n\n");
	SW_conv4_block4_0_bn(O161_SW,O162_SW, W162);
	printf("[C_verifier.cpp]Calculate Activation(Relu)163\n\n");
	SW_conv4_block4_0_relu(O162_SW,O163_SW);
	printf("[C_verifier.cpp]Calculate Conv2D164\n\n");
	SW_conv4_block4_1_conv(O163_SW,O164_SW,W164);
	printf("[C_verifier.cpp]Calculate BatchNormalization165\n\n");
	SW_conv4_block4_1_bn(O164_SW,O165_SW, W165);
	printf("[C_verifier.cpp]Calculate Activation(Relu)166\n\n");
	SW_conv4_block4_1_relu(O165_SW,O166_SW);
	printf("[C_verifier.cpp]Calculate Conv2D167\n\n");
	SW_conv4_block4_2_conv(O166_SW,O167_SW,W167);
	printf("[C_verifier.cpp]Calculate Concatenate168\n\n");
	SW_conv4_block4_concat(O161_SW, O167_SW, O168_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization169\n\n");
	SW_conv4_block5_0_bn(O168_SW,O169_SW, W169);
	printf("[C_verifier.cpp]Calculate Activation(Relu)170\n\n");
	SW_conv4_block5_0_relu(O169_SW,O170_SW);
	printf("[C_verifier.cpp]Calculate Conv2D171\n\n");
	SW_conv4_block5_1_conv(O170_SW,O171_SW,W171);
	printf("[C_verifier.cpp]Calculate BatchNormalization172\n\n");
	SW_conv4_block5_1_bn(O171_SW,O172_SW, W172);
	printf("[C_verifier.cpp]Calculate Activation(Relu)173\n\n");
	SW_conv4_block5_1_relu(O172_SW,O173_SW);
	printf("[C_verifier.cpp]Calculate Conv2D174\n\n");
	SW_conv4_block5_2_conv(O173_SW,O174_SW,W174);
	printf("[C_verifier.cpp]Calculate Concatenate175\n\n");
	SW_conv4_block5_concat(O168_SW, O174_SW, O175_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization176\n\n");
	SW_conv4_block6_0_bn(O175_SW,O176_SW, W176);
	printf("[C_verifier.cpp]Calculate Activation(Relu)177\n\n");
	SW_conv4_block6_0_relu(O176_SW,O177_SW);
	printf("[C_verifier.cpp]Calculate Conv2D178\n\n");
	SW_conv4_block6_1_conv(O177_SW,O178_SW,W178);
	printf("[C_verifier.cpp]Calculate BatchNormalization179\n\n");
	SW_conv4_block6_1_bn(O178_SW,O179_SW, W179);
	printf("[C_verifier.cpp]Calculate Activation(Relu)180\n\n");
	SW_conv4_block6_1_relu(O179_SW,O180_SW);
	printf("[C_verifier.cpp]Calculate Conv2D181\n\n");
	SW_conv4_block6_2_conv(O180_SW,O181_SW,W181);
	printf("[C_verifier.cpp]Calculate Concatenate182\n\n");
	SW_conv4_block6_concat(O175_SW, O181_SW, O182_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization183\n\n");
	SW_conv4_block7_0_bn(O182_SW,O183_SW, W183);
	printf("[C_verifier.cpp]Calculate Activation(Relu)184\n\n");
	SW_conv4_block7_0_relu(O183_SW,O184_SW);
	printf("[C_verifier.cpp]Calculate Conv2D185\n\n");
	SW_conv4_block7_1_conv(O184_SW,O185_SW,W185);
	printf("[C_verifier.cpp]Calculate BatchNormalization186\n\n");
	SW_conv4_block7_1_bn(O185_SW,O186_SW, W186);
	printf("[C_verifier.cpp]Calculate Activation(Relu)187\n\n");
	SW_conv4_block7_1_relu(O186_SW,O187_SW);
	printf("[C_verifier.cpp]Calculate Conv2D188\n\n");
	SW_conv4_block7_2_conv(O187_SW,O188_SW,W188);
	printf("[C_verifier.cpp]Calculate Concatenate189\n\n");
	SW_conv4_block7_concat(O182_SW, O188_SW, O189_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization190\n\n");
	SW_conv4_block8_0_bn(O189_SW,O190_SW, W190);
	printf("[C_verifier.cpp]Calculate Activation(Relu)191\n\n");
	SW_conv4_block8_0_relu(O190_SW,O191_SW);
	printf("[C_verifier.cpp]Calculate Conv2D192\n\n");
	SW_conv4_block8_1_conv(O191_SW,O192_SW,W192);
	printf("[C_verifier.cpp]Calculate BatchNormalization193\n\n");
	SW_conv4_block8_1_bn(O192_SW,O193_SW, W193);
	printf("[C_verifier.cpp]Calculate Activation(Relu)194\n\n");
	SW_conv4_block8_1_relu(O193_SW,O194_SW);
	printf("[C_verifier.cpp]Calculate Conv2D195\n\n");
	SW_conv4_block8_2_conv(O194_SW,O195_SW,W195);
	printf("[C_verifier.cpp]Calculate Concatenate196\n\n");
	SW_conv4_block8_concat(O189_SW, O195_SW, O196_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization197\n\n");
	SW_conv4_block9_0_bn(O196_SW,O197_SW, W197);
	printf("[C_verifier.cpp]Calculate Activation(Relu)198\n\n");
	SW_conv4_block9_0_relu(O197_SW,O198_SW);
	printf("[C_verifier.cpp]Calculate Conv2D199\n\n");
	SW_conv4_block9_1_conv(O198_SW,O199_SW,W199);
	printf("[C_verifier.cpp]Calculate BatchNormalization200\n\n");
	SW_conv4_block9_1_bn(O199_SW,O200_SW, W200);
	printf("[C_verifier.cpp]Calculate Activation(Relu)201\n\n");
	SW_conv4_block9_1_relu(O200_SW,O201_SW);
	printf("[C_verifier.cpp]Calculate Conv2D202\n\n");
	SW_conv4_block9_2_conv(O201_SW,O202_SW,W202);
	printf("[C_verifier.cpp]Calculate Concatenate203\n\n");
	SW_conv4_block9_concat(O196_SW, O202_SW, O203_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization204\n\n");
	SW_conv4_block10_0_bn(O203_SW,O204_SW, W204);
	printf("[C_verifier.cpp]Calculate Activation(Relu)205\n\n");
	SW_conv4_block10_0_relu(O204_SW,O205_SW);
	printf("[C_verifier.cpp]Calculate Conv2D206\n\n");
	SW_conv4_block10_1_conv(O205_SW,O206_SW,W206);
	printf("[C_verifier.cpp]Calculate BatchNormalization207\n\n");
	SW_conv4_block10_1_bn(O206_SW,O207_SW, W207);
	printf("[C_verifier.cpp]Calculate Activation(Relu)208\n\n");
	SW_conv4_block10_1_relu(O207_SW,O208_SW);
	printf("[C_verifier.cpp]Calculate Conv2D209\n\n");
	SW_conv4_block10_2_conv(O208_SW,O209_SW,W209);
	printf("[C_verifier.cpp]Calculate Concatenate210\n\n");
	SW_conv4_block10_concat(O203_SW, O209_SW, O210_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization211\n\n");
	SW_conv4_block11_0_bn(O210_SW,O211_SW, W211);
	printf("[C_verifier.cpp]Calculate Activation(Relu)212\n\n");
	SW_conv4_block11_0_relu(O211_SW,O212_SW);
	printf("[C_verifier.cpp]Calculate Conv2D213\n\n");
	SW_conv4_block11_1_conv(O212_SW,O213_SW,W213);
	printf("[C_verifier.cpp]Calculate BatchNormalization214\n\n");
	SW_conv4_block11_1_bn(O213_SW,O214_SW, W214);
	printf("[C_verifier.cpp]Calculate Activation(Relu)215\n\n");
	SW_conv4_block11_1_relu(O214_SW,O215_SW);
	printf("[C_verifier.cpp]Calculate Conv2D216\n\n");
	SW_conv4_block11_2_conv(O215_SW,O216_SW,W216);
	printf("[C_verifier.cpp]Calculate Concatenate217\n\n");
	SW_conv4_block11_concat(O210_SW, O216_SW, O217_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization218\n\n");
	SW_conv4_block12_0_bn(O217_SW,O218_SW, W218);
	printf("[C_verifier.cpp]Calculate Activation(Relu)219\n\n");
	SW_conv4_block12_0_relu(O218_SW,O219_SW);
	printf("[C_verifier.cpp]Calculate Conv2D220\n\n");
	SW_conv4_block12_1_conv(O219_SW,O220_SW,W220);
	printf("[C_verifier.cpp]Calculate BatchNormalization221\n\n");
	SW_conv4_block12_1_bn(O220_SW,O221_SW, W221);
	printf("[C_verifier.cpp]Calculate Activation(Relu)222\n\n");
	SW_conv4_block12_1_relu(O221_SW,O222_SW);
	printf("[C_verifier.cpp]Calculate Conv2D223\n\n");
	SW_conv4_block12_2_conv(O222_SW,O223_SW,W223);
	printf("[C_verifier.cpp]Calculate Concatenate224\n\n");
	SW_conv4_block12_concat(O217_SW, O223_SW, O224_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization225\n\n");
	SW_conv4_block13_0_bn(O224_SW,O225_SW, W225);
	printf("[C_verifier.cpp]Calculate Activation(Relu)226\n\n");
	SW_conv4_block13_0_relu(O225_SW,O226_SW);
	printf("[C_verifier.cpp]Calculate Conv2D227\n\n");
	SW_conv4_block13_1_conv(O226_SW,O227_SW,W227);
	printf("[C_verifier.cpp]Calculate BatchNormalization228\n\n");
	SW_conv4_block13_1_bn(O227_SW,O228_SW, W228);
	printf("[C_verifier.cpp]Calculate Activation(Relu)229\n\n");
	SW_conv4_block13_1_relu(O228_SW,O229_SW);
	printf("[C_verifier.cpp]Calculate Conv2D230\n\n");
	SW_conv4_block13_2_conv(O229_SW,O230_SW,W230);
	printf("[C_verifier.cpp]Calculate Concatenate231\n\n");
	SW_conv4_block13_concat(O224_SW, O230_SW, O231_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization232\n\n");
	SW_conv4_block14_0_bn(O231_SW,O232_SW, W232);
	printf("[C_verifier.cpp]Calculate Activation(Relu)233\n\n");
	SW_conv4_block14_0_relu(O232_SW,O233_SW);
	printf("[C_verifier.cpp]Calculate Conv2D234\n\n");
	SW_conv4_block14_1_conv(O233_SW,O234_SW,W234);
	printf("[C_verifier.cpp]Calculate BatchNormalization235\n\n");
	SW_conv4_block14_1_bn(O234_SW,O235_SW, W235);
	printf("[C_verifier.cpp]Calculate Activation(Relu)236\n\n");
	SW_conv4_block14_1_relu(O235_SW,O236_SW);
	printf("[C_verifier.cpp]Calculate Conv2D237\n\n");
	SW_conv4_block14_2_conv(O236_SW,O237_SW,W237);
	printf("[C_verifier.cpp]Calculate Concatenate238\n\n");
	SW_conv4_block14_concat(O231_SW, O237_SW, O238_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization239\n\n");
	SW_conv4_block15_0_bn(O238_SW,O239_SW, W239);
	printf("[C_verifier.cpp]Calculate Activation(Relu)240\n\n");
	SW_conv4_block15_0_relu(O239_SW,O240_SW);
	printf("[C_verifier.cpp]Calculate Conv2D241\n\n");
	SW_conv4_block15_1_conv(O240_SW,O241_SW,W241);
	printf("[C_verifier.cpp]Calculate BatchNormalization242\n\n");
	SW_conv4_block15_1_bn(O241_SW,O242_SW, W242);
	printf("[C_verifier.cpp]Calculate Activation(Relu)243\n\n");
	SW_conv4_block15_1_relu(O242_SW,O243_SW);
	printf("[C_verifier.cpp]Calculate Conv2D244\n\n");
	SW_conv4_block15_2_conv(O243_SW,O244_SW,W244);
	printf("[C_verifier.cpp]Calculate Concatenate245\n\n");
	SW_conv4_block15_concat(O238_SW, O244_SW, O245_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization246\n\n");
	SW_conv4_block16_0_bn(O245_SW,O246_SW, W246);
	printf("[C_verifier.cpp]Calculate Activation(Relu)247\n\n");
	SW_conv4_block16_0_relu(O246_SW,O247_SW);
	printf("[C_verifier.cpp]Calculate Conv2D248\n\n");
	SW_conv4_block16_1_conv(O247_SW,O248_SW,W248);
	printf("[C_verifier.cpp]Calculate BatchNormalization249\n\n");
	SW_conv4_block16_1_bn(O248_SW,O249_SW, W249);
	printf("[C_verifier.cpp]Calculate Activation(Relu)250\n\n");
	SW_conv4_block16_1_relu(O249_SW,O250_SW);
	printf("[C_verifier.cpp]Calculate Conv2D251\n\n");
	SW_conv4_block16_2_conv(O250_SW,O251_SW,W251);
	printf("[C_verifier.cpp]Calculate Concatenate252\n\n");
	SW_conv4_block16_concat(O245_SW, O251_SW, O252_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization253\n\n");
	SW_conv4_block17_0_bn(O252_SW,O253_SW, W253);
	printf("[C_verifier.cpp]Calculate Activation(Relu)254\n\n");
	SW_conv4_block17_0_relu(O253_SW,O254_SW);
	printf("[C_verifier.cpp]Calculate Conv2D255\n\n");
	SW_conv4_block17_1_conv(O254_SW,O255_SW,W255);
	printf("[C_verifier.cpp]Calculate BatchNormalization256\n\n");
	SW_conv4_block17_1_bn(O255_SW,O256_SW, W256);
	printf("[C_verifier.cpp]Calculate Activation(Relu)257\n\n");
	SW_conv4_block17_1_relu(O256_SW,O257_SW);
	printf("[C_verifier.cpp]Calculate Conv2D258\n\n");
	SW_conv4_block17_2_conv(O257_SW,O258_SW,W258);
	printf("[C_verifier.cpp]Calculate Concatenate259\n\n");
	SW_conv4_block17_concat(O252_SW, O258_SW, O259_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization260\n\n");
	SW_conv4_block18_0_bn(O259_SW,O260_SW, W260);
	printf("[C_verifier.cpp]Calculate Activation(Relu)261\n\n");
	SW_conv4_block18_0_relu(O260_SW,O261_SW);
	printf("[C_verifier.cpp]Calculate Conv2D262\n\n");
	SW_conv4_block18_1_conv(O261_SW,O262_SW,W262);
	printf("[C_verifier.cpp]Calculate BatchNormalization263\n\n");
	SW_conv4_block18_1_bn(O262_SW,O263_SW, W263);
	printf("[C_verifier.cpp]Calculate Activation(Relu)264\n\n");
	SW_conv4_block18_1_relu(O263_SW,O264_SW);
	printf("[C_verifier.cpp]Calculate Conv2D265\n\n");
	SW_conv4_block18_2_conv(O264_SW,O265_SW,W265);
	printf("[C_verifier.cpp]Calculate Concatenate266\n\n");
	SW_conv4_block18_concat(O259_SW, O265_SW, O266_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization267\n\n");
	SW_conv4_block19_0_bn(O266_SW,O267_SW, W267);
	printf("[C_verifier.cpp]Calculate Activation(Relu)268\n\n");
	SW_conv4_block19_0_relu(O267_SW,O268_SW);
	printf("[C_verifier.cpp]Calculate Conv2D269\n\n");
	SW_conv4_block19_1_conv(O268_SW,O269_SW,W269);
	printf("[C_verifier.cpp]Calculate BatchNormalization270\n\n");
	SW_conv4_block19_1_bn(O269_SW,O270_SW, W270);
	printf("[C_verifier.cpp]Calculate Activation(Relu)271\n\n");
	SW_conv4_block19_1_relu(O270_SW,O271_SW);
	printf("[C_verifier.cpp]Calculate Conv2D272\n\n");
	SW_conv4_block19_2_conv(O271_SW,O272_SW,W272);
	printf("[C_verifier.cpp]Calculate Concatenate273\n\n");
	SW_conv4_block19_concat(O266_SW, O272_SW, O273_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization274\n\n");
	SW_conv4_block20_0_bn(O273_SW,O274_SW, W274);
	printf("[C_verifier.cpp]Calculate Activation(Relu)275\n\n");
	SW_conv4_block20_0_relu(O274_SW,O275_SW);
	printf("[C_verifier.cpp]Calculate Conv2D276\n\n");
	SW_conv4_block20_1_conv(O275_SW,O276_SW,W276);
	printf("[C_verifier.cpp]Calculate BatchNormalization277\n\n");
	SW_conv4_block20_1_bn(O276_SW,O277_SW, W277);
	printf("[C_verifier.cpp]Calculate Activation(Relu)278\n\n");
	SW_conv4_block20_1_relu(O277_SW,O278_SW);
	printf("[C_verifier.cpp]Calculate Conv2D279\n\n");
	SW_conv4_block20_2_conv(O278_SW,O279_SW,W279);
	printf("[C_verifier.cpp]Calculate Concatenate280\n\n");
	SW_conv4_block20_concat(O273_SW, O279_SW, O280_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization281\n\n");
	SW_conv4_block21_0_bn(O280_SW,O281_SW, W281);
	printf("[C_verifier.cpp]Calculate Activation(Relu)282\n\n");
	SW_conv4_block21_0_relu(O281_SW,O282_SW);
	printf("[C_verifier.cpp]Calculate Conv2D283\n\n");
	SW_conv4_block21_1_conv(O282_SW,O283_SW,W283);
	printf("[C_verifier.cpp]Calculate BatchNormalization284\n\n");
	SW_conv4_block21_1_bn(O283_SW,O284_SW, W284);
	printf("[C_verifier.cpp]Calculate Activation(Relu)285\n\n");
	SW_conv4_block21_1_relu(O284_SW,O285_SW);
	printf("[C_verifier.cpp]Calculate Conv2D286\n\n");
	SW_conv4_block21_2_conv(O285_SW,O286_SW,W286);
	printf("[C_verifier.cpp]Calculate Concatenate287\n\n");
	SW_conv4_block21_concat(O280_SW, O286_SW, O287_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization288\n\n");
	SW_conv4_block22_0_bn(O287_SW,O288_SW, W288);
	printf("[C_verifier.cpp]Calculate Activation(Relu)289\n\n");
	SW_conv4_block22_0_relu(O288_SW,O289_SW);
	printf("[C_verifier.cpp]Calculate Conv2D290\n\n");
	SW_conv4_block22_1_conv(O289_SW,O290_SW,W290);
	printf("[C_verifier.cpp]Calculate BatchNormalization291\n\n");
	SW_conv4_block22_1_bn(O290_SW,O291_SW, W291);
	printf("[C_verifier.cpp]Calculate Activation(Relu)292\n\n");
	SW_conv4_block22_1_relu(O291_SW,O292_SW);
	printf("[C_verifier.cpp]Calculate Conv2D293\n\n");
	SW_conv4_block22_2_conv(O292_SW,O293_SW,W293);
	printf("[C_verifier.cpp]Calculate Concatenate294\n\n");
	SW_conv4_block22_concat(O287_SW, O293_SW, O294_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization295\n\n");
	SW_conv4_block23_0_bn(O294_SW,O295_SW, W295);
	printf("[C_verifier.cpp]Calculate Activation(Relu)296\n\n");
	SW_conv4_block23_0_relu(O295_SW,O296_SW);
	printf("[C_verifier.cpp]Calculate Conv2D297\n\n");
	SW_conv4_block23_1_conv(O296_SW,O297_SW,W297);
	printf("[C_verifier.cpp]Calculate BatchNormalization298\n\n");
	SW_conv4_block23_1_bn(O297_SW,O298_SW, W298);
	printf("[C_verifier.cpp]Calculate Activation(Relu)299\n\n");
	SW_conv4_block23_1_relu(O298_SW,O299_SW);
	printf("[C_verifier.cpp]Calculate Conv2D300\n\n");
	SW_conv4_block23_2_conv(O299_SW,O300_SW,W300);
	printf("[C_verifier.cpp]Calculate Concatenate301\n\n");
	SW_conv4_block23_concat(O294_SW, O300_SW, O301_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization302\n\n");
	SW_conv4_block24_0_bn(O301_SW,O302_SW, W302);
	printf("[C_verifier.cpp]Calculate Activation(Relu)303\n\n");
	SW_conv4_block24_0_relu(O302_SW,O303_SW);
	printf("[C_verifier.cpp]Calculate Conv2D304\n\n");
	SW_conv4_block24_1_conv(O303_SW,O304_SW,W304);
	printf("[C_verifier.cpp]Calculate BatchNormalization305\n\n");
	SW_conv4_block24_1_bn(O304_SW,O305_SW, W305);
	printf("[C_verifier.cpp]Calculate Activation(Relu)306\n\n");
	SW_conv4_block24_1_relu(O305_SW,O306_SW);
	printf("[C_verifier.cpp]Calculate Conv2D307\n\n");
	SW_conv4_block24_2_conv(O306_SW,O307_SW,W307);
	printf("[C_verifier.cpp]Calculate Concatenate308\n\n");
	SW_conv4_block24_concat(O301_SW, O307_SW, O308_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization309\n\n");
	SW_conv4_block25_0_bn(O308_SW,O309_SW, W309);
	printf("[C_verifier.cpp]Calculate Activation(Relu)310\n\n");
	SW_conv4_block25_0_relu(O309_SW,O310_SW);
	printf("[C_verifier.cpp]Calculate Conv2D311\n\n");
	SW_conv4_block25_1_conv(O310_SW,O311_SW,W311);
	printf("[C_verifier.cpp]Calculate BatchNormalization312\n\n");
	SW_conv4_block25_1_bn(O311_SW,O312_SW, W312);
	printf("[C_verifier.cpp]Calculate Activation(Relu)313\n\n");
	SW_conv4_block25_1_relu(O312_SW,O313_SW);
	printf("[C_verifier.cpp]Calculate Conv2D314\n\n");
	SW_conv4_block25_2_conv(O313_SW,O314_SW,W314);
	printf("[C_verifier.cpp]Calculate Concatenate315\n\n");
	SW_conv4_block25_concat(O308_SW, O314_SW, O315_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization316\n\n");
	SW_conv4_block26_0_bn(O315_SW,O316_SW, W316);
	printf("[C_verifier.cpp]Calculate Activation(Relu)317\n\n");
	SW_conv4_block26_0_relu(O316_SW,O317_SW);
	printf("[C_verifier.cpp]Calculate Conv2D318\n\n");
	SW_conv4_block26_1_conv(O317_SW,O318_SW,W318);
	printf("[C_verifier.cpp]Calculate BatchNormalization319\n\n");
	SW_conv4_block26_1_bn(O318_SW,O319_SW, W319);
	printf("[C_verifier.cpp]Calculate Activation(Relu)320\n\n");
	SW_conv4_block26_1_relu(O319_SW,O320_SW);
	printf("[C_verifier.cpp]Calculate Conv2D321\n\n");
	SW_conv4_block26_2_conv(O320_SW,O321_SW,W321);
	printf("[C_verifier.cpp]Calculate Concatenate322\n\n");
	SW_conv4_block26_concat(O315_SW, O321_SW, O322_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization323\n\n");
	SW_conv4_block27_0_bn(O322_SW,O323_SW, W323);
	printf("[C_verifier.cpp]Calculate Activation(Relu)324\n\n");
	SW_conv4_block27_0_relu(O323_SW,O324_SW);
	printf("[C_verifier.cpp]Calculate Conv2D325\n\n");
	SW_conv4_block27_1_conv(O324_SW,O325_SW,W325);
	printf("[C_verifier.cpp]Calculate BatchNormalization326\n\n");
	SW_conv4_block27_1_bn(O325_SW,O326_SW, W326);
	printf("[C_verifier.cpp]Calculate Activation(Relu)327\n\n");
	SW_conv4_block27_1_relu(O326_SW,O327_SW);
	printf("[C_verifier.cpp]Calculate Conv2D328\n\n");
	SW_conv4_block27_2_conv(O327_SW,O328_SW,W328);
	printf("[C_verifier.cpp]Calculate Concatenate329\n\n");
	SW_conv4_block27_concat(O322_SW, O328_SW, O329_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization330\n\n");
	SW_conv4_block28_0_bn(O329_SW,O330_SW, W330);
	printf("[C_verifier.cpp]Calculate Activation(Relu)331\n\n");
	SW_conv4_block28_0_relu(O330_SW,O331_SW);
	printf("[C_verifier.cpp]Calculate Conv2D332\n\n");
	SW_conv4_block28_1_conv(O331_SW,O332_SW,W332);
	printf("[C_verifier.cpp]Calculate BatchNormalization333\n\n");
	SW_conv4_block28_1_bn(O332_SW,O333_SW, W333);
	printf("[C_verifier.cpp]Calculate Activation(Relu)334\n\n");
	SW_conv4_block28_1_relu(O333_SW,O334_SW);
	printf("[C_verifier.cpp]Calculate Conv2D335\n\n");
	SW_conv4_block28_2_conv(O334_SW,O335_SW,W335);
	printf("[C_verifier.cpp]Calculate Concatenate336\n\n");
	SW_conv4_block28_concat(O329_SW, O335_SW, O336_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization337\n\n");
	SW_conv4_block29_0_bn(O336_SW,O337_SW, W337);
	printf("[C_verifier.cpp]Calculate Activation(Relu)338\n\n");
	SW_conv4_block29_0_relu(O337_SW,O338_SW);
	printf("[C_verifier.cpp]Calculate Conv2D339\n\n");
	SW_conv4_block29_1_conv(O338_SW,O339_SW,W339);
	printf("[C_verifier.cpp]Calculate BatchNormalization340\n\n");
	SW_conv4_block29_1_bn(O339_SW,O340_SW, W340);
	printf("[C_verifier.cpp]Calculate Activation(Relu)341\n\n");
	SW_conv4_block29_1_relu(O340_SW,O341_SW);
	printf("[C_verifier.cpp]Calculate Conv2D342\n\n");
	SW_conv4_block29_2_conv(O341_SW,O342_SW,W342);
	printf("[C_verifier.cpp]Calculate Concatenate343\n\n");
	SW_conv4_block29_concat(O336_SW, O342_SW, O343_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization344\n\n");
	SW_conv4_block30_0_bn(O343_SW,O344_SW, W344);
	printf("[C_verifier.cpp]Calculate Activation(Relu)345\n\n");
	SW_conv4_block30_0_relu(O344_SW,O345_SW);
	printf("[C_verifier.cpp]Calculate Conv2D346\n\n");
	SW_conv4_block30_1_conv(O345_SW,O346_SW,W346);
	printf("[C_verifier.cpp]Calculate BatchNormalization347\n\n");
	SW_conv4_block30_1_bn(O346_SW,O347_SW, W347);
	printf("[C_verifier.cpp]Calculate Activation(Relu)348\n\n");
	SW_conv4_block30_1_relu(O347_SW,O348_SW);
	printf("[C_verifier.cpp]Calculate Conv2D349\n\n");
	SW_conv4_block30_2_conv(O348_SW,O349_SW,W349);
	printf("[C_verifier.cpp]Calculate Concatenate350\n\n");
	SW_conv4_block30_concat(O343_SW, O349_SW, O350_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization351\n\n");
	SW_conv4_block31_0_bn(O350_SW,O351_SW, W351);
	printf("[C_verifier.cpp]Calculate Activation(Relu)352\n\n");
	SW_conv4_block31_0_relu(O351_SW,O352_SW);
	printf("[C_verifier.cpp]Calculate Conv2D353\n\n");
	SW_conv4_block31_1_conv(O352_SW,O353_SW,W353);
	printf("[C_verifier.cpp]Calculate BatchNormalization354\n\n");
	SW_conv4_block31_1_bn(O353_SW,O354_SW, W354);
	printf("[C_verifier.cpp]Calculate Activation(Relu)355\n\n");
	SW_conv4_block31_1_relu(O354_SW,O355_SW);
	printf("[C_verifier.cpp]Calculate Conv2D356\n\n");
	SW_conv4_block31_2_conv(O355_SW,O356_SW,W356);
	printf("[C_verifier.cpp]Calculate Concatenate357\n\n");
	SW_conv4_block31_concat(O350_SW, O356_SW, O357_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization358\n\n");
	SW_conv4_block32_0_bn(O357_SW,O358_SW, W358);
	printf("[C_verifier.cpp]Calculate Activation(Relu)359\n\n");
	SW_conv4_block32_0_relu(O358_SW,O359_SW);
	printf("[C_verifier.cpp]Calculate Conv2D360\n\n");
	SW_conv4_block32_1_conv(O359_SW,O360_SW,W360);
	printf("[C_verifier.cpp]Calculate BatchNormalization361\n\n");
	SW_conv4_block32_1_bn(O360_SW,O361_SW, W361);
	printf("[C_verifier.cpp]Calculate Activation(Relu)362\n\n");
	SW_conv4_block32_1_relu(O361_SW,O362_SW);
	printf("[C_verifier.cpp]Calculate Conv2D363\n\n");
	SW_conv4_block32_2_conv(O362_SW,O363_SW,W363);
	printf("[C_verifier.cpp]Calculate Concatenate364\n\n");
	SW_conv4_block32_concat(O357_SW, O363_SW, O364_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization365\n\n");
	SW_pool4_bn(O364_SW,O365_SW, W365);
	printf("[C_verifier.cpp]Calculate Activation(Relu)366\n\n");
	SW_pool4_relu(O365_SW,O366_SW);
	printf("[C_verifier.cpp]Calculate Conv2D367\n\n");
	SW_pool4_conv(O366_SW,O367_SW,W367);
	printf("[C_verifier.cpp]Calculate AveragePooling2D368\n\n");
	SW_pool4_pool(O367_SW,O368_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization369\n\n");
	SW_conv5_block1_0_bn(O368_SW,O369_SW, W369);
	printf("[C_verifier.cpp]Calculate Activation(Relu)370\n\n");
	SW_conv5_block1_0_relu(O369_SW,O370_SW);
	printf("[C_verifier.cpp]Calculate Conv2D371\n\n");
	SW_conv5_block1_1_conv(O370_SW,O371_SW,W371);
	printf("[C_verifier.cpp]Calculate BatchNormalization372\n\n");
	SW_conv5_block1_1_bn(O371_SW,O372_SW, W372);
	printf("[C_verifier.cpp]Calculate Activation(Relu)373\n\n");
	SW_conv5_block1_1_relu(O372_SW,O373_SW);
	printf("[C_verifier.cpp]Calculate Conv2D374\n\n");
	SW_conv5_block1_2_conv(O373_SW,O374_SW,W374);
	printf("[C_verifier.cpp]Calculate Concatenate375\n\n");
	SW_conv5_block1_concat(O368_SW, O374_SW, O375_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization376\n\n");
	SW_conv5_block2_0_bn(O375_SW,O376_SW, W376);
	printf("[C_verifier.cpp]Calculate Activation(Relu)377\n\n");
	SW_conv5_block2_0_relu(O376_SW,O377_SW);
	printf("[C_verifier.cpp]Calculate Conv2D378\n\n");
	SW_conv5_block2_1_conv(O377_SW,O378_SW,W378);
	printf("[C_verifier.cpp]Calculate BatchNormalization379\n\n");
	SW_conv5_block2_1_bn(O378_SW,O379_SW, W379);
	printf("[C_verifier.cpp]Calculate Activation(Relu)380\n\n");
	SW_conv5_block2_1_relu(O379_SW,O380_SW);
	printf("[C_verifier.cpp]Calculate Conv2D381\n\n");
	SW_conv5_block2_2_conv(O380_SW,O381_SW,W381);
	printf("[C_verifier.cpp]Calculate Concatenate382\n\n");
	SW_conv5_block2_concat(O375_SW, O381_SW, O382_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization383\n\n");
	SW_conv5_block3_0_bn(O382_SW,O383_SW, W383);
	printf("[C_verifier.cpp]Calculate Activation(Relu)384\n\n");
	SW_conv5_block3_0_relu(O383_SW,O384_SW);
	printf("[C_verifier.cpp]Calculate Conv2D385\n\n");
	SW_conv5_block3_1_conv(O384_SW,O385_SW,W385);
	printf("[C_verifier.cpp]Calculate BatchNormalization386\n\n");
	SW_conv5_block3_1_bn(O385_SW,O386_SW, W386);
	printf("[C_verifier.cpp]Calculate Activation(Relu)387\n\n");
	SW_conv5_block3_1_relu(O386_SW,O387_SW);
	printf("[C_verifier.cpp]Calculate Conv2D388\n\n");
	SW_conv5_block3_2_conv(O387_SW,O388_SW,W388);
	printf("[C_verifier.cpp]Calculate Concatenate389\n\n");
	SW_conv5_block3_concat(O382_SW, O388_SW, O389_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization390\n\n");
	SW_conv5_block4_0_bn(O389_SW,O390_SW, W390);
	printf("[C_verifier.cpp]Calculate Activation(Relu)391\n\n");
	SW_conv5_block4_0_relu(O390_SW,O391_SW);
	printf("[C_verifier.cpp]Calculate Conv2D392\n\n");
	SW_conv5_block4_1_conv(O391_SW,O392_SW,W392);
	printf("[C_verifier.cpp]Calculate BatchNormalization393\n\n");
	SW_conv5_block4_1_bn(O392_SW,O393_SW, W393);
	printf("[C_verifier.cpp]Calculate Activation(Relu)394\n\n");
	SW_conv5_block4_1_relu(O393_SW,O394_SW);
	printf("[C_verifier.cpp]Calculate Conv2D395\n\n");
	SW_conv5_block4_2_conv(O394_SW,O395_SW,W395);
	printf("[C_verifier.cpp]Calculate Concatenate396\n\n");
	SW_conv5_block4_concat(O389_SW, O395_SW, O396_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization397\n\n");
	SW_conv5_block5_0_bn(O396_SW,O397_SW, W397);
	printf("[C_verifier.cpp]Calculate Activation(Relu)398\n\n");
	SW_conv5_block5_0_relu(O397_SW,O398_SW);
	printf("[C_verifier.cpp]Calculate Conv2D399\n\n");
	SW_conv5_block5_1_conv(O398_SW,O399_SW,W399);
	printf("[C_verifier.cpp]Calculate BatchNormalization400\n\n");
	SW_conv5_block5_1_bn(O399_SW,O400_SW, W400);
	printf("[C_verifier.cpp]Calculate Activation(Relu)401\n\n");
	SW_conv5_block5_1_relu(O400_SW,O401_SW);
	printf("[C_verifier.cpp]Calculate Conv2D402\n\n");
	SW_conv5_block5_2_conv(O401_SW,O402_SW,W402);
	printf("[C_verifier.cpp]Calculate Concatenate403\n\n");
	SW_conv5_block5_concat(O396_SW, O402_SW, O403_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization404\n\n");
	SW_conv5_block6_0_bn(O403_SW,O404_SW, W404);
	printf("[C_verifier.cpp]Calculate Activation(Relu)405\n\n");
	SW_conv5_block6_0_relu(O404_SW,O405_SW);
	printf("[C_verifier.cpp]Calculate Conv2D406\n\n");
	SW_conv5_block6_1_conv(O405_SW,O406_SW,W406);
	printf("[C_verifier.cpp]Calculate BatchNormalization407\n\n");
	SW_conv5_block6_1_bn(O406_SW,O407_SW, W407);
	printf("[C_verifier.cpp]Calculate Activation(Relu)408\n\n");
	SW_conv5_block6_1_relu(O407_SW,O408_SW);
	printf("[C_verifier.cpp]Calculate Conv2D409\n\n");
	SW_conv5_block6_2_conv(O408_SW,O409_SW,W409);
	printf("[C_verifier.cpp]Calculate Concatenate410\n\n");
	SW_conv5_block6_concat(O403_SW, O409_SW, O410_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization411\n\n");
	SW_conv5_block7_0_bn(O410_SW,O411_SW, W411);
	printf("[C_verifier.cpp]Calculate Activation(Relu)412\n\n");
	SW_conv5_block7_0_relu(O411_SW,O412_SW);
	printf("[C_verifier.cpp]Calculate Conv2D413\n\n");
	SW_conv5_block7_1_conv(O412_SW,O413_SW,W413);
	printf("[C_verifier.cpp]Calculate BatchNormalization414\n\n");
	SW_conv5_block7_1_bn(O413_SW,O414_SW, W414);
	printf("[C_verifier.cpp]Calculate Activation(Relu)415\n\n");
	SW_conv5_block7_1_relu(O414_SW,O415_SW);
	printf("[C_verifier.cpp]Calculate Conv2D416\n\n");
	SW_conv5_block7_2_conv(O415_SW,O416_SW,W416);
	printf("[C_verifier.cpp]Calculate Concatenate417\n\n");
	SW_conv5_block7_concat(O410_SW, O416_SW, O417_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization418\n\n");
	SW_conv5_block8_0_bn(O417_SW,O418_SW, W418);
	printf("[C_verifier.cpp]Calculate Activation(Relu)419\n\n");
	SW_conv5_block8_0_relu(O418_SW,O419_SW);
	printf("[C_verifier.cpp]Calculate Conv2D420\n\n");
	SW_conv5_block8_1_conv(O419_SW,O420_SW,W420);
	printf("[C_verifier.cpp]Calculate BatchNormalization421\n\n");
	SW_conv5_block8_1_bn(O420_SW,O421_SW, W421);
	printf("[C_verifier.cpp]Calculate Activation(Relu)422\n\n");
	SW_conv5_block8_1_relu(O421_SW,O422_SW);
	printf("[C_verifier.cpp]Calculate Conv2D423\n\n");
	SW_conv5_block8_2_conv(O422_SW,O423_SW,W423);
	printf("[C_verifier.cpp]Calculate Concatenate424\n\n");
	SW_conv5_block8_concat(O417_SW, O423_SW, O424_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization425\n\n");
	SW_conv5_block9_0_bn(O424_SW,O425_SW, W425);
	printf("[C_verifier.cpp]Calculate Activation(Relu)426\n\n");
	SW_conv5_block9_0_relu(O425_SW,O426_SW);
	printf("[C_verifier.cpp]Calculate Conv2D427\n\n");
	SW_conv5_block9_1_conv(O426_SW,O427_SW,W427);
	printf("[C_verifier.cpp]Calculate BatchNormalization428\n\n");
	SW_conv5_block9_1_bn(O427_SW,O428_SW, W428);
	printf("[C_verifier.cpp]Calculate Activation(Relu)429\n\n");
	SW_conv5_block9_1_relu(O428_SW,O429_SW);
	printf("[C_verifier.cpp]Calculate Conv2D430\n\n");
	SW_conv5_block9_2_conv(O429_SW,O430_SW,W430);
	printf("[C_verifier.cpp]Calculate Concatenate431\n\n");
	SW_conv5_block9_concat(O424_SW, O430_SW, O431_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization432\n\n");
	SW_conv5_block10_0_bn(O431_SW,O432_SW, W432);
	printf("[C_verifier.cpp]Calculate Activation(Relu)433\n\n");
	SW_conv5_block10_0_relu(O432_SW,O433_SW);
	printf("[C_verifier.cpp]Calculate Conv2D434\n\n");
	SW_conv5_block10_1_conv(O433_SW,O434_SW,W434);
	printf("[C_verifier.cpp]Calculate BatchNormalization435\n\n");
	SW_conv5_block10_1_bn(O434_SW,O435_SW, W435);
	printf("[C_verifier.cpp]Calculate Activation(Relu)436\n\n");
	SW_conv5_block10_1_relu(O435_SW,O436_SW);
	printf("[C_verifier.cpp]Calculate Conv2D437\n\n");
	SW_conv5_block10_2_conv(O436_SW,O437_SW,W437);
	printf("[C_verifier.cpp]Calculate Concatenate438\n\n");
	SW_conv5_block10_concat(O431_SW, O437_SW, O438_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization439\n\n");
	SW_conv5_block11_0_bn(O438_SW,O439_SW, W439);
	printf("[C_verifier.cpp]Calculate Activation(Relu)440\n\n");
	SW_conv5_block11_0_relu(O439_SW,O440_SW);
	printf("[C_verifier.cpp]Calculate Conv2D441\n\n");
	SW_conv5_block11_1_conv(O440_SW,O441_SW,W441);
	printf("[C_verifier.cpp]Calculate BatchNormalization442\n\n");
	SW_conv5_block11_1_bn(O441_SW,O442_SW, W442);
	printf("[C_verifier.cpp]Calculate Activation(Relu)443\n\n");
	SW_conv5_block11_1_relu(O442_SW,O443_SW);
	printf("[C_verifier.cpp]Calculate Conv2D444\n\n");
	SW_conv5_block11_2_conv(O443_SW,O444_SW,W444);
	printf("[C_verifier.cpp]Calculate Concatenate445\n\n");
	SW_conv5_block11_concat(O438_SW, O444_SW, O445_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization446\n\n");
	SW_conv5_block12_0_bn(O445_SW,O446_SW, W446);
	printf("[C_verifier.cpp]Calculate Activation(Relu)447\n\n");
	SW_conv5_block12_0_relu(O446_SW,O447_SW);
	printf("[C_verifier.cpp]Calculate Conv2D448\n\n");
	SW_conv5_block12_1_conv(O447_SW,O448_SW,W448);
	printf("[C_verifier.cpp]Calculate BatchNormalization449\n\n");
	SW_conv5_block12_1_bn(O448_SW,O449_SW, W449);
	printf("[C_verifier.cpp]Calculate Activation(Relu)450\n\n");
	SW_conv5_block12_1_relu(O449_SW,O450_SW);
	printf("[C_verifier.cpp]Calculate Conv2D451\n\n");
	SW_conv5_block12_2_conv(O450_SW,O451_SW,W451);
	printf("[C_verifier.cpp]Calculate Concatenate452\n\n");
	SW_conv5_block12_concat(O445_SW, O451_SW, O452_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization453\n\n");
	SW_conv5_block13_0_bn(O452_SW,O453_SW, W453);
	printf("[C_verifier.cpp]Calculate Activation(Relu)454\n\n");
	SW_conv5_block13_0_relu(O453_SW,O454_SW);
	printf("[C_verifier.cpp]Calculate Conv2D455\n\n");
	SW_conv5_block13_1_conv(O454_SW,O455_SW,W455);
	printf("[C_verifier.cpp]Calculate BatchNormalization456\n\n");
	SW_conv5_block13_1_bn(O455_SW,O456_SW, W456);
	printf("[C_verifier.cpp]Calculate Activation(Relu)457\n\n");
	SW_conv5_block13_1_relu(O456_SW,O457_SW);
	printf("[C_verifier.cpp]Calculate Conv2D458\n\n");
	SW_conv5_block13_2_conv(O457_SW,O458_SW,W458);
	printf("[C_verifier.cpp]Calculate Concatenate459\n\n");
	SW_conv5_block13_concat(O452_SW, O458_SW, O459_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization460\n\n");
	SW_conv5_block14_0_bn(O459_SW,O460_SW, W460);
	printf("[C_verifier.cpp]Calculate Activation(Relu)461\n\n");
	SW_conv5_block14_0_relu(O460_SW,O461_SW);
	printf("[C_verifier.cpp]Calculate Conv2D462\n\n");
	SW_conv5_block14_1_conv(O461_SW,O462_SW,W462);
	printf("[C_verifier.cpp]Calculate BatchNormalization463\n\n");
	SW_conv5_block14_1_bn(O462_SW,O463_SW, W463);
	printf("[C_verifier.cpp]Calculate Activation(Relu)464\n\n");
	SW_conv5_block14_1_relu(O463_SW,O464_SW);
	printf("[C_verifier.cpp]Calculate Conv2D465\n\n");
	SW_conv5_block14_2_conv(O464_SW,O465_SW,W465);
	printf("[C_verifier.cpp]Calculate Concatenate466\n\n");
	SW_conv5_block14_concat(O459_SW, O465_SW, O466_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization467\n\n");
	SW_conv5_block15_0_bn(O466_SW,O467_SW, W467);
	printf("[C_verifier.cpp]Calculate Activation(Relu)468\n\n");
	SW_conv5_block15_0_relu(O467_SW,O468_SW);
	printf("[C_verifier.cpp]Calculate Conv2D469\n\n");
	SW_conv5_block15_1_conv(O468_SW,O469_SW,W469);
	printf("[C_verifier.cpp]Calculate BatchNormalization470\n\n");
	SW_conv5_block15_1_bn(O469_SW,O470_SW, W470);
	printf("[C_verifier.cpp]Calculate Activation(Relu)471\n\n");
	SW_conv5_block15_1_relu(O470_SW,O471_SW);
	printf("[C_verifier.cpp]Calculate Conv2D472\n\n");
	SW_conv5_block15_2_conv(O471_SW,O472_SW,W472);
	printf("[C_verifier.cpp]Calculate Concatenate473\n\n");
	SW_conv5_block15_concat(O466_SW, O472_SW, O473_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization474\n\n");
	SW_conv5_block16_0_bn(O473_SW,O474_SW, W474);
	printf("[C_verifier.cpp]Calculate Activation(Relu)475\n\n");
	SW_conv5_block16_0_relu(O474_SW,O475_SW);
	printf("[C_verifier.cpp]Calculate Conv2D476\n\n");
	SW_conv5_block16_1_conv(O475_SW,O476_SW,W476);
	printf("[C_verifier.cpp]Calculate BatchNormalization477\n\n");
	SW_conv5_block16_1_bn(O476_SW,O477_SW, W477);
	printf("[C_verifier.cpp]Calculate Activation(Relu)478\n\n");
	SW_conv5_block16_1_relu(O477_SW,O478_SW);
	printf("[C_verifier.cpp]Calculate Conv2D479\n\n");
	SW_conv5_block16_2_conv(O478_SW,O479_SW,W479);
	printf("[C_verifier.cpp]Calculate Concatenate480\n\n");
	SW_conv5_block16_concat(O473_SW, O479_SW, O480_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization481\n\n");
	SW_conv5_block17_0_bn(O480_SW,O481_SW, W481);
	printf("[C_verifier.cpp]Calculate Activation(Relu)482\n\n");
	SW_conv5_block17_0_relu(O481_SW,O482_SW);
	printf("[C_verifier.cpp]Calculate Conv2D483\n\n");
	SW_conv5_block17_1_conv(O482_SW,O483_SW,W483);
	printf("[C_verifier.cpp]Calculate BatchNormalization484\n\n");
	SW_conv5_block17_1_bn(O483_SW,O484_SW, W484);
	printf("[C_verifier.cpp]Calculate Activation(Relu)485\n\n");
	SW_conv5_block17_1_relu(O484_SW,O485_SW);
	printf("[C_verifier.cpp]Calculate Conv2D486\n\n");
	SW_conv5_block17_2_conv(O485_SW,O486_SW,W486);
	printf("[C_verifier.cpp]Calculate Concatenate487\n\n");
	SW_conv5_block17_concat(O480_SW, O486_SW, O487_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization488\n\n");
	SW_conv5_block18_0_bn(O487_SW,O488_SW, W488);
	printf("[C_verifier.cpp]Calculate Activation(Relu)489\n\n");
	SW_conv5_block18_0_relu(O488_SW,O489_SW);
	printf("[C_verifier.cpp]Calculate Conv2D490\n\n");
	SW_conv5_block18_1_conv(O489_SW,O490_SW,W490);
	printf("[C_verifier.cpp]Calculate BatchNormalization491\n\n");
	SW_conv5_block18_1_bn(O490_SW,O491_SW, W491);
	printf("[C_verifier.cpp]Calculate Activation(Relu)492\n\n");
	SW_conv5_block18_1_relu(O491_SW,O492_SW);
	printf("[C_verifier.cpp]Calculate Conv2D493\n\n");
	SW_conv5_block18_2_conv(O492_SW,O493_SW,W493);
	printf("[C_verifier.cpp]Calculate Concatenate494\n\n");
	SW_conv5_block18_concat(O487_SW, O493_SW, O494_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization495\n\n");
	SW_conv5_block19_0_bn(O494_SW,O495_SW, W495);
	printf("[C_verifier.cpp]Calculate Activation(Relu)496\n\n");
	SW_conv5_block19_0_relu(O495_SW,O496_SW);
	printf("[C_verifier.cpp]Calculate Conv2D497\n\n");
	SW_conv5_block19_1_conv(O496_SW,O497_SW,W497);
	printf("[C_verifier.cpp]Calculate BatchNormalization498\n\n");
	SW_conv5_block19_1_bn(O497_SW,O498_SW, W498);
	printf("[C_verifier.cpp]Calculate Activation(Relu)499\n\n");
	SW_conv5_block19_1_relu(O498_SW,O499_SW);
	printf("[C_verifier.cpp]Calculate Conv2D500\n\n");
	SW_conv5_block19_2_conv(O499_SW,O500_SW,W500);
	printf("[C_verifier.cpp]Calculate Concatenate501\n\n");
	SW_conv5_block19_concat(O494_SW, O500_SW, O501_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization502\n\n");
	SW_conv5_block20_0_bn(O501_SW,O502_SW, W502);
	printf("[C_verifier.cpp]Calculate Activation(Relu)503\n\n");
	SW_conv5_block20_0_relu(O502_SW,O503_SW);
	printf("[C_verifier.cpp]Calculate Conv2D504\n\n");
	SW_conv5_block20_1_conv(O503_SW,O504_SW,W504);
	printf("[C_verifier.cpp]Calculate BatchNormalization505\n\n");
	SW_conv5_block20_1_bn(O504_SW,O505_SW, W505);
	printf("[C_verifier.cpp]Calculate Activation(Relu)506\n\n");
	SW_conv5_block20_1_relu(O505_SW,O506_SW);
	printf("[C_verifier.cpp]Calculate Conv2D507\n\n");
	SW_conv5_block20_2_conv(O506_SW,O507_SW,W507);
	printf("[C_verifier.cpp]Calculate Concatenate508\n\n");
	SW_conv5_block20_concat(O501_SW, O507_SW, O508_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization509\n\n");
	SW_conv5_block21_0_bn(O508_SW,O509_SW, W509);
	printf("[C_verifier.cpp]Calculate Activation(Relu)510\n\n");
	SW_conv5_block21_0_relu(O509_SW,O510_SW);
	printf("[C_verifier.cpp]Calculate Conv2D511\n\n");
	SW_conv5_block21_1_conv(O510_SW,O511_SW,W511);
	printf("[C_verifier.cpp]Calculate BatchNormalization512\n\n");
	SW_conv5_block21_1_bn(O511_SW,O512_SW, W512);
	printf("[C_verifier.cpp]Calculate Activation(Relu)513\n\n");
	SW_conv5_block21_1_relu(O512_SW,O513_SW);
	printf("[C_verifier.cpp]Calculate Conv2D514\n\n");
	SW_conv5_block21_2_conv(O513_SW,O514_SW,W514);
	printf("[C_verifier.cpp]Calculate Concatenate515\n\n");
	SW_conv5_block21_concat(O508_SW, O514_SW, O515_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization516\n\n");
	SW_conv5_block22_0_bn(O515_SW,O516_SW, W516);
	printf("[C_verifier.cpp]Calculate Activation(Relu)517\n\n");
	SW_conv5_block22_0_relu(O516_SW,O517_SW);
	printf("[C_verifier.cpp]Calculate Conv2D518\n\n");
	SW_conv5_block22_1_conv(O517_SW,O518_SW,W518);
	printf("[C_verifier.cpp]Calculate BatchNormalization519\n\n");
	SW_conv5_block22_1_bn(O518_SW,O519_SW, W519);
	printf("[C_verifier.cpp]Calculate Activation(Relu)520\n\n");
	SW_conv5_block22_1_relu(O519_SW,O520_SW);
	printf("[C_verifier.cpp]Calculate Conv2D521\n\n");
	SW_conv5_block22_2_conv(O520_SW,O521_SW,W521);
	printf("[C_verifier.cpp]Calculate Concatenate522\n\n");
	SW_conv5_block22_concat(O515_SW, O521_SW, O522_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization523\n\n");
	SW_conv5_block23_0_bn(O522_SW,O523_SW, W523);
	printf("[C_verifier.cpp]Calculate Activation(Relu)524\n\n");
	SW_conv5_block23_0_relu(O523_SW,O524_SW);
	printf("[C_verifier.cpp]Calculate Conv2D525\n\n");
	SW_conv5_block23_1_conv(O524_SW,O525_SW,W525);
	printf("[C_verifier.cpp]Calculate BatchNormalization526\n\n");
	SW_conv5_block23_1_bn(O525_SW,O526_SW, W526);
	printf("[C_verifier.cpp]Calculate Activation(Relu)527\n\n");
	SW_conv5_block23_1_relu(O526_SW,O527_SW);
	printf("[C_verifier.cpp]Calculate Conv2D528\n\n");
	SW_conv5_block23_2_conv(O527_SW,O528_SW,W528);
	printf("[C_verifier.cpp]Calculate Concatenate529\n\n");
	SW_conv5_block23_concat(O522_SW, O528_SW, O529_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization530\n\n");
	SW_conv5_block24_0_bn(O529_SW,O530_SW, W530);
	printf("[C_verifier.cpp]Calculate Activation(Relu)531\n\n");
	SW_conv5_block24_0_relu(O530_SW,O531_SW);
	printf("[C_verifier.cpp]Calculate Conv2D532\n\n");
	SW_conv5_block24_1_conv(O531_SW,O532_SW,W532);
	printf("[C_verifier.cpp]Calculate BatchNormalization533\n\n");
	SW_conv5_block24_1_bn(O532_SW,O533_SW, W533);
	printf("[C_verifier.cpp]Calculate Activation(Relu)534\n\n");
	SW_conv5_block24_1_relu(O533_SW,O534_SW);
	printf("[C_verifier.cpp]Calculate Conv2D535\n\n");
	SW_conv5_block24_2_conv(O534_SW,O535_SW,W535);
	printf("[C_verifier.cpp]Calculate Concatenate536\n\n");
	SW_conv5_block24_concat(O529_SW, O535_SW, O536_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization537\n\n");
	SW_conv5_block25_0_bn(O536_SW,O537_SW, W537);
	printf("[C_verifier.cpp]Calculate Activation(Relu)538\n\n");
	SW_conv5_block25_0_relu(O537_SW,O538_SW);
	printf("[C_verifier.cpp]Calculate Conv2D539\n\n");
	SW_conv5_block25_1_conv(O538_SW,O539_SW,W539);
	printf("[C_verifier.cpp]Calculate BatchNormalization540\n\n");
	SW_conv5_block25_1_bn(O539_SW,O540_SW, W540);
	printf("[C_verifier.cpp]Calculate Activation(Relu)541\n\n");
	SW_conv5_block25_1_relu(O540_SW,O541_SW);
	printf("[C_verifier.cpp]Calculate Conv2D542\n\n");
	SW_conv5_block25_2_conv(O541_SW,O542_SW,W542);
	printf("[C_verifier.cpp]Calculate Concatenate543\n\n");
	SW_conv5_block25_concat(O536_SW, O542_SW, O543_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization544\n\n");
	SW_conv5_block26_0_bn(O543_SW,O544_SW, W544);
	printf("[C_verifier.cpp]Calculate Activation(Relu)545\n\n");
	SW_conv5_block26_0_relu(O544_SW,O545_SW);
	printf("[C_verifier.cpp]Calculate Conv2D546\n\n");
	SW_conv5_block26_1_conv(O545_SW,O546_SW,W546);
	printf("[C_verifier.cpp]Calculate BatchNormalization547\n\n");
	SW_conv5_block26_1_bn(O546_SW,O547_SW, W547);
	printf("[C_verifier.cpp]Calculate Activation(Relu)548\n\n");
	SW_conv5_block26_1_relu(O547_SW,O548_SW);
	printf("[C_verifier.cpp]Calculate Conv2D549\n\n");
	SW_conv5_block26_2_conv(O548_SW,O549_SW,W549);
	printf("[C_verifier.cpp]Calculate Concatenate550\n\n");
	SW_conv5_block26_concat(O543_SW, O549_SW, O550_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization551\n\n");
	SW_conv5_block27_0_bn(O550_SW,O551_SW, W551);
	printf("[C_verifier.cpp]Calculate Activation(Relu)552\n\n");
	SW_conv5_block27_0_relu(O551_SW,O552_SW);
	printf("[C_verifier.cpp]Calculate Conv2D553\n\n");
	SW_conv5_block27_1_conv(O552_SW,O553_SW,W553);
	printf("[C_verifier.cpp]Calculate BatchNormalization554\n\n");
	SW_conv5_block27_1_bn(O553_SW,O554_SW, W554);
	printf("[C_verifier.cpp]Calculate Activation(Relu)555\n\n");
	SW_conv5_block27_1_relu(O554_SW,O555_SW);
	printf("[C_verifier.cpp]Calculate Conv2D556\n\n");
	SW_conv5_block27_2_conv(O555_SW,O556_SW,W556);
	printf("[C_verifier.cpp]Calculate Concatenate557\n\n");
	SW_conv5_block27_concat(O550_SW, O556_SW, O557_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization558\n\n");
	SW_conv5_block28_0_bn(O557_SW,O558_SW, W558);
	printf("[C_verifier.cpp]Calculate Activation(Relu)559\n\n");
	SW_conv5_block28_0_relu(O558_SW,O559_SW);
	printf("[C_verifier.cpp]Calculate Conv2D560\n\n");
	SW_conv5_block28_1_conv(O559_SW,O560_SW,W560);
	printf("[C_verifier.cpp]Calculate BatchNormalization561\n\n");
	SW_conv5_block28_1_bn(O560_SW,O561_SW, W561);
	printf("[C_verifier.cpp]Calculate Activation(Relu)562\n\n");
	SW_conv5_block28_1_relu(O561_SW,O562_SW);
	printf("[C_verifier.cpp]Calculate Conv2D563\n\n");
	SW_conv5_block28_2_conv(O562_SW,O563_SW,W563);
	printf("[C_verifier.cpp]Calculate Concatenate564\n\n");
	SW_conv5_block28_concat(O557_SW, O563_SW, O564_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization565\n\n");
	SW_conv5_block29_0_bn(O564_SW,O565_SW, W565);
	printf("[C_verifier.cpp]Calculate Activation(Relu)566\n\n");
	SW_conv5_block29_0_relu(O565_SW,O566_SW);
	printf("[C_verifier.cpp]Calculate Conv2D567\n\n");
	SW_conv5_block29_1_conv(O566_SW,O567_SW,W567);
	printf("[C_verifier.cpp]Calculate BatchNormalization568\n\n");
	SW_conv5_block29_1_bn(O567_SW,O568_SW, W568);
	printf("[C_verifier.cpp]Calculate Activation(Relu)569\n\n");
	SW_conv5_block29_1_relu(O568_SW,O569_SW);
	printf("[C_verifier.cpp]Calculate Conv2D570\n\n");
	SW_conv5_block29_2_conv(O569_SW,O570_SW,W570);
	printf("[C_verifier.cpp]Calculate Concatenate571\n\n");
	SW_conv5_block29_concat(O564_SW, O570_SW, O571_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization572\n\n");
	SW_conv5_block30_0_bn(O571_SW,O572_SW, W572);
	printf("[C_verifier.cpp]Calculate Activation(Relu)573\n\n");
	SW_conv5_block30_0_relu(O572_SW,O573_SW);
	printf("[C_verifier.cpp]Calculate Conv2D574\n\n");
	SW_conv5_block30_1_conv(O573_SW,O574_SW,W574);
	printf("[C_verifier.cpp]Calculate BatchNormalization575\n\n");
	SW_conv5_block30_1_bn(O574_SW,O575_SW, W575);
	printf("[C_verifier.cpp]Calculate Activation(Relu)576\n\n");
	SW_conv5_block30_1_relu(O575_SW,O576_SW);
	printf("[C_verifier.cpp]Calculate Conv2D577\n\n");
	SW_conv5_block30_2_conv(O576_SW,O577_SW,W577);
	printf("[C_verifier.cpp]Calculate Concatenate578\n\n");
	SW_conv5_block30_concat(O571_SW, O577_SW, O578_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization579\n\n");
	SW_conv5_block31_0_bn(O578_SW,O579_SW, W579);
	printf("[C_verifier.cpp]Calculate Activation(Relu)580\n\n");
	SW_conv5_block31_0_relu(O579_SW,O580_SW);
	printf("[C_verifier.cpp]Calculate Conv2D581\n\n");
	SW_conv5_block31_1_conv(O580_SW,O581_SW,W581);
	printf("[C_verifier.cpp]Calculate BatchNormalization582\n\n");
	SW_conv5_block31_1_bn(O581_SW,O582_SW, W582);
	printf("[C_verifier.cpp]Calculate Activation(Relu)583\n\n");
	SW_conv5_block31_1_relu(O582_SW,O583_SW);
	printf("[C_verifier.cpp]Calculate Conv2D584\n\n");
	SW_conv5_block31_2_conv(O583_SW,O584_SW,W584);
	printf("[C_verifier.cpp]Calculate Concatenate585\n\n");
	SW_conv5_block31_concat(O578_SW, O584_SW, O585_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization586\n\n");
	SW_conv5_block32_0_bn(O585_SW,O586_SW, W586);
	printf("[C_verifier.cpp]Calculate Activation(Relu)587\n\n");
	SW_conv5_block32_0_relu(O586_SW,O587_SW);
	printf("[C_verifier.cpp]Calculate Conv2D588\n\n");
	SW_conv5_block32_1_conv(O587_SW,O588_SW,W588);
	printf("[C_verifier.cpp]Calculate BatchNormalization589\n\n");
	SW_conv5_block32_1_bn(O588_SW,O589_SW, W589);
	printf("[C_verifier.cpp]Calculate Activation(Relu)590\n\n");
	SW_conv5_block32_1_relu(O589_SW,O590_SW);
	printf("[C_verifier.cpp]Calculate Conv2D591\n\n");
	SW_conv5_block32_2_conv(O590_SW,O591_SW,W591);
	printf("[C_verifier.cpp]Calculate Concatenate592\n\n");
	SW_conv5_block32_concat(O585_SW, O591_SW, O592_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization593\n\n");
	SW_bn(O592_SW,O593_SW, W593);
	printf("[C_verifier.cpp]Calculate GlobalAveragePooling2D594\n\n");
	SW_avg_pool(O593_SW,O594_SW);
	printf("[C_verifier.cpp]Calculate Dense595\n\n");
	SW_fc1000(O594_SW,O595_SW,W595,B595);
	

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


fprintf(o_stream,"%s","ZeroPadding2D : [[");
for (k = 0; k < 114 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 114 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O5_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O5_SW[y][k][x]);
		}
		if(x != 114 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 114 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","MaxPooling2D : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O6_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O6_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O7_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O7_SW[y][k][x]);
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O8_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O8_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O9_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O9_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O10_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O10_SW[y][k][x]);
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O11_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O11_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
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


fprintf(o_stream,"%s","Concatenate : [[");
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


fprintf(o_stream,"%s","BatchNormalization : [[");
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 96 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
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


fprintf(o_stream,"%s","BatchNormalization : [[");
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


fprintf(o_stream,"%s","Activation : [[");
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
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
		for(y = 0; y < 128 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
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
		for(y = 0; y < 128 ; y++) {
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
		for(y = 0; y < 128 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
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
		for(y = 0; y < 32 ; y++) {
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O27_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O27_SW[y][k][x]);
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
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O28_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O28_SW[y][k][x]);
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O29_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O29_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O30_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O30_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O31_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O31_SW[y][k][x]);
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O32_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O32_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O33_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O33_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O34_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O34_SW[y][k][x]);
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
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O35_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O35_SW[y][k][x]);
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O36_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O36_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O37_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O37_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O38_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O38_SW[y][k][x]);
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O39_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O39_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O40_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O40_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O41_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O41_SW[y][k][x]);
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
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O42_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O42_SW[y][k][x]);
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O43_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O43_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O44_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O44_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O45_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O45_SW[y][k][x]);
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O46_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O46_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O47_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O47_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O48_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O48_SW[y][k][x]);
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
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O49_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O49_SW[y][k][x]);
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O50_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O50_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O51_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O51_SW[y][k][x]);
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


fprintf(o_stream,"%s","AveragePooling2D : [[");
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


fprintf(o_stream,"%s","BatchNormalization : [[");
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


fprintf(o_stream,"%s","Activation : [[");
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


fprintf(o_stream,"%s","Conv2D : [[");
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
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


fprintf(o_stream,"%s","BatchNormalization : [[");
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


fprintf(o_stream,"%s","Activation : [[");
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
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
		for(y = 0; y < 192 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
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


fprintf(o_stream,"%s","Activation : [[");
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 224 ; y++) {
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
		for(y = 0; y < 224 ; y++) {
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
		for(y = 0; y < 224 ; y++) {
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
		for(y = 0; y < 128 ; y++) {
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
		for(y = 0; y < 128 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O80_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O80_SW[y][k][x]);
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
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O81_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O81_SW[y][k][x]);
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
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O82_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O82_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O83_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O83_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O84_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O84_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O85_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O85_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O86_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O86_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 288 ; y++) {
			fprintf(o_stream,"%.6f ",O87_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O87_SW[y][k][x]);
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
		for(y = 0; y < 288 ; y++) {
			fprintf(o_stream,"%.6f ",O88_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O88_SW[y][k][x]);
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
		for(y = 0; y < 288 ; y++) {
			fprintf(o_stream,"%.6f ",O89_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O89_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O90_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O90_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O91_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O91_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O92_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O92_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O93_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O93_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O94_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O94_SW[y][k][x]);
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
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O95_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O95_SW[y][k][x]);
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
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O96_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O96_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O97_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O97_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O98_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O98_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O99_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O99_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O100_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O100_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 352 ; y++) {
			fprintf(o_stream,"%.6f ",O101_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O101_SW[y][k][x]);
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
		for(y = 0; y < 352 ; y++) {
			fprintf(o_stream,"%.6f ",O102_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O102_SW[y][k][x]);
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
		for(y = 0; y < 352 ; y++) {
			fprintf(o_stream,"%.6f ",O103_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O103_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O104_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O104_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O105_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O105_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O106_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O106_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O107_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O107_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O108_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O108_SW[y][k][x]);
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
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O109_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O109_SW[y][k][x]);
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
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O110_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O110_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O111_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O111_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O112_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O112_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O113_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O113_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O114_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O114_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 416 ; y++) {
			fprintf(o_stream,"%.6f ",O115_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O115_SW[y][k][x]);
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
		for(y = 0; y < 416 ; y++) {
			fprintf(o_stream,"%.6f ",O116_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O116_SW[y][k][x]);
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
		for(y = 0; y < 416 ; y++) {
			fprintf(o_stream,"%.6f ",O117_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O117_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O118_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O118_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O119_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O119_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O120_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O120_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O121_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O121_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 448 ; y++) {
			fprintf(o_stream,"%.6f ",O122_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O122_SW[y][k][x]);
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
		for(y = 0; y < 448 ; y++) {
			fprintf(o_stream,"%.6f ",O123_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O123_SW[y][k][x]);
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
		for(y = 0; y < 448 ; y++) {
			fprintf(o_stream,"%.6f ",O124_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O124_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O125_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O125_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O126_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O126_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O127_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O127_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O128_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O128_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 480 ; y++) {
			fprintf(o_stream,"%.6f ",O129_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O129_SW[y][k][x]);
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
		for(y = 0; y < 480 ; y++) {
			fprintf(o_stream,"%.6f ",O130_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O130_SW[y][k][x]);
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
		for(y = 0; y < 480 ; y++) {
			fprintf(o_stream,"%.6f ",O131_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O131_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O132_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O132_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O133_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O133_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O134_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O134_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O135_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O135_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O136_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O136_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O137_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O137_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O138_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O138_SW[y][k][x]);
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
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O139_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O139_SW[y][k][x]);
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


fprintf(o_stream,"%s","AveragePooling2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O142_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O142_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O143_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O143_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O144_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O144_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O145_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O145_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O146_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O146_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 288 ; y++) {
			fprintf(o_stream,"%.6f ",O147_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O147_SW[y][k][x]);
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
		for(y = 0; y < 288 ; y++) {
			fprintf(o_stream,"%.6f ",O148_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O148_SW[y][k][x]);
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
		for(y = 0; y < 288 ; y++) {
			fprintf(o_stream,"%.6f ",O149_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O149_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O150_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O150_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O151_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O151_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O152_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O152_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O153_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O153_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O154_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O154_SW[y][k][x]);
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
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O155_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O155_SW[y][k][x]);
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
		for(y = 0; y < 320 ; y++) {
			fprintf(o_stream,"%.6f ",O156_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O156_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O157_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O157_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O158_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O158_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O159_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O159_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O160_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O160_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 352 ; y++) {
			fprintf(o_stream,"%.6f ",O161_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O161_SW[y][k][x]);
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
		for(y = 0; y < 352 ; y++) {
			fprintf(o_stream,"%.6f ",O162_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O162_SW[y][k][x]);
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
		for(y = 0; y < 352 ; y++) {
			fprintf(o_stream,"%.6f ",O163_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O163_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O164_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O164_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O165_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O165_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O166_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O166_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O167_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O167_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O168_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O168_SW[y][k][x]);
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
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O169_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O169_SW[y][k][x]);
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
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O170_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O170_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O171_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O171_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O172_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O172_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O173_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O173_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O174_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O174_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 416 ; y++) {
			fprintf(o_stream,"%.6f ",O175_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O175_SW[y][k][x]);
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
		for(y = 0; y < 416 ; y++) {
			fprintf(o_stream,"%.6f ",O176_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O176_SW[y][k][x]);
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
		for(y = 0; y < 416 ; y++) {
			fprintf(o_stream,"%.6f ",O177_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O177_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O178_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O178_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O179_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O179_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O180_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O180_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O181_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O181_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 448 ; y++) {
			fprintf(o_stream,"%.6f ",O182_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O182_SW[y][k][x]);
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
		for(y = 0; y < 448 ; y++) {
			fprintf(o_stream,"%.6f ",O183_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O183_SW[y][k][x]);
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
		for(y = 0; y < 448 ; y++) {
			fprintf(o_stream,"%.6f ",O184_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O184_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O185_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O185_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O186_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O186_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O187_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O187_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O188_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O188_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 480 ; y++) {
			fprintf(o_stream,"%.6f ",O189_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O189_SW[y][k][x]);
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
		for(y = 0; y < 480 ; y++) {
			fprintf(o_stream,"%.6f ",O190_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O190_SW[y][k][x]);
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
		for(y = 0; y < 480 ; y++) {
			fprintf(o_stream,"%.6f ",O191_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O191_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O192_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O192_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O193_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O193_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O194_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O194_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O195_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O195_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O196_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O196_SW[y][k][x]);
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
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O197_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O197_SW[y][k][x]);
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
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O198_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O198_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O199_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O199_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O200_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O200_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O201_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O201_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O202_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O202_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 544 ; y++) {
			fprintf(o_stream,"%.6f ",O203_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O203_SW[y][k][x]);
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
		for(y = 0; y < 544 ; y++) {
			fprintf(o_stream,"%.6f ",O204_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O204_SW[y][k][x]);
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
		for(y = 0; y < 544 ; y++) {
			fprintf(o_stream,"%.6f ",O205_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O205_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O206_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O206_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O207_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O207_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O208_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O208_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O209_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O209_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 576 ; y++) {
			fprintf(o_stream,"%.6f ",O210_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O210_SW[y][k][x]);
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
		for(y = 0; y < 576 ; y++) {
			fprintf(o_stream,"%.6f ",O211_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O211_SW[y][k][x]);
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
		for(y = 0; y < 576 ; y++) {
			fprintf(o_stream,"%.6f ",O212_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O212_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O213_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O213_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O214_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O214_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O215_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O215_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O216_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O216_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 608 ; y++) {
			fprintf(o_stream,"%.6f ",O217_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O217_SW[y][k][x]);
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
		for(y = 0; y < 608 ; y++) {
			fprintf(o_stream,"%.6f ",O218_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O218_SW[y][k][x]);
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
		for(y = 0; y < 608 ; y++) {
			fprintf(o_stream,"%.6f ",O219_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O219_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O220_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O220_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O221_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O221_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O222_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O222_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O223_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O223_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 640 ; y++) {
			fprintf(o_stream,"%.6f ",O224_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O224_SW[y][k][x]);
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
		for(y = 0; y < 640 ; y++) {
			fprintf(o_stream,"%.6f ",O225_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O225_SW[y][k][x]);
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
		for(y = 0; y < 640 ; y++) {
			fprintf(o_stream,"%.6f ",O226_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O226_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O227_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O227_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O228_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O228_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O229_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O229_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O230_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O230_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 672 ; y++) {
			fprintf(o_stream,"%.6f ",O231_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O231_SW[y][k][x]);
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
		for(y = 0; y < 672 ; y++) {
			fprintf(o_stream,"%.6f ",O232_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O232_SW[y][k][x]);
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
		for(y = 0; y < 672 ; y++) {
			fprintf(o_stream,"%.6f ",O233_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O233_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O234_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O234_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O235_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O235_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O236_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O236_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O237_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O237_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 704 ; y++) {
			fprintf(o_stream,"%.6f ",O238_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O238_SW[y][k][x]);
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
		for(y = 0; y < 704 ; y++) {
			fprintf(o_stream,"%.6f ",O239_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O239_SW[y][k][x]);
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
		for(y = 0; y < 704 ; y++) {
			fprintf(o_stream,"%.6f ",O240_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O240_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O241_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O241_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O242_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O242_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O243_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O243_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O244_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O244_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 736 ; y++) {
			fprintf(o_stream,"%.6f ",O245_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O245_SW[y][k][x]);
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
		for(y = 0; y < 736 ; y++) {
			fprintf(o_stream,"%.6f ",O246_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O246_SW[y][k][x]);
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
		for(y = 0; y < 736 ; y++) {
			fprintf(o_stream,"%.6f ",O247_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O247_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O248_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O248_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O249_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O249_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O250_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O250_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O251_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O251_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 768 ; y++) {
			fprintf(o_stream,"%.6f ",O252_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O252_SW[y][k][x]);
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
		for(y = 0; y < 768 ; y++) {
			fprintf(o_stream,"%.6f ",O253_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O253_SW[y][k][x]);
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
		for(y = 0; y < 768 ; y++) {
			fprintf(o_stream,"%.6f ",O254_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O254_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O255_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O255_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O256_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O256_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O257_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O257_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O258_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O258_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 800 ; y++) {
			fprintf(o_stream,"%.6f ",O259_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O259_SW[y][k][x]);
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
		for(y = 0; y < 800 ; y++) {
			fprintf(o_stream,"%.6f ",O260_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O260_SW[y][k][x]);
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
		for(y = 0; y < 800 ; y++) {
			fprintf(o_stream,"%.6f ",O261_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O261_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O262_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O262_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O263_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O263_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O264_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O264_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O265_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O265_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 832 ; y++) {
			fprintf(o_stream,"%.6f ",O266_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O266_SW[y][k][x]);
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
		for(y = 0; y < 832 ; y++) {
			fprintf(o_stream,"%.6f ",O267_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O267_SW[y][k][x]);
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
		for(y = 0; y < 832 ; y++) {
			fprintf(o_stream,"%.6f ",O268_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O268_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O269_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O269_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O270_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O270_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O271_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O271_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O272_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O272_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 864 ; y++) {
			fprintf(o_stream,"%.6f ",O273_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O273_SW[y][k][x]);
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
		for(y = 0; y < 864 ; y++) {
			fprintf(o_stream,"%.6f ",O274_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O274_SW[y][k][x]);
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
		for(y = 0; y < 864 ; y++) {
			fprintf(o_stream,"%.6f ",O275_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O275_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O276_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O276_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O277_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O277_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O278_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O278_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O279_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O279_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 896 ; y++) {
			fprintf(o_stream,"%.6f ",O280_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O280_SW[y][k][x]);
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
		for(y = 0; y < 896 ; y++) {
			fprintf(o_stream,"%.6f ",O281_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O281_SW[y][k][x]);
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
		for(y = 0; y < 896 ; y++) {
			fprintf(o_stream,"%.6f ",O282_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O282_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O283_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O283_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O284_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O284_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O285_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O285_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O286_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O286_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 928 ; y++) {
			fprintf(o_stream,"%.6f ",O287_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O287_SW[y][k][x]);
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
		for(y = 0; y < 928 ; y++) {
			fprintf(o_stream,"%.6f ",O288_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O288_SW[y][k][x]);
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
		for(y = 0; y < 928 ; y++) {
			fprintf(o_stream,"%.6f ",O289_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O289_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O290_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O290_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O291_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O291_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O292_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O292_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O293_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O293_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 960 ; y++) {
			fprintf(o_stream,"%.6f ",O294_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O294_SW[y][k][x]);
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
		for(y = 0; y < 960 ; y++) {
			fprintf(o_stream,"%.6f ",O295_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O295_SW[y][k][x]);
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
		for(y = 0; y < 960 ; y++) {
			fprintf(o_stream,"%.6f ",O296_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O296_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O297_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O297_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O298_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O298_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O299_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O299_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O300_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O300_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 992 ; y++) {
			fprintf(o_stream,"%.6f ",O301_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O301_SW[y][k][x]);
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
		for(y = 0; y < 992 ; y++) {
			fprintf(o_stream,"%.6f ",O302_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O302_SW[y][k][x]);
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
		for(y = 0; y < 992 ; y++) {
			fprintf(o_stream,"%.6f ",O303_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O303_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O304_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O304_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O305_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O305_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O306_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O306_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O307_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O307_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O308_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O308_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O309_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O309_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O310_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O310_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O311_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O311_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O312_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O312_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O313_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O313_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O314_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O314_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1056 ; y++) {
			fprintf(o_stream,"%.6f ",O315_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O315_SW[y][k][x]);
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
		for(y = 0; y < 1056 ; y++) {
			fprintf(o_stream,"%.6f ",O316_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O316_SW[y][k][x]);
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
		for(y = 0; y < 1056 ; y++) {
			fprintf(o_stream,"%.6f ",O317_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O317_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O318_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O318_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O319_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O319_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O320_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O320_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O321_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O321_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O322_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O322_SW[y][k][x]);
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
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O323_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O323_SW[y][k][x]);
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
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O324_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O324_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O325_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O325_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O326_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O326_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O327_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O327_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O328_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O328_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1120 ; y++) {
			fprintf(o_stream,"%.6f ",O329_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O329_SW[y][k][x]);
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
		for(y = 0; y < 1120 ; y++) {
			fprintf(o_stream,"%.6f ",O330_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O330_SW[y][k][x]);
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
		for(y = 0; y < 1120 ; y++) {
			fprintf(o_stream,"%.6f ",O331_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O331_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O332_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O332_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O333_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O333_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O334_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O334_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O335_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O335_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1152 ; y++) {
			fprintf(o_stream,"%.6f ",O336_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O336_SW[y][k][x]);
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
		for(y = 0; y < 1152 ; y++) {
			fprintf(o_stream,"%.6f ",O337_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O337_SW[y][k][x]);
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
		for(y = 0; y < 1152 ; y++) {
			fprintf(o_stream,"%.6f ",O338_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O338_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O339_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O339_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O340_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O340_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O341_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O341_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O342_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O342_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1184 ; y++) {
			fprintf(o_stream,"%.6f ",O343_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O343_SW[y][k][x]);
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
		for(y = 0; y < 1184 ; y++) {
			fprintf(o_stream,"%.6f ",O344_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O344_SW[y][k][x]);
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
		for(y = 0; y < 1184 ; y++) {
			fprintf(o_stream,"%.6f ",O345_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O345_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O346_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O346_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O347_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O347_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O348_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O348_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O349_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O349_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1216 ; y++) {
			fprintf(o_stream,"%.6f ",O350_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O350_SW[y][k][x]);
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
		for(y = 0; y < 1216 ; y++) {
			fprintf(o_stream,"%.6f ",O351_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O351_SW[y][k][x]);
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
		for(y = 0; y < 1216 ; y++) {
			fprintf(o_stream,"%.6f ",O352_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O352_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O353_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O353_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O354_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O354_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O355_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O355_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O356_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O356_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1248 ; y++) {
			fprintf(o_stream,"%.6f ",O357_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O357_SW[y][k][x]);
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
		for(y = 0; y < 1248 ; y++) {
			fprintf(o_stream,"%.6f ",O358_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O358_SW[y][k][x]);
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
		for(y = 0; y < 1248 ; y++) {
			fprintf(o_stream,"%.6f ",O359_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O359_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O360_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O360_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O361_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O361_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O362_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O362_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O363_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O363_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1280 ; y++) {
			fprintf(o_stream,"%.6f ",O364_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O364_SW[y][k][x]);
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
		for(y = 0; y < 1280 ; y++) {
			fprintf(o_stream,"%.6f ",O365_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O365_SW[y][k][x]);
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
		for(y = 0; y < 1280 ; y++) {
			fprintf(o_stream,"%.6f ",O366_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O366_SW[y][k][x]);
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
		for(y = 0; y < 640 ; y++) {
			fprintf(o_stream,"%.6f ",O367_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O367_SW[y][k][x]);
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


fprintf(o_stream,"%s","AveragePooling2D : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 640 ; y++) {
			fprintf(o_stream,"%.6f ",O368_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O368_SW[y][k][x]);
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
		for(y = 0; y < 640 ; y++) {
			fprintf(o_stream,"%.6f ",O369_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O369_SW[y][k][x]);
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
		for(y = 0; y < 640 ; y++) {
			fprintf(o_stream,"%.6f ",O370_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O370_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O371_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O371_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O372_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O372_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O373_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O373_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O374_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O374_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 672 ; y++) {
			fprintf(o_stream,"%.6f ",O375_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O375_SW[y][k][x]);
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
		for(y = 0; y < 672 ; y++) {
			fprintf(o_stream,"%.6f ",O376_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O376_SW[y][k][x]);
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
		for(y = 0; y < 672 ; y++) {
			fprintf(o_stream,"%.6f ",O377_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O377_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O378_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O378_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O379_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O379_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O380_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O380_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O381_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O381_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 704 ; y++) {
			fprintf(o_stream,"%.6f ",O382_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O382_SW[y][k][x]);
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
		for(y = 0; y < 704 ; y++) {
			fprintf(o_stream,"%.6f ",O383_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O383_SW[y][k][x]);
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
		for(y = 0; y < 704 ; y++) {
			fprintf(o_stream,"%.6f ",O384_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O384_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O385_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O385_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O386_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O386_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O387_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O387_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O388_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O388_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 736 ; y++) {
			fprintf(o_stream,"%.6f ",O389_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O389_SW[y][k][x]);
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
		for(y = 0; y < 736 ; y++) {
			fprintf(o_stream,"%.6f ",O390_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O390_SW[y][k][x]);
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
		for(y = 0; y < 736 ; y++) {
			fprintf(o_stream,"%.6f ",O391_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O391_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O392_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O392_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O393_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O393_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O394_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O394_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O395_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O395_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 768 ; y++) {
			fprintf(o_stream,"%.6f ",O396_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O396_SW[y][k][x]);
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
		for(y = 0; y < 768 ; y++) {
			fprintf(o_stream,"%.6f ",O397_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O397_SW[y][k][x]);
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
		for(y = 0; y < 768 ; y++) {
			fprintf(o_stream,"%.6f ",O398_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O398_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O399_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O399_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O400_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O400_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O401_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O401_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O402_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O402_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 800 ; y++) {
			fprintf(o_stream,"%.6f ",O403_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O403_SW[y][k][x]);
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
		for(y = 0; y < 800 ; y++) {
			fprintf(o_stream,"%.6f ",O404_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O404_SW[y][k][x]);
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
		for(y = 0; y < 800 ; y++) {
			fprintf(o_stream,"%.6f ",O405_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O405_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O406_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O406_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O407_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O407_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O408_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O408_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O409_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O409_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 832 ; y++) {
			fprintf(o_stream,"%.6f ",O410_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O410_SW[y][k][x]);
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
		for(y = 0; y < 832 ; y++) {
			fprintf(o_stream,"%.6f ",O411_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O411_SW[y][k][x]);
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
		for(y = 0; y < 832 ; y++) {
			fprintf(o_stream,"%.6f ",O412_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O412_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O413_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O413_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O414_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O414_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O415_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O415_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O416_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O416_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 864 ; y++) {
			fprintf(o_stream,"%.6f ",O417_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O417_SW[y][k][x]);
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
		for(y = 0; y < 864 ; y++) {
			fprintf(o_stream,"%.6f ",O418_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O418_SW[y][k][x]);
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
		for(y = 0; y < 864 ; y++) {
			fprintf(o_stream,"%.6f ",O419_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O419_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O420_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O420_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O421_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O421_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O422_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O422_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O423_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O423_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 896 ; y++) {
			fprintf(o_stream,"%.6f ",O424_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O424_SW[y][k][x]);
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
		for(y = 0; y < 896 ; y++) {
			fprintf(o_stream,"%.6f ",O425_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O425_SW[y][k][x]);
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
		for(y = 0; y < 896 ; y++) {
			fprintf(o_stream,"%.6f ",O426_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O426_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O427_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O427_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O428_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O428_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O429_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O429_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O430_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O430_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 928 ; y++) {
			fprintf(o_stream,"%.6f ",O431_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O431_SW[y][k][x]);
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
		for(y = 0; y < 928 ; y++) {
			fprintf(o_stream,"%.6f ",O432_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O432_SW[y][k][x]);
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
		for(y = 0; y < 928 ; y++) {
			fprintf(o_stream,"%.6f ",O433_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O433_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O434_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O434_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O435_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O435_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O436_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O436_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O437_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O437_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 960 ; y++) {
			fprintf(o_stream,"%.6f ",O438_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O438_SW[y][k][x]);
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
		for(y = 0; y < 960 ; y++) {
			fprintf(o_stream,"%.6f ",O439_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O439_SW[y][k][x]);
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
		for(y = 0; y < 960 ; y++) {
			fprintf(o_stream,"%.6f ",O440_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O440_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O441_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O441_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O442_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O442_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O443_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O443_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O444_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O444_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 992 ; y++) {
			fprintf(o_stream,"%.6f ",O445_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O445_SW[y][k][x]);
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
		for(y = 0; y < 992 ; y++) {
			fprintf(o_stream,"%.6f ",O446_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O446_SW[y][k][x]);
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
		for(y = 0; y < 992 ; y++) {
			fprintf(o_stream,"%.6f ",O447_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O447_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O448_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O448_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O449_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O449_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O450_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O450_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O451_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O451_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O452_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O452_SW[y][k][x]);
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
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O453_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O453_SW[y][k][x]);
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
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O454_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O454_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O455_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O455_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O456_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O456_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O457_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O457_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O458_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O458_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1056 ; y++) {
			fprintf(o_stream,"%.6f ",O459_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O459_SW[y][k][x]);
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
		for(y = 0; y < 1056 ; y++) {
			fprintf(o_stream,"%.6f ",O460_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O460_SW[y][k][x]);
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
		for(y = 0; y < 1056 ; y++) {
			fprintf(o_stream,"%.6f ",O461_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O461_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O462_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O462_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O463_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O463_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O464_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O464_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O465_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O465_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O466_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O466_SW[y][k][x]);
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
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O467_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O467_SW[y][k][x]);
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
		for(y = 0; y < 1088 ; y++) {
			fprintf(o_stream,"%.6f ",O468_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O468_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O469_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O469_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O470_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O470_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O471_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O471_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O472_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O472_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1120 ; y++) {
			fprintf(o_stream,"%.6f ",O473_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O473_SW[y][k][x]);
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
		for(y = 0; y < 1120 ; y++) {
			fprintf(o_stream,"%.6f ",O474_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O474_SW[y][k][x]);
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
		for(y = 0; y < 1120 ; y++) {
			fprintf(o_stream,"%.6f ",O475_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O475_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O476_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O476_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O477_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O477_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O478_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O478_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O479_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O479_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1152 ; y++) {
			fprintf(o_stream,"%.6f ",O480_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O480_SW[y][k][x]);
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
		for(y = 0; y < 1152 ; y++) {
			fprintf(o_stream,"%.6f ",O481_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O481_SW[y][k][x]);
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
		for(y = 0; y < 1152 ; y++) {
			fprintf(o_stream,"%.6f ",O482_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O482_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O483_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O483_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O484_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O484_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O485_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O485_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O486_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O486_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1184 ; y++) {
			fprintf(o_stream,"%.6f ",O487_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O487_SW[y][k][x]);
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
		for(y = 0; y < 1184 ; y++) {
			fprintf(o_stream,"%.6f ",O488_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O488_SW[y][k][x]);
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
		for(y = 0; y < 1184 ; y++) {
			fprintf(o_stream,"%.6f ",O489_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O489_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O490_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O490_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O491_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O491_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O492_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O492_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O493_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O493_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1216 ; y++) {
			fprintf(o_stream,"%.6f ",O494_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O494_SW[y][k][x]);
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
		for(y = 0; y < 1216 ; y++) {
			fprintf(o_stream,"%.6f ",O495_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O495_SW[y][k][x]);
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
		for(y = 0; y < 1216 ; y++) {
			fprintf(o_stream,"%.6f ",O496_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O496_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O497_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O497_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O498_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O498_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O499_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O499_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O500_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O500_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1248 ; y++) {
			fprintf(o_stream,"%.6f ",O501_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O501_SW[y][k][x]);
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
		for(y = 0; y < 1248 ; y++) {
			fprintf(o_stream,"%.6f ",O502_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O502_SW[y][k][x]);
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
		for(y = 0; y < 1248 ; y++) {
			fprintf(o_stream,"%.6f ",O503_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O503_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O504_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O504_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O505_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O505_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O506_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O506_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O507_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O507_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1280 ; y++) {
			fprintf(o_stream,"%.6f ",O508_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O508_SW[y][k][x]);
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
		for(y = 0; y < 1280 ; y++) {
			fprintf(o_stream,"%.6f ",O509_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O509_SW[y][k][x]);
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
		for(y = 0; y < 1280 ; y++) {
			fprintf(o_stream,"%.6f ",O510_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O510_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O511_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O511_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O512_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O512_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O513_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O513_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O514_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O514_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1312 ; y++) {
			fprintf(o_stream,"%.6f ",O515_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O515_SW[y][k][x]);
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
		for(y = 0; y < 1312 ; y++) {
			fprintf(o_stream,"%.6f ",O516_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O516_SW[y][k][x]);
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
		for(y = 0; y < 1312 ; y++) {
			fprintf(o_stream,"%.6f ",O517_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O517_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O518_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O518_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O519_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O519_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O520_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O520_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O521_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O521_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1344 ; y++) {
			fprintf(o_stream,"%.6f ",O522_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O522_SW[y][k][x]);
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
		for(y = 0; y < 1344 ; y++) {
			fprintf(o_stream,"%.6f ",O523_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O523_SW[y][k][x]);
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
		for(y = 0; y < 1344 ; y++) {
			fprintf(o_stream,"%.6f ",O524_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O524_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O525_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O525_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O526_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O526_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O527_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O527_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O528_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O528_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1376 ; y++) {
			fprintf(o_stream,"%.6f ",O529_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O529_SW[y][k][x]);
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
		for(y = 0; y < 1376 ; y++) {
			fprintf(o_stream,"%.6f ",O530_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O530_SW[y][k][x]);
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
		for(y = 0; y < 1376 ; y++) {
			fprintf(o_stream,"%.6f ",O531_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O531_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O532_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O532_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O533_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O533_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O534_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O534_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O535_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O535_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1408 ; y++) {
			fprintf(o_stream,"%.6f ",O536_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O536_SW[y][k][x]);
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
		for(y = 0; y < 1408 ; y++) {
			fprintf(o_stream,"%.6f ",O537_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O537_SW[y][k][x]);
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
		for(y = 0; y < 1408 ; y++) {
			fprintf(o_stream,"%.6f ",O538_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O538_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O539_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O539_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O540_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O540_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O541_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O541_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O542_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O542_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1440 ; y++) {
			fprintf(o_stream,"%.6f ",O543_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O543_SW[y][k][x]);
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
		for(y = 0; y < 1440 ; y++) {
			fprintf(o_stream,"%.6f ",O544_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O544_SW[y][k][x]);
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
		for(y = 0; y < 1440 ; y++) {
			fprintf(o_stream,"%.6f ",O545_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O545_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O546_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O546_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O547_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O547_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O548_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O548_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O549_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O549_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1472 ; y++) {
			fprintf(o_stream,"%.6f ",O550_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O550_SW[y][k][x]);
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
		for(y = 0; y < 1472 ; y++) {
			fprintf(o_stream,"%.6f ",O551_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O551_SW[y][k][x]);
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
		for(y = 0; y < 1472 ; y++) {
			fprintf(o_stream,"%.6f ",O552_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O552_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O553_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O553_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O554_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O554_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O555_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O555_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O556_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O556_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1504 ; y++) {
			fprintf(o_stream,"%.6f ",O557_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O557_SW[y][k][x]);
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
		for(y = 0; y < 1504 ; y++) {
			fprintf(o_stream,"%.6f ",O558_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O558_SW[y][k][x]);
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
		for(y = 0; y < 1504 ; y++) {
			fprintf(o_stream,"%.6f ",O559_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O559_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O560_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O560_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O561_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O561_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O562_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O562_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O563_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O563_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1536 ; y++) {
			fprintf(o_stream,"%.6f ",O564_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O564_SW[y][k][x]);
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
		for(y = 0; y < 1536 ; y++) {
			fprintf(o_stream,"%.6f ",O565_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O565_SW[y][k][x]);
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
		for(y = 0; y < 1536 ; y++) {
			fprintf(o_stream,"%.6f ",O566_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O566_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O567_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O567_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O568_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O568_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O569_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O569_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O570_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O570_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1568 ; y++) {
			fprintf(o_stream,"%.6f ",O571_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O571_SW[y][k][x]);
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
		for(y = 0; y < 1568 ; y++) {
			fprintf(o_stream,"%.6f ",O572_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O572_SW[y][k][x]);
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
		for(y = 0; y < 1568 ; y++) {
			fprintf(o_stream,"%.6f ",O573_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O573_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O574_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O574_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O575_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O575_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O576_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O576_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O577_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O577_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1600 ; y++) {
			fprintf(o_stream,"%.6f ",O578_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O578_SW[y][k][x]);
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
		for(y = 0; y < 1600 ; y++) {
			fprintf(o_stream,"%.6f ",O579_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O579_SW[y][k][x]);
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
		for(y = 0; y < 1600 ; y++) {
			fprintf(o_stream,"%.6f ",O580_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O580_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O581_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O581_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O582_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O582_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O583_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O583_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O584_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O584_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1632 ; y++) {
			fprintf(o_stream,"%.6f ",O585_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O585_SW[y][k][x]);
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
		for(y = 0; y < 1632 ; y++) {
			fprintf(o_stream,"%.6f ",O586_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O586_SW[y][k][x]);
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
		for(y = 0; y < 1632 ; y++) {
			fprintf(o_stream,"%.6f ",O587_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O587_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O588_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O588_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O589_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O589_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O590_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O590_SW[y][k][x]);
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
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",O591_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O591_SW[y][k][x]);
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


fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1664 ; y++) {
			fprintf(o_stream,"%.6f ",O592_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O592_SW[y][k][x]);
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
		for(y = 0; y < 1664 ; y++) {
			fprintf(o_stream,"%.6f ",O593_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O593_SW[y][k][x]);
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


fprintf(o_stream,"%s","GlobalAveragePooling2D : [[");
for (k = 0; k <  1664 ; k++) {
	fprintf(o_stream,"%.6f ",O594_SW[k]);
	fprintf(c_num,"%.6f ",O594_SW[k]);
}
fprintf(o_stream,"%s","]]\n\n");


fprintf(o_stream,"%s","Dense : [[");
for (k = 0; k <  1000 ; k++) {
	fprintf(o_stream,"%.6f ",O595_SW[k]);
	fprintf(c_num,"%.6f ",O595_SW[k]);
}
fprintf(o_stream,"%s","]]\n\n");





    fclose(w_stream);
    fclose(i_stream);
    fclose(o_stream);
    fclose(c_num);

    return 0;
}