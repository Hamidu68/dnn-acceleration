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
void SW_conv_dw_3_relu(DATA_T I[128][56][56], DATA_T O[128][56][56])
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
void SW_conv_pw_3(DATA_T I[128][56][56], DATA_T O[128][56][56], DATA_T W[128][128][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(56 - 1) - 56 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
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

void SW_conv_pw_3_bn(DATA_T I[128][56][56], DATA_T O[128][56][56], DATA_T W[4][128]) {
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
void SW_conv_pw_3_relu(DATA_T I[128][56][56], DATA_T O[128][56][56])
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
void SW_conv_pad_4(DATA_T I[128][56][56],DATA_T O[128][57][57]) {
	int m, x, y, i, j;
	for (m = 0; m < 128; m++) {
		for (x = 0; x < 57; x++) {
			for (y = 0; y < 57; y++) {
				O[m][x][y] = 0;
			}
		}
	}
	for (m = 0; m < 128; m++) {
		for (x = 0; x < 56; x++) {
			for (y = 0; y < 56; y++) {
				O[m][x+ 0][y+ 0] = I[m][x][y];
			}
		}
	}
}
void SW_conv_dw_4(DATA_T I[128][57][57], DATA_T O[128][28][28], DATA_T W[128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {
						if (x + i <= 57 && y + j <= 57) {
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
void SW_conv_dw_4_bn(DATA_T I[128][28][28], DATA_T O[128][28][28], DATA_T W[4][128]) {
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
void SW_conv_dw_4_relu(DATA_T I[128][28][28], DATA_T O[128][28][28])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
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
void SW_conv_pw_4(DATA_T I[128][28][28], DATA_T O[256][28][28], DATA_T W[256][128][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 1)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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

void SW_conv_pw_4_bn(DATA_T I[256][28][28], DATA_T O[256][28][28], DATA_T W[4][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_conv_pw_4_relu(DATA_T I[256][28][28], DATA_T O[256][28][28])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<256; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
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
void SW_conv_dw_5(DATA_T I[256][28][28], DATA_T O[256][28][28], DATA_T W[256][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 3)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;

				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {
						if (x*1 + i < 28 + p && y*1 + j < 28 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv_dw_5_bn(DATA_T I[256][28][28], DATA_T O[256][28][28], DATA_T W[4][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_conv_dw_5_relu(DATA_T I[256][28][28], DATA_T O[256][28][28])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<256; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
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
void SW_conv_pw_5(DATA_T I[256][28][28], DATA_T O[256][28][28], DATA_T W[256][256][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 1)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<256; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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

void SW_conv_pw_5_bn(DATA_T I[256][28][28], DATA_T O[256][28][28], DATA_T W[4][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_conv_pw_5_relu(DATA_T I[256][28][28], DATA_T O[256][28][28])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<256; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
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
void SW_conv_pad_6(DATA_T I[256][28][28],DATA_T O[256][29][29]) {
	int m, x, y, i, j;
	for (m = 0; m < 256; m++) {
		for (x = 0; x < 29; x++) {
			for (y = 0; y < 29; y++) {
				O[m][x][y] = 0;
			}
		}
	}
	for (m = 0; m < 256; m++) {
		for (x = 0; x < 28; x++) {
			for (y = 0; y < 28; y++) {
				O[m][x+ 0][y+ 0] = I[m][x][y];
			}
		}
	}
}
void SW_conv_dw_6(DATA_T I[256][29][29], DATA_T O[256][14][14], DATA_T W[256][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {
						if (x + i <= 29 && y + j <= 29) {
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
void SW_conv_dw_6_bn(DATA_T I[256][14][14], DATA_T O[256][14][14], DATA_T W[4][256]) {
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
void SW_conv_dw_6_relu(DATA_T I[256][14][14], DATA_T O[256][14][14])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<256; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
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
void SW_conv_pw_6(DATA_T I[256][14][14], DATA_T O[512][14][14], DATA_T W[512][256][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<512; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<256; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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

void SW_conv_pw_6_bn(DATA_T I[512][14][14], DATA_T O[512][14][14], DATA_T W[4][512]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_conv_pw_6_relu(DATA_T I[512][14][14], DATA_T O[512][14][14])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<512; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
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
void SW_conv_dw_7(DATA_T I[512][14][14], DATA_T O[512][14][14], DATA_T W[512][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<512; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;

				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {
						if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv_dw_7_bn(DATA_T I[512][14][14], DATA_T O[512][14][14], DATA_T W[4][512]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_conv_dw_7_relu(DATA_T I[512][14][14], DATA_T O[512][14][14])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<512; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
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
void SW_conv_pw_7(DATA_T I[512][14][14], DATA_T O[512][14][14], DATA_T W[512][512][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<512; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<512; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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

void SW_conv_pw_7_bn(DATA_T I[512][14][14], DATA_T O[512][14][14], DATA_T W[4][512]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_conv_pw_7_relu(DATA_T I[512][14][14], DATA_T O[512][14][14])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<512; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
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
void SW_conv_dw_8(DATA_T I[512][14][14], DATA_T O[512][14][14], DATA_T W[512][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<512; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;

				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {
						if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv_dw_8_bn(DATA_T I[512][14][14], DATA_T O[512][14][14], DATA_T W[4][512]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_conv_dw_8_relu(DATA_T I[512][14][14], DATA_T O[512][14][14])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<512; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
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
void SW_conv_pw_8(DATA_T I[512][14][14], DATA_T O[512][14][14], DATA_T W[512][512][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<512; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<512; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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

void SW_conv_pw_8_bn(DATA_T I[512][14][14], DATA_T O[512][14][14], DATA_T W[4][512]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_conv_pw_8_relu(DATA_T I[512][14][14], DATA_T O[512][14][14])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<512; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
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
void SW_conv_dw_9(DATA_T I[512][14][14], DATA_T O[512][14][14], DATA_T W[512][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<512; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;

				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {
						if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv_dw_9_bn(DATA_T I[512][14][14], DATA_T O[512][14][14], DATA_T W[4][512]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_conv_dw_9_relu(DATA_T I[512][14][14], DATA_T O[512][14][14])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<512; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
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
void SW_conv_pw_9(DATA_T I[512][14][14], DATA_T O[512][14][14], DATA_T W[512][512][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<512; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<512; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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

void SW_conv_pw_9_bn(DATA_T I[512][14][14], DATA_T O[512][14][14], DATA_T W[4][512]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_conv_pw_9_relu(DATA_T I[512][14][14], DATA_T O[512][14][14])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<512; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
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
void SW_conv_dw_10(DATA_T I[512][14][14], DATA_T O[512][14][14], DATA_T W[512][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<512; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;

				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {
						if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv_dw_10_bn(DATA_T I[512][14][14], DATA_T O[512][14][14], DATA_T W[4][512]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_conv_dw_10_relu(DATA_T I[512][14][14], DATA_T O[512][14][14])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<512; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
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
void SW_conv_pw_10(DATA_T I[512][14][14], DATA_T O[512][14][14], DATA_T W[512][512][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<512; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<512; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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

void SW_conv_pw_10_bn(DATA_T I[512][14][14], DATA_T O[512][14][14], DATA_T W[4][512]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_conv_pw_10_relu(DATA_T I[512][14][14], DATA_T O[512][14][14])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<512; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
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
void SW_conv_dw_11(DATA_T I[512][14][14], DATA_T O[512][14][14], DATA_T W[512][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<512; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;

				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {
						if (x*1 + i < 14 + p && y*1 + j < 14 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv_dw_11_bn(DATA_T I[512][14][14], DATA_T O[512][14][14], DATA_T W[4][512]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_conv_dw_11_relu(DATA_T I[512][14][14], DATA_T O[512][14][14])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<512; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
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
void SW_conv_pw_11(DATA_T I[512][14][14], DATA_T O[512][14][14], DATA_T W[512][512][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<512; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<512; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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

void SW_conv_pw_11_bn(DATA_T I[512][14][14], DATA_T O[512][14][14], DATA_T W[4][512]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_conv_pw_11_relu(DATA_T I[512][14][14], DATA_T O[512][14][14])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<512; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
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
void SW_conv_pad_12(DATA_T I[512][14][14],DATA_T O[512][15][15]) {
	int m, x, y, i, j;
	for (m = 0; m < 512; m++) {
		for (x = 0; x < 15; x++) {
			for (y = 0; y < 15; y++) {
				O[m][x][y] = 0;
			}
		}
	}
	for (m = 0; m < 512; m++) {
		for (x = 0; x < 14; x++) {
			for (y = 0; y < 14; y++) {
				O[m][x+ 0][y+ 0] = I[m][x][y];
			}
		}
	}
}
void SW_conv_dw_12(DATA_T I[512][15][15], DATA_T O[512][7][7], DATA_T W[512][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<512; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {
						if (x + i <= 15 && y + j <= 15) {
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
void SW_conv_dw_12_bn(DATA_T I[512][7][7], DATA_T O[512][7][7], DATA_T W[4][512]) {
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
void SW_conv_dw_12_relu(DATA_T I[512][7][7], DATA_T O[512][7][7])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<512; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
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
void SW_conv_pw_12(DATA_T I[512][7][7], DATA_T O[1024][7][7], DATA_T W[1024][512][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 1)/2;
	for (m = 0; m<1024; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<512; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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

void SW_conv_pw_12_bn(DATA_T I[1024][7][7], DATA_T O[1024][7][7], DATA_T W[4][1024]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_conv_pw_12_relu(DATA_T I[1024][7][7], DATA_T O[1024][7][7])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<1024; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
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
void SW_conv_dw_13(DATA_T I[1024][7][7], DATA_T O[1024][7][7], DATA_T W[1024][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<1024; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;

				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {
						if (x*1 + i < 7 + p && y*1 + j < 7 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv_dw_13_bn(DATA_T I[1024][7][7], DATA_T O[1024][7][7], DATA_T W[4][1024]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_conv_dw_13_relu(DATA_T I[1024][7][7], DATA_T O[1024][7][7])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<1024; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
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
void SW_conv_pw_13(DATA_T I[1024][7][7], DATA_T O[1024][7][7], DATA_T W[1024][1024][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 1)/2;
	for (m = 0; m<1024; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<1024; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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

void SW_conv_pw_13_bn(DATA_T I[1024][7][7], DATA_T O[1024][7][7], DATA_T W[4][1024]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_conv_pw_13_relu(DATA_T I[1024][7][7], DATA_T O[1024][7][7])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<1024; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
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
void SW_global_average_pooling2d_1(DATA_T I[1024][7][7], DATA_T O[1024]) {
	int m, x, y;
	double avg;
	int div = 7 * 7;
	for (m = 0; m < 1024; m++){
		avg = 0;
		for (x = 0; x < 7; x++) {
			for (y = 0; y < 7; y++) {
				avg += I[m][x][y];
			}
		}
		O[m] = avg/div;
	}
}
void SW_reshape_1(DATA_T I[1024], DATA_T O[1024][1][1]) {
	int i;

	for (i = 0; i < 1024; i++)
		O[i][0][0] = I[i];
}
void SW_conv_preds(DATA_T I[1024][1][1], DATA_T O[1000][1][1], DATA_T W[1000][1024][1][1], DATA_T B[1000]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(1 - 1) - 1 + 1)/2;
	for (m = 0; m<1000; m++) {
		for (x = 0; x<1; x++) {
			for (y = 0; y<1; y++) {
				ofm = 0;
				for (k = 0; k<1024; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i < 1 + p && y + j < 1 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_act_softmax(DATA_T I[1000][1][1], DATA_T O[1000][1][1]) {
	int i;
	DATA_T maximum;
	DATA_T denom = I[0][0][0];
	for (i = 1; i < 1000; i++)
		denom += I[i][0][0];
	for (i = 0; i < 1000; i++){
		O[i][0][0] = I[i][0][0] / denom; //float
		if(i==0)
		    maximum=O[i][0][0];
		else{
		    if(maximum<O[i][0][0])
		        maximum=O[i][0][0];

		}
	}
	for (i = 0; i < 1000; i++){
	    if(maximum!=O[i][0][0])
	        O[i][0][0]=0;
	    else
	        O[i][0][0]=1;
	}

}
void SW_reshape_2(DATA_T I[1000][1][1], DATA_T O[1000]) {
	int i;

	for (i = 0; i < 1000; i++)
		O[i] = I[i][0][0];
}


//argv[1] = init_weight.txt, argv[2] = init_input.txt
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
	static DATA_T W21[128][128][1][1];
	static DATA_T W22[4][128];
	static DATA_T W25[128][3][3];
	static DATA_T W26[4][128];
	static DATA_T W28[256][128][1][1];
	static DATA_T W29[4][256];
	static DATA_T W31[256][3][3];
	static DATA_T W32[4][256];
	static DATA_T W34[256][256][1][1];
	static DATA_T W35[4][256];
	static DATA_T W38[256][3][3];
	static DATA_T W39[4][256];
	static DATA_T W41[512][256][1][1];
	static DATA_T W42[4][512];
	static DATA_T W44[512][3][3];
	static DATA_T W45[4][512];
	static DATA_T W47[512][512][1][1];
	static DATA_T W48[4][512];
	static DATA_T W50[512][3][3];
	static DATA_T W51[4][512];
	static DATA_T W53[512][512][1][1];
	static DATA_T W54[4][512];
	static DATA_T W56[512][3][3];
	static DATA_T W57[4][512];
	static DATA_T W59[512][512][1][1];
	static DATA_T W60[4][512];
	static DATA_T W62[512][3][3];
	static DATA_T W63[4][512];
	static DATA_T W65[512][512][1][1];
	static DATA_T W66[4][512];
	static DATA_T W68[512][3][3];
	static DATA_T W69[4][512];
	static DATA_T W71[512][512][1][1];
	static DATA_T W72[4][512];
	static DATA_T W75[512][3][3];
	static DATA_T W76[4][512];
	static DATA_T W78[1024][512][1][1];
	static DATA_T W79[4][1024];
	static DATA_T W81[1024][3][3];
	static DATA_T W82[4][1024];
	static DATA_T W84[1024][1024][1][1];
	static DATA_T W85[4][1024];
	static DATA_T W89[1000][1024][1][1];
	static DATA_T B89[1000];
	

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
	static DATA_T O20_SW[128][56][56];
	static DATA_T O21_SW[128][56][56];
	static DATA_T O22_SW[128][56][56];
	static DATA_T O23_SW[128][56][56];
	static DATA_T O24_SW[128][57][57];
	static DATA_T O25_SW[128][28][28];
	static DATA_T O26_SW[128][28][28];
	static DATA_T O27_SW[128][28][28];
	static DATA_T O28_SW[256][28][28];
	static DATA_T O29_SW[256][28][28];
	static DATA_T O30_SW[256][28][28];
	static DATA_T O31_SW[256][28][28];
	static DATA_T O32_SW[256][28][28];
	static DATA_T O33_SW[256][28][28];
	static DATA_T O34_SW[256][28][28];
	static DATA_T O35_SW[256][28][28];
	static DATA_T O36_SW[256][28][28];
	static DATA_T O37_SW[256][29][29];
	static DATA_T O38_SW[256][14][14];
	static DATA_T O39_SW[256][14][14];
	static DATA_T O40_SW[256][14][14];
	static DATA_T O41_SW[512][14][14];
	static DATA_T O42_SW[512][14][14];
	static DATA_T O43_SW[512][14][14];
	static DATA_T O44_SW[512][14][14];
	static DATA_T O45_SW[512][14][14];
	static DATA_T O46_SW[512][14][14];
	static DATA_T O47_SW[512][14][14];
	static DATA_T O48_SW[512][14][14];
	static DATA_T O49_SW[512][14][14];
	static DATA_T O50_SW[512][14][14];
	static DATA_T O51_SW[512][14][14];
	static DATA_T O52_SW[512][14][14];
	static DATA_T O53_SW[512][14][14];
	static DATA_T O54_SW[512][14][14];
	static DATA_T O55_SW[512][14][14];
	static DATA_T O56_SW[512][14][14];
	static DATA_T O57_SW[512][14][14];
	static DATA_T O58_SW[512][14][14];
	static DATA_T O59_SW[512][14][14];
	static DATA_T O60_SW[512][14][14];
	static DATA_T O61_SW[512][14][14];
	static DATA_T O62_SW[512][14][14];
	static DATA_T O63_SW[512][14][14];
	static DATA_T O64_SW[512][14][14];
	static DATA_T O65_SW[512][14][14];
	static DATA_T O66_SW[512][14][14];
	static DATA_T O67_SW[512][14][14];
	static DATA_T O68_SW[512][14][14];
	static DATA_T O69_SW[512][14][14];
	static DATA_T O70_SW[512][14][14];
	static DATA_T O71_SW[512][14][14];
	static DATA_T O72_SW[512][14][14];
	static DATA_T O73_SW[512][14][14];
	static DATA_T O74_SW[512][15][15];
	static DATA_T O75_SW[512][7][7];
	static DATA_T O76_SW[512][7][7];
	static DATA_T O77_SW[512][7][7];
	static DATA_T O78_SW[1024][7][7];
	static DATA_T O79_SW[1024][7][7];
	static DATA_T O80_SW[1024][7][7];
	static DATA_T O81_SW[1024][7][7];
	static DATA_T O82_SW[1024][7][7];
	static DATA_T O83_SW[1024][7][7];
	static DATA_T O84_SW[1024][7][7];
	static DATA_T O85_SW[1024][7][7];
	static DATA_T O86_SW[1024][7][7];
	static DATA_T O87_SW[1024];
	static DATA_T O88_SW[1024][1][1];
	static DATA_T O89_SW[1000][1][1];
	static DATA_T O90_SW[1000][1][1];
	static DATA_T O91_SW[1000];
	

    FILE *w_stream = fopen(argv[1], "rb");
    if (w_stream == NULL) printf("weight file was not opened");
    FILE *i_stream = fopen(argv[2], "rb");
    if (i_stream == NULL) printf("input file was not opened");
    FILE *o_stream = fopen("Produced_code/mobilenet/Output/c_output.txt", "w");
    if (o_stream == NULL) printf("Output file was not opened");
    FILE *c_num = fopen("Produced_code/mobilenet/Output/c_output_num.txt", "w");
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
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W21[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B21[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W22[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W25[j][m][k] = (DATA_T) trash;
			}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B25[m] = (DATA_T) trash;
}
*/
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W26[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W28[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B28[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W29[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W31[j][m][k] = (DATA_T) trash;
			}
	}
}

/*
for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B31[m] = (DATA_T) trash;
}
*/
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W32[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W34[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B34[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W35[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W38[j][m][k] = (DATA_T) trash;
			}
	}
}

/*
for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B38[m] = (DATA_T) trash;
}
*/
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W39[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W41[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B41[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 512 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W42[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W44[j][m][k] = (DATA_T) trash;
			}
	}
}

/*
for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B44[m] = (DATA_T) trash;
}
*/
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 512 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W45[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W47[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B47[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 512 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W48[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W50[j][m][k] = (DATA_T) trash;
			}
	}
}

/*
for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B50[m] = (DATA_T) trash;
}
*/
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 512 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W51[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W53[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B53[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 512 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W54[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W56[j][m][k] = (DATA_T) trash;
			}
	}
}

/*
for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B56[m] = (DATA_T) trash;
}
*/
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 512 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W57[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W59[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B59[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 512 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W60[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W62[j][m][k] = (DATA_T) trash;
			}
	}
}

/*
for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B62[m] = (DATA_T) trash;
}
*/
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 512 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W63[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W65[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B65[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 512 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W66[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W68[j][m][k] = (DATA_T) trash;
			}
	}
}

/*
for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B68[m] = (DATA_T) trash;
}
*/
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 512 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W69[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W71[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B71[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 512 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W72[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W75[j][m][k] = (DATA_T) trash;
			}
	}
}

/*
for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B75[m] = (DATA_T) trash;
}
*/
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 512 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W76[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 1024 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W78[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 1024 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B78[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1024 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W79[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
			for (j = 0; j < 1024 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W81[j][m][k] = (DATA_T) trash;
			}
	}
}

/*
for (m = 0; m < 1024 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B81[m] = (DATA_T) trash;
}
*/
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1024 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W82[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1024 ; i++) {
			for (j = 0; j < 1024 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W84[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 1024 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B84[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1024 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W85[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 1024 ; i++) {
			for (j = 0; j < 1000 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W89[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}


for (m = 0; m < 1000 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B89[m] = (DATA_T) trash;
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
	printf("[C_verifier.cpp]Calculate Relu20\n\n");
	SW_conv_dw_3_relu(O19_SW,O20_SW);
	printf("[C_verifier.cpp]Calculate Conv2D21\n\n");
	SW_conv_pw_3(O20_SW,O21_SW,W21);
	printf("[C_verifier.cpp]Calculate BatchNormalization22\n\n");
	SW_conv_pw_3_bn(O21_SW,O22_SW, W22);
	printf("[C_verifier.cpp]Calculate Relu23\n\n");
	SW_conv_pw_3_relu(O22_SW,O23_SW);
	printf("[C_verifier.cpp]Calculate ZeroPadding2D24\n\n");
	SW_conv_pad_4(O23_SW,O24_SW);
	printf("[C_verifier.cpp]Calculate DepthwiseConv2D25\n\n");
	SW_conv_dw_4(O24_SW,O25_SW,W25);
	printf("[C_verifier.cpp]Calculate BatchNormalization26\n\n");
	SW_conv_dw_4_bn(O25_SW,O26_SW, W26);
	printf("[C_verifier.cpp]Calculate Relu27\n\n");
	SW_conv_dw_4_relu(O26_SW,O27_SW);
	printf("[C_verifier.cpp]Calculate Conv2D28\n\n");
	SW_conv_pw_4(O27_SW,O28_SW,W28);
	printf("[C_verifier.cpp]Calculate BatchNormalization29\n\n");
	SW_conv_pw_4_bn(O28_SW,O29_SW, W29);
	printf("[C_verifier.cpp]Calculate Relu30\n\n");
	SW_conv_pw_4_relu(O29_SW,O30_SW);
	printf("[C_verifier.cpp]Calculate DepthwiseConv2D31\n\n");
	SW_conv_dw_5(O30_SW,O31_SW,W31);
	printf("[C_verifier.cpp]Calculate BatchNormalization32\n\n");
	SW_conv_dw_5_bn(O31_SW,O32_SW, W32);
	printf("[C_verifier.cpp]Calculate Relu33\n\n");
	SW_conv_dw_5_relu(O32_SW,O33_SW);
	printf("[C_verifier.cpp]Calculate Conv2D34\n\n");
	SW_conv_pw_5(O33_SW,O34_SW,W34);
	printf("[C_verifier.cpp]Calculate BatchNormalization35\n\n");
	SW_conv_pw_5_bn(O34_SW,O35_SW, W35);
	printf("[C_verifier.cpp]Calculate Relu36\n\n");
	SW_conv_pw_5_relu(O35_SW,O36_SW);
	printf("[C_verifier.cpp]Calculate ZeroPadding2D37\n\n");
	SW_conv_pad_6(O36_SW,O37_SW);
	printf("[C_verifier.cpp]Calculate DepthwiseConv2D38\n\n");
	SW_conv_dw_6(O37_SW,O38_SW,W38);
	printf("[C_verifier.cpp]Calculate BatchNormalization39\n\n");
	SW_conv_dw_6_bn(O38_SW,O39_SW, W39);
	printf("[C_verifier.cpp]Calculate Relu40\n\n");
	SW_conv_dw_6_relu(O39_SW,O40_SW);
	printf("[C_verifier.cpp]Calculate Conv2D41\n\n");
	SW_conv_pw_6(O40_SW,O41_SW,W41);
	printf("[C_verifier.cpp]Calculate BatchNormalization42\n\n");
	SW_conv_pw_6_bn(O41_SW,O42_SW, W42);
	printf("[C_verifier.cpp]Calculate Relu43\n\n");
	SW_conv_pw_6_relu(O42_SW,O43_SW);
	printf("[C_verifier.cpp]Calculate DepthwiseConv2D44\n\n");
	SW_conv_dw_7(O43_SW,O44_SW,W44);
	printf("[C_verifier.cpp]Calculate BatchNormalization45\n\n");
	SW_conv_dw_7_bn(O44_SW,O45_SW, W45);
	printf("[C_verifier.cpp]Calculate Relu46\n\n");
	SW_conv_dw_7_relu(O45_SW,O46_SW);
	printf("[C_verifier.cpp]Calculate Conv2D47\n\n");
	SW_conv_pw_7(O46_SW,O47_SW,W47);
	printf("[C_verifier.cpp]Calculate BatchNormalization48\n\n");
	SW_conv_pw_7_bn(O47_SW,O48_SW, W48);
	printf("[C_verifier.cpp]Calculate Relu49\n\n");
	SW_conv_pw_7_relu(O48_SW,O49_SW);
	printf("[C_verifier.cpp]Calculate DepthwiseConv2D50\n\n");
	SW_conv_dw_8(O49_SW,O50_SW,W50);
	printf("[C_verifier.cpp]Calculate BatchNormalization51\n\n");
	SW_conv_dw_8_bn(O50_SW,O51_SW, W51);
	printf("[C_verifier.cpp]Calculate Relu52\n\n");
	SW_conv_dw_8_relu(O51_SW,O52_SW);
	printf("[C_verifier.cpp]Calculate Conv2D53\n\n");
	SW_conv_pw_8(O52_SW,O53_SW,W53);
	printf("[C_verifier.cpp]Calculate BatchNormalization54\n\n");
	SW_conv_pw_8_bn(O53_SW,O54_SW, W54);
	printf("[C_verifier.cpp]Calculate Relu55\n\n");
	SW_conv_pw_8_relu(O54_SW,O55_SW);
	printf("[C_verifier.cpp]Calculate DepthwiseConv2D56\n\n");
	SW_conv_dw_9(O55_SW,O56_SW,W56);
	printf("[C_verifier.cpp]Calculate BatchNormalization57\n\n");
	SW_conv_dw_9_bn(O56_SW,O57_SW, W57);
	printf("[C_verifier.cpp]Calculate Relu58\n\n");
	SW_conv_dw_9_relu(O57_SW,O58_SW);
	printf("[C_verifier.cpp]Calculate Conv2D59\n\n");
	SW_conv_pw_9(O58_SW,O59_SW,W59);
	printf("[C_verifier.cpp]Calculate BatchNormalization60\n\n");
	SW_conv_pw_9_bn(O59_SW,O60_SW, W60);
	printf("[C_verifier.cpp]Calculate Relu61\n\n");
	SW_conv_pw_9_relu(O60_SW,O61_SW);
	printf("[C_verifier.cpp]Calculate DepthwiseConv2D62\n\n");
	SW_conv_dw_10(O61_SW,O62_SW,W62);
	printf("[C_verifier.cpp]Calculate BatchNormalization63\n\n");
	SW_conv_dw_10_bn(O62_SW,O63_SW, W63);
	printf("[C_verifier.cpp]Calculate Relu64\n\n");
	SW_conv_dw_10_relu(O63_SW,O64_SW);
	printf("[C_verifier.cpp]Calculate Conv2D65\n\n");
	SW_conv_pw_10(O64_SW,O65_SW,W65);
	printf("[C_verifier.cpp]Calculate BatchNormalization66\n\n");
	SW_conv_pw_10_bn(O65_SW,O66_SW, W66);
	printf("[C_verifier.cpp]Calculate Relu67\n\n");
	SW_conv_pw_10_relu(O66_SW,O67_SW);
	printf("[C_verifier.cpp]Calculate DepthwiseConv2D68\n\n");
	SW_conv_dw_11(O67_SW,O68_SW,W68);
	printf("[C_verifier.cpp]Calculate BatchNormalization69\n\n");
	SW_conv_dw_11_bn(O68_SW,O69_SW, W69);
	printf("[C_verifier.cpp]Calculate Relu70\n\n");
	SW_conv_dw_11_relu(O69_SW,O70_SW);
	printf("[C_verifier.cpp]Calculate Conv2D71\n\n");
	SW_conv_pw_11(O70_SW,O71_SW,W71);
	printf("[C_verifier.cpp]Calculate BatchNormalization72\n\n");
	SW_conv_pw_11_bn(O71_SW,O72_SW, W72);
	printf("[C_verifier.cpp]Calculate Relu73\n\n");
	SW_conv_pw_11_relu(O72_SW,O73_SW);
	printf("[C_verifier.cpp]Calculate ZeroPadding2D74\n\n");
	SW_conv_pad_12(O73_SW,O74_SW);
	printf("[C_verifier.cpp]Calculate DepthwiseConv2D75\n\n");
	SW_conv_dw_12(O74_SW,O75_SW,W75);
	printf("[C_verifier.cpp]Calculate BatchNormalization76\n\n");
	SW_conv_dw_12_bn(O75_SW,O76_SW, W76);
	printf("[C_verifier.cpp]Calculate Relu77\n\n");
	SW_conv_dw_12_relu(O76_SW,O77_SW);
	printf("[C_verifier.cpp]Calculate Conv2D78\n\n");
	SW_conv_pw_12(O77_SW,O78_SW,W78);
	printf("[C_verifier.cpp]Calculate BatchNormalization79\n\n");
	SW_conv_pw_12_bn(O78_SW,O79_SW, W79);
	printf("[C_verifier.cpp]Calculate Relu80\n\n");
	SW_conv_pw_12_relu(O79_SW,O80_SW);
	printf("[C_verifier.cpp]Calculate DepthwiseConv2D81\n\n");
	SW_conv_dw_13(O80_SW,O81_SW,W81);
	printf("[C_verifier.cpp]Calculate BatchNormalization82\n\n");
	SW_conv_dw_13_bn(O81_SW,O82_SW, W82);
	printf("[C_verifier.cpp]Calculate Relu83\n\n");
	SW_conv_dw_13_relu(O82_SW,O83_SW);
	printf("[C_verifier.cpp]Calculate Conv2D84\n\n");
	SW_conv_pw_13(O83_SW,O84_SW,W84);
	printf("[C_verifier.cpp]Calculate BatchNormalization85\n\n");
	SW_conv_pw_13_bn(O84_SW,O85_SW, W85);
	printf("[C_verifier.cpp]Calculate Relu86\n\n");
	SW_conv_pw_13_relu(O85_SW,O86_SW);
	printf("[C_verifier.cpp]Calculate GlobalAveragePooling2D87\n\n");
	SW_global_average_pooling2d_1(O86_SW,O87_SW);
	printf("[C_verifier.cpp]Calculate Reshape88\n\n");
	SW_reshape_1(O87_SW,O88_SW);
	printf("[C_verifier.cpp]Calculate Conv2D89\n\n");
	SW_conv_preds(O88_SW,O89_SW,W89,B89);
	printf("[C_verifier.cpp]Calculate Activation(Relu)90\n\n");
	SW_act_softmax(O89_SW,O90_SW);
	printf("[C_verifier.cpp]Calculate Reshape91\n\n");
	SW_reshape_2(O90_SW,O91_SW);
	

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


fprintf(o_stream,"%s","ReLU : [[");
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


fprintf(o_stream,"%s","Conv2D : [[");
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


fprintf(o_stream,"%s","BatchNormalization : [[");
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


fprintf(o_stream,"%s","ReLU : [[");
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


fprintf(o_stream,"%s","ZeroPadding2D : [[");
for (k = 0; k < 57 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 57 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O24_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O24_SW[y][k][x]);
		}
		if(x != 57 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 57 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","DepthwiseConv2D : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O25_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O25_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O26_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O26_SW[y][k][x]);
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


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O27_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O27_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O28_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O28_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O29_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O29_SW[y][k][x]);
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


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O30_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O30_SW[y][k][x]);
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


fprintf(o_stream,"%s","DepthwiseConv2D : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O31_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O31_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O32_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O32_SW[y][k][x]);
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


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O33_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O33_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O34_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O34_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O35_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O35_SW[y][k][x]);
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


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O36_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O36_SW[y][k][x]);
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


fprintf(o_stream,"%s","ZeroPadding2D : [[");
for (k = 0; k < 29 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 29 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O37_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O37_SW[y][k][x]);
		}
		if(x != 29 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 29 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","DepthwiseConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O38_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O38_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O39_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O39_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O40_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O40_SW[y][k][x]);
		}
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
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O41_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O41_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O42_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O42_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O43_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O43_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","DepthwiseConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O44_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O44_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O45_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O45_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O46_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O46_SW[y][k][x]);
		}
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
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O47_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O47_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O48_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O48_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O49_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O49_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","DepthwiseConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O50_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O50_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O51_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O51_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O52_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O52_SW[y][k][x]);
		}
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
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O53_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O53_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O54_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O54_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O55_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O55_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","DepthwiseConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O56_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O56_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O57_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O57_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O58_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O58_SW[y][k][x]);
		}
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
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O59_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O59_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O60_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O60_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O61_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O61_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","DepthwiseConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O62_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O62_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O63_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O63_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O64_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O64_SW[y][k][x]);
		}
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
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O65_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O65_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O66_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O66_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O67_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O67_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","DepthwiseConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O68_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O68_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O69_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O69_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O70_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O70_SW[y][k][x]);
		}
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
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O71_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O71_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O72_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O72_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O73_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O73_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","ZeroPadding2D : [[");
for (k = 0; k < 15 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 15 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O74_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O74_SW[y][k][x]);
		}
		if(x != 15 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 15 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","DepthwiseConv2D : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O75_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O75_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O76_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O76_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O77_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O77_SW[y][k][x]);
		}
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
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O78_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O78_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O79_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O79_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O80_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O80_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","DepthwiseConv2D : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O81_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O81_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O82_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O82_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O83_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O83_SW[y][k][x]);
		}
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
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O84_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O84_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O85_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O85_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O86_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O86_SW[y][k][x]);
		}
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
for (k = 0; k <  1024 ; k++) {
	fprintf(o_stream,"%.6f ",O87_SW[k]);
	fprintf(c_num,"%.6f ",O87_SW[k]);
}
fprintf(o_stream,"%s","]]\n\n");


fprintf(o_stream,"%s","Reshape : [[");
for (k = 0; k < 1 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 1 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O88_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O88_SW[y][k][x]);
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 1 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 1 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1000 ; y++) {
			fprintf(o_stream,"%.6f ",O89_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O89_SW[y][k][x]);
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 1 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 1 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1000 ; y++) {
			fprintf(o_stream,"%.6f ",O90_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O90_SW[y][k][x]);
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


fprintf(o_stream,"%s","Reshape : [[");
for (k = 0; k <  1000 ; k++) {
	fprintf(o_stream,"%.6f ",O91_SW[k]);
	fprintf(c_num,"%.6f ",O91_SW[k]);
}
fprintf(o_stream,"%s","]]\n\n");





    fclose(w_stream);
    fclose(i_stream);
    fclose(o_stream);
    fclose(c_num);

    return 0;
}