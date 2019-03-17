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

void SW_block_3_expand_BN(DATA_T I[144][56][56], DATA_T O[144][56][56], DATA_T W[4][144]) {
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
void SW_block_3_expand_relu(DATA_T I[144][56][56], DATA_T O[144][56][56])
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
void SW_block_3_depthwise(DATA_T I[144][56][56], DATA_T O[144][28][28], DATA_T W[144][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (2 *(28 - 1) - 56 + 3)/2;
	for (m = 0; m<144; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;

				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {
						if (x*2 + i < 56 + p && y*2 + j < 56 + p && x*2 + i -p >= 0 && y*2 + j -p >= 0) {
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

void SW_block_3_depthwise_BN(DATA_T I[144][28][28], DATA_T O[144][28][28], DATA_T W[4][144]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 144; m++){
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
void SW_block_3_depthwise_relu(DATA_T I[144][28][28], DATA_T O[144][28][28])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<144; m++) {
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
void SW_block_3_project(DATA_T I[144][28][28], DATA_T O[32][28][28], DATA_T W[32][144][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<144; k++) {
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

void SW_block_3_project_BN(DATA_T I[32][28][28], DATA_T O[32][28][28], DATA_T W[4][32]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 32; m++){
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
void SW_block_4_expand(DATA_T I[32][28][28], DATA_T O[192][28][28], DATA_T W[192][32][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<32; k++) {
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

void SW_block_4_expand_BN(DATA_T I[192][28][28], DATA_T O[192][28][28], DATA_T W[4][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_block_4_expand_relu(DATA_T I[192][28][28], DATA_T O[192][28][28])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<192; m++) {
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
void SW_block_4_depthwise(DATA_T I[192][28][28], DATA_T O[192][28][28], DATA_T W[192][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 3)/2;
	for (m = 0; m<192; m++) {
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

void SW_block_4_depthwise_BN(DATA_T I[192][28][28], DATA_T O[192][28][28], DATA_T W[4][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_block_4_depthwise_relu(DATA_T I[192][28][28], DATA_T O[192][28][28])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<192; m++) {
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
void SW_block_4_project(DATA_T I[192][28][28], DATA_T O[32][28][28], DATA_T W[32][192][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<192; k++) {
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

void SW_block_4_project_BN(DATA_T I[32][28][28], DATA_T O[32][28][28], DATA_T W[4][32]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 32; m++){
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
void SW_block_4_add(DATA_T I1[32][28][28], DATA_T I2[32][28][28], DATA_T O[32][28][28]) {
	int m, x, y;
	for (m = 0; m< 32; m++) {
		for (x = 0; x < 28; x++) {
			for(y = 0; y < 28; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_block_5_expand(DATA_T I[32][28][28], DATA_T O[192][28][28], DATA_T W[192][32][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<32; k++) {
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

void SW_block_5_expand_BN(DATA_T I[192][28][28], DATA_T O[192][28][28], DATA_T W[4][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_block_5_expand_relu(DATA_T I[192][28][28], DATA_T O[192][28][28])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<192; m++) {
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
void SW_block_5_depthwise(DATA_T I[192][28][28], DATA_T O[192][28][28], DATA_T W[192][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 3)/2;
	for (m = 0; m<192; m++) {
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

void SW_block_5_depthwise_BN(DATA_T I[192][28][28], DATA_T O[192][28][28], DATA_T W[4][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_block_5_depthwise_relu(DATA_T I[192][28][28], DATA_T O[192][28][28])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<192; m++) {
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
void SW_block_5_project(DATA_T I[192][28][28], DATA_T O[32][28][28], DATA_T W[32][192][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<192; k++) {
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

void SW_block_5_project_BN(DATA_T I[32][28][28], DATA_T O[32][28][28], DATA_T W[4][32]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 32; m++){
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
void SW_block_5_add(DATA_T I1[32][28][28], DATA_T I2[32][28][28], DATA_T O[32][28][28]) {
	int m, x, y;
	for (m = 0; m< 32; m++) {
		for (x = 0; x < 28; x++) {
			for(y = 0; y < 28; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_block_6_expand(DATA_T I[32][28][28], DATA_T O[192][28][28], DATA_T W[192][32][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<32; k++) {
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

void SW_block_6_expand_BN(DATA_T I[192][28][28], DATA_T O[192][28][28], DATA_T W[4][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_block_6_expand_relu(DATA_T I[192][28][28], DATA_T O[192][28][28])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<192; m++) {
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
void SW_block_6_depthwise(DATA_T I[192][28][28], DATA_T O[192][14][14], DATA_T W[192][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (2 *(14 - 1) - 28 + 3)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;

				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {
						if (x*2 + i < 28 + p && y*2 + j < 28 + p && x*2 + i -p >= 0 && y*2 + j -p >= 0) {
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

void SW_block_6_depthwise_BN(DATA_T I[192][14][14], DATA_T O[192][14][14], DATA_T W[4][192]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 192; m++){
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
void SW_block_6_depthwise_relu(DATA_T I[192][14][14], DATA_T O[192][14][14])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<192; m++) {
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
void SW_block_6_project(DATA_T I[192][14][14], DATA_T O[64][14][14], DATA_T W[64][192][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<192; k++) {
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

void SW_block_6_project_BN(DATA_T I[64][14][14], DATA_T O[64][14][14], DATA_T W[4][64]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 64; m++){
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
void SW_block_7_expand(DATA_T I[64][14][14], DATA_T O[384][14][14], DATA_T W[384][64][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<384; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<64; k++) {
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

void SW_block_7_expand_BN(DATA_T I[384][14][14], DATA_T O[384][14][14], DATA_T W[4][384]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_block_7_expand_relu(DATA_T I[384][14][14], DATA_T O[384][14][14])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<384; m++) {
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
void SW_block_7_depthwise(DATA_T I[384][14][14], DATA_T O[384][14][14], DATA_T W[384][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<384; m++) {
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

void SW_block_7_depthwise_BN(DATA_T I[384][14][14], DATA_T O[384][14][14], DATA_T W[4][384]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_block_7_depthwise_relu(DATA_T I[384][14][14], DATA_T O[384][14][14])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<384; m++) {
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
void SW_block_7_project(DATA_T I[384][14][14], DATA_T O[64][14][14], DATA_T W[64][384][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
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

void SW_block_7_project_BN(DATA_T I[64][14][14], DATA_T O[64][14][14], DATA_T W[4][64]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 64; m++){
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
void SW_block_7_add(DATA_T I1[64][14][14], DATA_T I2[64][14][14], DATA_T O[64][14][14]) {
	int m, x, y;
	for (m = 0; m< 64; m++) {
		for (x = 0; x < 14; x++) {
			for(y = 0; y < 14; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_block_8_expand(DATA_T I[64][14][14], DATA_T O[384][14][14], DATA_T W[384][64][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<384; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<64; k++) {
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

void SW_block_8_expand_BN(DATA_T I[384][14][14], DATA_T O[384][14][14], DATA_T W[4][384]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_block_8_expand_relu(DATA_T I[384][14][14], DATA_T O[384][14][14])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<384; m++) {
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
void SW_block_8_depthwise(DATA_T I[384][14][14], DATA_T O[384][14][14], DATA_T W[384][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<384; m++) {
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

void SW_block_8_depthwise_BN(DATA_T I[384][14][14], DATA_T O[384][14][14], DATA_T W[4][384]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_block_8_depthwise_relu(DATA_T I[384][14][14], DATA_T O[384][14][14])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<384; m++) {
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
void SW_block_8_project(DATA_T I[384][14][14], DATA_T O[64][14][14], DATA_T W[64][384][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
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

void SW_block_8_project_BN(DATA_T I[64][14][14], DATA_T O[64][14][14], DATA_T W[4][64]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 64; m++){
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
void SW_block_8_add(DATA_T I1[64][14][14], DATA_T I2[64][14][14], DATA_T O[64][14][14]) {
	int m, x, y;
	for (m = 0; m< 64; m++) {
		for (x = 0; x < 14; x++) {
			for(y = 0; y < 14; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_block_9_expand(DATA_T I[64][14][14], DATA_T O[384][14][14], DATA_T W[384][64][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<384; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<64; k++) {
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

void SW_block_9_expand_BN(DATA_T I[384][14][14], DATA_T O[384][14][14], DATA_T W[4][384]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_block_9_expand_relu(DATA_T I[384][14][14], DATA_T O[384][14][14])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<384; m++) {
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
void SW_block_9_depthwise(DATA_T I[384][14][14], DATA_T O[384][14][14], DATA_T W[384][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<384; m++) {
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

void SW_block_9_depthwise_BN(DATA_T I[384][14][14], DATA_T O[384][14][14], DATA_T W[4][384]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_block_9_depthwise_relu(DATA_T I[384][14][14], DATA_T O[384][14][14])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<384; m++) {
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
void SW_block_9_project(DATA_T I[384][14][14], DATA_T O[64][14][14], DATA_T W[64][384][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
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

void SW_block_9_project_BN(DATA_T I[64][14][14], DATA_T O[64][14][14], DATA_T W[4][64]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 64; m++){
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
void SW_block_9_add(DATA_T I1[64][14][14], DATA_T I2[64][14][14], DATA_T O[64][14][14]) {
	int m, x, y;
	for (m = 0; m< 64; m++) {
		for (x = 0; x < 14; x++) {
			for(y = 0; y < 14; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_block_10_expand(DATA_T I[64][14][14], DATA_T O[384][14][14], DATA_T W[384][64][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<384; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<64; k++) {
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

void SW_block_10_expand_BN(DATA_T I[384][14][14], DATA_T O[384][14][14], DATA_T W[4][384]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_block_10_expand_relu(DATA_T I[384][14][14], DATA_T O[384][14][14])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<384; m++) {
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
void SW_block_10_depthwise(DATA_T I[384][14][14], DATA_T O[384][14][14], DATA_T W[384][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<384; m++) {
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

void SW_block_10_depthwise_BN(DATA_T I[384][14][14], DATA_T O[384][14][14], DATA_T W[4][384]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_block_10_depthwise_relu(DATA_T I[384][14][14], DATA_T O[384][14][14])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<384; m++) {
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
void SW_block_10_project(DATA_T I[384][14][14], DATA_T O[96][14][14], DATA_T W[96][384][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<96; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<384; k++) {
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

void SW_block_10_project_BN(DATA_T I[96][14][14], DATA_T O[96][14][14], DATA_T W[4][96]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 96; m++){
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
void SW_block_11_expand(DATA_T I[96][14][14], DATA_T O[576][14][14], DATA_T W[576][96][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<576; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<96; k++) {
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

void SW_block_11_expand_BN(DATA_T I[576][14][14], DATA_T O[576][14][14], DATA_T W[4][576]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_block_11_expand_relu(DATA_T I[576][14][14], DATA_T O[576][14][14])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<576; m++) {
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
void SW_block_11_depthwise(DATA_T I[576][14][14], DATA_T O[576][14][14], DATA_T W[576][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<576; m++) {
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

void SW_block_11_depthwise_BN(DATA_T I[576][14][14], DATA_T O[576][14][14], DATA_T W[4][576]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_block_11_depthwise_relu(DATA_T I[576][14][14], DATA_T O[576][14][14])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<576; m++) {
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
void SW_block_11_project(DATA_T I[576][14][14], DATA_T O[96][14][14], DATA_T W[96][576][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<96; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<576; k++) {
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

void SW_block_11_project_BN(DATA_T I[96][14][14], DATA_T O[96][14][14], DATA_T W[4][96]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 96; m++){
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
void SW_block_11_add(DATA_T I1[96][14][14], DATA_T I2[96][14][14], DATA_T O[96][14][14]) {
	int m, x, y;
	for (m = 0; m< 96; m++) {
		for (x = 0; x < 14; x++) {
			for(y = 0; y < 14; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_block_12_expand(DATA_T I[96][14][14], DATA_T O[576][14][14], DATA_T W[576][96][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<576; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<96; k++) {
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

void SW_block_12_expand_BN(DATA_T I[576][14][14], DATA_T O[576][14][14], DATA_T W[4][576]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_block_12_expand_relu(DATA_T I[576][14][14], DATA_T O[576][14][14])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<576; m++) {
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
void SW_block_12_depthwise(DATA_T I[576][14][14], DATA_T O[576][14][14], DATA_T W[576][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<576; m++) {
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

void SW_block_12_depthwise_BN(DATA_T I[576][14][14], DATA_T O[576][14][14], DATA_T W[4][576]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_block_12_depthwise_relu(DATA_T I[576][14][14], DATA_T O[576][14][14])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<576; m++) {
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
void SW_block_12_project(DATA_T I[576][14][14], DATA_T O[96][14][14], DATA_T W[96][576][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<96; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<576; k++) {
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

void SW_block_12_project_BN(DATA_T I[96][14][14], DATA_T O[96][14][14], DATA_T W[4][96]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 96; m++){
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
void SW_block_12_add(DATA_T I1[96][14][14], DATA_T I2[96][14][14], DATA_T O[96][14][14]) {
	int m, x, y;
	for (m = 0; m< 96; m++) {
		for (x = 0; x < 14; x++) {
			for(y = 0; y < 14; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_block_13_expand(DATA_T I[96][14][14], DATA_T O[576][14][14], DATA_T W[576][96][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<576; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<96; k++) {
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

void SW_block_13_expand_BN(DATA_T I[576][14][14], DATA_T O[576][14][14], DATA_T W[4][576]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_block_13_expand_relu(DATA_T I[576][14][14], DATA_T O[576][14][14])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<576; m++) {
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
void SW_block_13_depthwise(DATA_T I[576][14][14], DATA_T O[576][7][7], DATA_T W[576][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (2 *(7 - 1) - 14 + 3)/2;
	for (m = 0; m<576; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;

				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {
						if (x*2 + i < 14 + p && y*2 + j < 14 + p && x*2 + i -p >= 0 && y*2 + j -p >= 0) {
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

void SW_block_13_depthwise_BN(DATA_T I[576][7][7], DATA_T O[576][7][7], DATA_T W[4][576]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 576; m++){
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
void SW_block_13_depthwise_relu(DATA_T I[576][7][7], DATA_T O[576][7][7])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<576; m++) {
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
void SW_block_13_project(DATA_T I[576][7][7], DATA_T O[160][7][7], DATA_T W[160][576][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<576; k++) {
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

void SW_block_13_project_BN(DATA_T I[160][7][7], DATA_T O[160][7][7], DATA_T W[4][160]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 160; m++){
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
void SW_block_14_expand(DATA_T I[160][7][7], DATA_T O[960][7][7], DATA_T W[960][160][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 1)/2;
	for (m = 0; m<960; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<160; k++) {
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

void SW_block_14_expand_BN(DATA_T I[960][7][7], DATA_T O[960][7][7], DATA_T W[4][960]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_block_14_expand_relu(DATA_T I[960][7][7], DATA_T O[960][7][7])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<960; m++) {
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
void SW_block_14_depthwise(DATA_T I[960][7][7], DATA_T O[960][7][7], DATA_T W[960][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<960; m++) {
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

void SW_block_14_depthwise_BN(DATA_T I[960][7][7], DATA_T O[960][7][7], DATA_T W[4][960]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_block_14_depthwise_relu(DATA_T I[960][7][7], DATA_T O[960][7][7])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<960; m++) {
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
void SW_block_14_project(DATA_T I[960][7][7], DATA_T O[160][7][7], DATA_T W[160][960][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<960; k++) {
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

void SW_block_14_project_BN(DATA_T I[160][7][7], DATA_T O[160][7][7], DATA_T W[4][160]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 160; m++){
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
void SW_block_14_add(DATA_T I1[160][7][7], DATA_T I2[160][7][7], DATA_T O[160][7][7]) {
	int m, x, y;
	for (m = 0; m< 160; m++) {
		for (x = 0; x < 7; x++) {
			for(y = 0; y < 7; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_block_15_expand(DATA_T I[160][7][7], DATA_T O[960][7][7], DATA_T W[960][160][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 1)/2;
	for (m = 0; m<960; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<160; k++) {
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

void SW_block_15_expand_BN(DATA_T I[960][7][7], DATA_T O[960][7][7], DATA_T W[4][960]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_block_15_expand_relu(DATA_T I[960][7][7], DATA_T O[960][7][7])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<960; m++) {
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
void SW_block_15_depthwise(DATA_T I[960][7][7], DATA_T O[960][7][7], DATA_T W[960][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<960; m++) {
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

void SW_block_15_depthwise_BN(DATA_T I[960][7][7], DATA_T O[960][7][7], DATA_T W[4][960]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_block_15_depthwise_relu(DATA_T I[960][7][7], DATA_T O[960][7][7])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<960; m++) {
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
void SW_block_15_project(DATA_T I[960][7][7], DATA_T O[160][7][7], DATA_T W[160][960][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<960; k++) {
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

void SW_block_15_project_BN(DATA_T I[160][7][7], DATA_T O[160][7][7], DATA_T W[4][160]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 160; m++){
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
void SW_block_15_add(DATA_T I1[160][7][7], DATA_T I2[160][7][7], DATA_T O[160][7][7]) {
	int m, x, y;
	for (m = 0; m< 160; m++) {
		for (x = 0; x < 7; x++) {
			for(y = 0; y < 7; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_block_16_expand(DATA_T I[160][7][7], DATA_T O[960][7][7], DATA_T W[960][160][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 1)/2;
	for (m = 0; m<960; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<160; k++) {
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

void SW_block_16_expand_BN(DATA_T I[960][7][7], DATA_T O[960][7][7], DATA_T W[4][960]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_block_16_expand_relu(DATA_T I[960][7][7], DATA_T O[960][7][7])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<960; m++) {
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
void SW_block_16_depthwise(DATA_T I[960][7][7], DATA_T O[960][7][7], DATA_T W[960][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<960; m++) {
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

void SW_block_16_depthwise_BN(DATA_T I[960][7][7], DATA_T O[960][7][7], DATA_T W[4][960]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_block_16_depthwise_relu(DATA_T I[960][7][7], DATA_T O[960][7][7])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<960; m++) {
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
void SW_block_16_project(DATA_T I[960][7][7], DATA_T O[320][7][7], DATA_T W[320][960][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 1)/2;
	for (m = 0; m<320; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<960; k++) {
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

void SW_block_16_project_BN(DATA_T I[320][7][7], DATA_T O[320][7][7], DATA_T W[4][320]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 320; m++){
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
void SW_Conv_1(DATA_T I[320][7][7], DATA_T O[1280][7][7], DATA_T W[1280][320][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<1280; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<320; k++) {
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

void SW_Conv_1_bn(DATA_T I[1280][7][7], DATA_T O[1280][7][7], DATA_T W[4][1280]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_out_relu(DATA_T I[1280][7][7], DATA_T O[1280][7][7])
{
	int m, x, y;
	DATA_T ifm;
	DATA_T mv = 6;
	for (m = 0; m<1280; m++) {
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
void SW_global_average_pooling2d_1(DATA_T I[1280][7][7], DATA_T O[1280]) {
	int m, x, y;
	double avg;
	int div = 7 * 7;
	for (m = 0; m < 1280; m++){
		avg = 0;
		for (x = 0; x < 7; x++) {
			for (y = 0; y < 7; y++) {
				avg += I[m][x][y];
			}
		}
		O[m] = avg/div;
	}
}
void SW_Logits(DATA_T I[1280], DATA_T O[1000], DATA_T W[1000][1280], DATA_T B[1000])
{
    //Dense
	int m, c;
	DATA_T maximum;
	DATA_T denom=0;

	for(m=0; m<1000; m++){
        O[m] = 0;
		for (c = 0; c < 1280; c++){
			O[m] += W[m][c] * I[c];
        }
        O[m] += B[m];
        denom+=O[m];
    }

    //Softmax

    maximum=O[0]/denom;
    for (m = 0; m < 1000; m++){
		O[m]=O[m]/denom;
		if(maximum<O[m])
		    maximum=O[m];
    }

    for (m = 0; m < 1000; m++){
	    if(maximum!=O[m])
	        O[m]=0;
	    else
	        O[m]=1;
    }


}


//argv[1] = init_weight.txt, argv[2] = init_input.txt
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
	static DATA_T W27[4][144];
	static DATA_T W29[144][3][3];
	static DATA_T W30[4][144];
	static DATA_T W32[32][144][1][1];
	static DATA_T W33[4][32];
	static DATA_T W34[192][32][1][1];
	static DATA_T W35[4][192];
	static DATA_T W37[192][3][3];
	static DATA_T W38[4][192];
	static DATA_T W40[32][192][1][1];
	static DATA_T W41[4][32];
	static DATA_T W43[192][32][1][1];
	static DATA_T W44[4][192];
	static DATA_T W46[192][3][3];
	static DATA_T W47[4][192];
	static DATA_T W49[32][192][1][1];
	static DATA_T W50[4][32];
	static DATA_T W52[192][32][1][1];
	static DATA_T W53[4][192];
	static DATA_T W55[192][3][3];
	static DATA_T W56[4][192];
	static DATA_T W58[64][192][1][1];
	static DATA_T W59[4][64];
	static DATA_T W60[384][64][1][1];
	static DATA_T W61[4][384];
	static DATA_T W63[384][3][3];
	static DATA_T W64[4][384];
	static DATA_T W66[64][384][1][1];
	static DATA_T W67[4][64];
	static DATA_T W69[384][64][1][1];
	static DATA_T W70[4][384];
	static DATA_T W72[384][3][3];
	static DATA_T W73[4][384];
	static DATA_T W75[64][384][1][1];
	static DATA_T W76[4][64];
	static DATA_T W78[384][64][1][1];
	static DATA_T W79[4][384];
	static DATA_T W81[384][3][3];
	static DATA_T W82[4][384];
	static DATA_T W84[64][384][1][1];
	static DATA_T W85[4][64];
	static DATA_T W87[384][64][1][1];
	static DATA_T W88[4][384];
	static DATA_T W90[384][3][3];
	static DATA_T W91[4][384];
	static DATA_T W93[96][384][1][1];
	static DATA_T W94[4][96];
	static DATA_T W95[576][96][1][1];
	static DATA_T W96[4][576];
	static DATA_T W98[576][3][3];
	static DATA_T W99[4][576];
	static DATA_T W101[96][576][1][1];
	static DATA_T W102[4][96];
	static DATA_T W104[576][96][1][1];
	static DATA_T W105[4][576];
	static DATA_T W107[576][3][3];
	static DATA_T W108[4][576];
	static DATA_T W110[96][576][1][1];
	static DATA_T W111[4][96];
	static DATA_T W113[576][96][1][1];
	static DATA_T W114[4][576];
	static DATA_T W116[576][3][3];
	static DATA_T W117[4][576];
	static DATA_T W119[160][576][1][1];
	static DATA_T W120[4][160];
	static DATA_T W121[960][160][1][1];
	static DATA_T W122[4][960];
	static DATA_T W124[960][3][3];
	static DATA_T W125[4][960];
	static DATA_T W127[160][960][1][1];
	static DATA_T W128[4][160];
	static DATA_T W130[960][160][1][1];
	static DATA_T W131[4][960];
	static DATA_T W133[960][3][3];
	static DATA_T W134[4][960];
	static DATA_T W136[160][960][1][1];
	static DATA_T W137[4][160];
	static DATA_T W139[960][160][1][1];
	static DATA_T W140[4][960];
	static DATA_T W142[960][3][3];
	static DATA_T W143[4][960];
	static DATA_T W145[320][960][1][1];
	static DATA_T W146[4][320];
	static DATA_T W147[1280][320][1][1];
	static DATA_T W148[4][1280];
	static DATA_T B151[1000];
	static DATA_T W151[1000][1280];
	

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
	static DATA_T O27_SW[144][56][56];
	static DATA_T O28_SW[144][56][56];
	static DATA_T O29_SW[144][28][28];
	static DATA_T O30_SW[144][28][28];
	static DATA_T O31_SW[144][28][28];
	static DATA_T O32_SW[32][28][28];
	static DATA_T O33_SW[32][28][28];
	static DATA_T O34_SW[192][28][28];
	static DATA_T O35_SW[192][28][28];
	static DATA_T O36_SW[192][28][28];
	static DATA_T O37_SW[192][28][28];
	static DATA_T O38_SW[192][28][28];
	static DATA_T O39_SW[192][28][28];
	static DATA_T O40_SW[32][28][28];
	static DATA_T O41_SW[32][28][28];
	static DATA_T O42_SW[32][28][28];
	static DATA_T O43_SW[192][28][28];
	static DATA_T O44_SW[192][28][28];
	static DATA_T O45_SW[192][28][28];
	static DATA_T O46_SW[192][28][28];
	static DATA_T O47_SW[192][28][28];
	static DATA_T O48_SW[192][28][28];
	static DATA_T O49_SW[32][28][28];
	static DATA_T O50_SW[32][28][28];
	static DATA_T O51_SW[32][28][28];
	static DATA_T O52_SW[192][28][28];
	static DATA_T O53_SW[192][28][28];
	static DATA_T O54_SW[192][28][28];
	static DATA_T O55_SW[192][14][14];
	static DATA_T O56_SW[192][14][14];
	static DATA_T O57_SW[192][14][14];
	static DATA_T O58_SW[64][14][14];
	static DATA_T O59_SW[64][14][14];
	static DATA_T O60_SW[384][14][14];
	static DATA_T O61_SW[384][14][14];
	static DATA_T O62_SW[384][14][14];
	static DATA_T O63_SW[384][14][14];
	static DATA_T O64_SW[384][14][14];
	static DATA_T O65_SW[384][14][14];
	static DATA_T O66_SW[64][14][14];
	static DATA_T O67_SW[64][14][14];
	static DATA_T O68_SW[64][14][14];
	static DATA_T O69_SW[384][14][14];
	static DATA_T O70_SW[384][14][14];
	static DATA_T O71_SW[384][14][14];
	static DATA_T O72_SW[384][14][14];
	static DATA_T O73_SW[384][14][14];
	static DATA_T O74_SW[384][14][14];
	static DATA_T O75_SW[64][14][14];
	static DATA_T O76_SW[64][14][14];
	static DATA_T O77_SW[64][14][14];
	static DATA_T O78_SW[384][14][14];
	static DATA_T O79_SW[384][14][14];
	static DATA_T O80_SW[384][14][14];
	static DATA_T O81_SW[384][14][14];
	static DATA_T O82_SW[384][14][14];
	static DATA_T O83_SW[384][14][14];
	static DATA_T O84_SW[64][14][14];
	static DATA_T O85_SW[64][14][14];
	static DATA_T O86_SW[64][14][14];
	static DATA_T O87_SW[384][14][14];
	static DATA_T O88_SW[384][14][14];
	static DATA_T O89_SW[384][14][14];
	static DATA_T O90_SW[384][14][14];
	static DATA_T O91_SW[384][14][14];
	static DATA_T O92_SW[384][14][14];
	static DATA_T O93_SW[96][14][14];
	static DATA_T O94_SW[96][14][14];
	static DATA_T O95_SW[576][14][14];
	static DATA_T O96_SW[576][14][14];
	static DATA_T O97_SW[576][14][14];
	static DATA_T O98_SW[576][14][14];
	static DATA_T O99_SW[576][14][14];
	static DATA_T O100_SW[576][14][14];
	static DATA_T O101_SW[96][14][14];
	static DATA_T O102_SW[96][14][14];
	static DATA_T O103_SW[96][14][14];
	static DATA_T O104_SW[576][14][14];
	static DATA_T O105_SW[576][14][14];
	static DATA_T O106_SW[576][14][14];
	static DATA_T O107_SW[576][14][14];
	static DATA_T O108_SW[576][14][14];
	static DATA_T O109_SW[576][14][14];
	static DATA_T O110_SW[96][14][14];
	static DATA_T O111_SW[96][14][14];
	static DATA_T O112_SW[96][14][14];
	static DATA_T O113_SW[576][14][14];
	static DATA_T O114_SW[576][14][14];
	static DATA_T O115_SW[576][14][14];
	static DATA_T O116_SW[576][7][7];
	static DATA_T O117_SW[576][7][7];
	static DATA_T O118_SW[576][7][7];
	static DATA_T O119_SW[160][7][7];
	static DATA_T O120_SW[160][7][7];
	static DATA_T O121_SW[960][7][7];
	static DATA_T O122_SW[960][7][7];
	static DATA_T O123_SW[960][7][7];
	static DATA_T O124_SW[960][7][7];
	static DATA_T O125_SW[960][7][7];
	static DATA_T O126_SW[960][7][7];
	static DATA_T O127_SW[160][7][7];
	static DATA_T O128_SW[160][7][7];
	static DATA_T O129_SW[160][7][7];
	static DATA_T O130_SW[960][7][7];
	static DATA_T O131_SW[960][7][7];
	static DATA_T O132_SW[960][7][7];
	static DATA_T O133_SW[960][7][7];
	static DATA_T O134_SW[960][7][7];
	static DATA_T O135_SW[960][7][7];
	static DATA_T O136_SW[160][7][7];
	static DATA_T O137_SW[160][7][7];
	static DATA_T O138_SW[160][7][7];
	static DATA_T O139_SW[960][7][7];
	static DATA_T O140_SW[960][7][7];
	static DATA_T O141_SW[960][7][7];
	static DATA_T O142_SW[960][7][7];
	static DATA_T O143_SW[960][7][7];
	static DATA_T O144_SW[960][7][7];
	static DATA_T O145_SW[320][7][7];
	static DATA_T O146_SW[320][7][7];
	static DATA_T O147_SW[1280][7][7];
	static DATA_T O148_SW[1280][7][7];
	static DATA_T O149_SW[1280][7][7];
	static DATA_T O150_SW[1280];
	static DATA_T O151_SW[1000];
	

    FILE *w_stream = fopen(argv[1], "rb");
    if (w_stream == NULL) printf("weight file was not opened");
    FILE *i_stream = fopen(argv[2], "rb");
    if (i_stream == NULL) printf("input file was not opened");
    FILE *o_stream = fopen("Produced_code/mobilenetv2/Output/c_output.txt", "w");
    if (o_stream == NULL) printf("Output file was not opened");
    FILE *c_num = fopen("Produced_code/mobilenetv2/Output/c_output_num.txt", "w");
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

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 144 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W27[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
			for (j = 0; j < 144 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W29[j][m][k] = (DATA_T) trash;
			}
	}
}

/*
for (m = 0; m < 144 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B29[m] = (DATA_T) trash;
}
*/
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 144 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W30[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 144 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W32[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B32[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W33[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W34[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B34[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W35[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W37[j][m][k] = (DATA_T) trash;
			}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B37[m] = (DATA_T) trash;
}
*/
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W38[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 192 ; i++) {
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
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W41[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W43[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B43[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W44[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W46[j][m][k] = (DATA_T) trash;
			}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B46[m] = (DATA_T) trash;
}
*/
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W47[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 192 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W49[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B49[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 32 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W50[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W52[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B52[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W53[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W55[j][m][k] = (DATA_T) trash;
			}
	}
}

/*
for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B55[m] = (DATA_T) trash;
}
*/
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 192 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W56[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 192 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W58[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B58[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W59[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 384 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W60[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 384 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B60[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 384 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W61[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
			for (j = 0; j < 384 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W63[j][m][k] = (DATA_T) trash;
			}
	}
}

/*
for (m = 0; m < 384 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B63[m] = (DATA_T) trash;
}
*/
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 384 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W64[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W66[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B66[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W67[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 384 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W69[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 384 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B69[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 384 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W70[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
			for (j = 0; j < 384 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W72[j][m][k] = (DATA_T) trash;
			}
	}
}

/*
for (m = 0; m < 384 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B72[m] = (DATA_T) trash;
}
*/
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 384 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W73[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 384 ; i++) {
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

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W76[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 384 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W78[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 384 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B78[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 384 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W79[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
			for (j = 0; j < 384 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W81[j][m][k] = (DATA_T) trash;
			}
	}
}

/*
for (m = 0; m < 384 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B81[m] = (DATA_T) trash;
}
*/
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 384 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W82[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W84[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B84[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W85[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 384 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W87[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 384 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B87[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 384 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W88[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
			for (j = 0; j < 384 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W90[j][m][k] = (DATA_T) trash;
			}
	}
}

/*
for (m = 0; m < 384 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B90[m] = (DATA_T) trash;
}
*/
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 384 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W91[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 96 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W93[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 96 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B93[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 96 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W94[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 96 ; i++) {
			for (j = 0; j < 576 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W95[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 576 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B95[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 576 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W96[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
			for (j = 0; j < 576 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W98[j][m][k] = (DATA_T) trash;
			}
	}
}

/*
for (m = 0; m < 576 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B98[m] = (DATA_T) trash;
}
*/
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 576 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W99[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 576 ; i++) {
			for (j = 0; j < 96 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W101[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 96 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B101[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 96 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W102[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 96 ; i++) {
			for (j = 0; j < 576 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W104[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 576 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B104[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 576 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W105[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
			for (j = 0; j < 576 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W107[j][m][k] = (DATA_T) trash;
			}
	}
}

/*
for (m = 0; m < 576 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B107[m] = (DATA_T) trash;
}
*/
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 576 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W108[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 576 ; i++) {
			for (j = 0; j < 96 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W110[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 96 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B110[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 96 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W111[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 96 ; i++) {
			for (j = 0; j < 576 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W113[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 576 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B113[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 576 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W114[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
			for (j = 0; j < 576 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W116[j][m][k] = (DATA_T) trash;
			}
	}
}

/*
for (m = 0; m < 576 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B116[m] = (DATA_T) trash;
}
*/
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 576 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W117[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 576 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W119[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B119[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W120[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 960 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W121[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 960 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B121[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 960 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W122[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
			for (j = 0; j < 960 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W124[j][m][k] = (DATA_T) trash;
			}
	}
}

/*
for (m = 0; m < 960 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B124[m] = (DATA_T) trash;
}
*/
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 960 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W125[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 960 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W127[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B127[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W128[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 960 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W130[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 960 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B130[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 960 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W131[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
			for (j = 0; j < 960 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W133[j][m][k] = (DATA_T) trash;
			}
	}
}

/*
for (m = 0; m < 960 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B133[m] = (DATA_T) trash;
}
*/
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 960 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W134[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 960 ; i++) {
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

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 160 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W137[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 960 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W139[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 960 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B139[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 960 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W140[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
			for (j = 0; j < 960 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W142[j][m][k] = (DATA_T) trash;
			}
	}
}

/*
for (m = 0; m < 960 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B142[m] = (DATA_T) trash;
}
*/
	for (x = 0; x < 4; x++) {
    for (y = 0; y < 960 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W143[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 960 ; i++) {
			for (j = 0; j < 320 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W145[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 320 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B145[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 320 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W146[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 320 ; i++) {
			for (j = 0; j < 1280 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W147[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 1280 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B147[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1280 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W148[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1280 ; m++) {
	for (k = 0; k < 1000 ; k++) {
		fread(&trash, sizeof(int), 1, w_stream);
        W151[k][m] = (DATA_T) trash;
	}
}


for (m = 0; m < 1000 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B151[m] = (DATA_T) trash;
}

	
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
	printf("[C_verifier.cpp]Calculate BatchNormalization27\n\n");
	SW_block_3_expand_BN(O26_SW,O27_SW, W27);
	printf("[C_verifier.cpp]Calculate Relu28\n\n");
	SW_block_3_expand_relu(O27_SW,O28_SW);
	printf("[C_verifier.cpp]Calculate DepthwiseConv2D29\n\n");
	SW_block_3_depthwise(O28_SW,O29_SW,W29);
	printf("[C_verifier.cpp]Calculate BatchNormalization30\n\n");
	SW_block_3_depthwise_BN(O29_SW,O30_SW, W30);
	printf("[C_verifier.cpp]Calculate Relu31\n\n");
	SW_block_3_depthwise_relu(O30_SW,O31_SW);
	printf("[C_verifier.cpp]Calculate Conv2D32\n\n");
	SW_block_3_project(O31_SW,O32_SW,W32);
	printf("[C_verifier.cpp]Calculate BatchNormalization33\n\n");
	SW_block_3_project_BN(O32_SW,O33_SW, W33);
	printf("[C_verifier.cpp]Calculate Conv2D34\n\n");
	SW_block_4_expand(O33_SW,O34_SW,W34);
	printf("[C_verifier.cpp]Calculate BatchNormalization35\n\n");
	SW_block_4_expand_BN(O34_SW,O35_SW, W35);
	printf("[C_verifier.cpp]Calculate Relu36\n\n");
	SW_block_4_expand_relu(O35_SW,O36_SW);
	printf("[C_verifier.cpp]Calculate DepthwiseConv2D37\n\n");
	SW_block_4_depthwise(O36_SW,O37_SW,W37);
	printf("[C_verifier.cpp]Calculate BatchNormalization38\n\n");
	SW_block_4_depthwise_BN(O37_SW,O38_SW, W38);
	printf("[C_verifier.cpp]Calculate Relu39\n\n");
	SW_block_4_depthwise_relu(O38_SW,O39_SW);
	printf("[C_verifier.cpp]Calculate Conv2D40\n\n");
	SW_block_4_project(O39_SW,O40_SW,W40);
	printf("[C_verifier.cpp]Calculate BatchNormalization41\n\n");
	SW_block_4_project_BN(O40_SW,O41_SW, W41);
	printf("[C_verifier.cpp]Calculate Add42\n\n");
	SW_block_4_add(O33_SW,O41_SW,O42_SW);
	printf("[C_verifier.cpp]Calculate Conv2D43\n\n");
	SW_block_5_expand(O42_SW,O43_SW,W43);
	printf("[C_verifier.cpp]Calculate BatchNormalization44\n\n");
	SW_block_5_expand_BN(O43_SW,O44_SW, W44);
	printf("[C_verifier.cpp]Calculate Relu45\n\n");
	SW_block_5_expand_relu(O44_SW,O45_SW);
	printf("[C_verifier.cpp]Calculate DepthwiseConv2D46\n\n");
	SW_block_5_depthwise(O45_SW,O46_SW,W46);
	printf("[C_verifier.cpp]Calculate BatchNormalization47\n\n");
	SW_block_5_depthwise_BN(O46_SW,O47_SW, W47);
	printf("[C_verifier.cpp]Calculate Relu48\n\n");
	SW_block_5_depthwise_relu(O47_SW,O48_SW);
	printf("[C_verifier.cpp]Calculate Conv2D49\n\n");
	SW_block_5_project(O48_SW,O49_SW,W49);
	printf("[C_verifier.cpp]Calculate BatchNormalization50\n\n");
	SW_block_5_project_BN(O49_SW,O50_SW, W50);
	printf("[C_verifier.cpp]Calculate Add51\n\n");
	SW_block_5_add(O42_SW,O50_SW,O51_SW);
	printf("[C_verifier.cpp]Calculate Conv2D52\n\n");
	SW_block_6_expand(O51_SW,O52_SW,W52);
	printf("[C_verifier.cpp]Calculate BatchNormalization53\n\n");
	SW_block_6_expand_BN(O52_SW,O53_SW, W53);
	printf("[C_verifier.cpp]Calculate Relu54\n\n");
	SW_block_6_expand_relu(O53_SW,O54_SW);
	printf("[C_verifier.cpp]Calculate DepthwiseConv2D55\n\n");
	SW_block_6_depthwise(O54_SW,O55_SW,W55);
	printf("[C_verifier.cpp]Calculate BatchNormalization56\n\n");
	SW_block_6_depthwise_BN(O55_SW,O56_SW, W56);
	printf("[C_verifier.cpp]Calculate Relu57\n\n");
	SW_block_6_depthwise_relu(O56_SW,O57_SW);
	printf("[C_verifier.cpp]Calculate Conv2D58\n\n");
	SW_block_6_project(O57_SW,O58_SW,W58);
	printf("[C_verifier.cpp]Calculate BatchNormalization59\n\n");
	SW_block_6_project_BN(O58_SW,O59_SW, W59);
	printf("[C_verifier.cpp]Calculate Conv2D60\n\n");
	SW_block_7_expand(O59_SW,O60_SW,W60);
	printf("[C_verifier.cpp]Calculate BatchNormalization61\n\n");
	SW_block_7_expand_BN(O60_SW,O61_SW, W61);
	printf("[C_verifier.cpp]Calculate Relu62\n\n");
	SW_block_7_expand_relu(O61_SW,O62_SW);
	printf("[C_verifier.cpp]Calculate DepthwiseConv2D63\n\n");
	SW_block_7_depthwise(O62_SW,O63_SW,W63);
	printf("[C_verifier.cpp]Calculate BatchNormalization64\n\n");
	SW_block_7_depthwise_BN(O63_SW,O64_SW, W64);
	printf("[C_verifier.cpp]Calculate Relu65\n\n");
	SW_block_7_depthwise_relu(O64_SW,O65_SW);
	printf("[C_verifier.cpp]Calculate Conv2D66\n\n");
	SW_block_7_project(O65_SW,O66_SW,W66);
	printf("[C_verifier.cpp]Calculate BatchNormalization67\n\n");
	SW_block_7_project_BN(O66_SW,O67_SW, W67);
	printf("[C_verifier.cpp]Calculate Add68\n\n");
	SW_block_7_add(O59_SW,O67_SW,O68_SW);
	printf("[C_verifier.cpp]Calculate Conv2D69\n\n");
	SW_block_8_expand(O68_SW,O69_SW,W69);
	printf("[C_verifier.cpp]Calculate BatchNormalization70\n\n");
	SW_block_8_expand_BN(O69_SW,O70_SW, W70);
	printf("[C_verifier.cpp]Calculate Relu71\n\n");
	SW_block_8_expand_relu(O70_SW,O71_SW);
	printf("[C_verifier.cpp]Calculate DepthwiseConv2D72\n\n");
	SW_block_8_depthwise(O71_SW,O72_SW,W72);
	printf("[C_verifier.cpp]Calculate BatchNormalization73\n\n");
	SW_block_8_depthwise_BN(O72_SW,O73_SW, W73);
	printf("[C_verifier.cpp]Calculate Relu74\n\n");
	SW_block_8_depthwise_relu(O73_SW,O74_SW);
	printf("[C_verifier.cpp]Calculate Conv2D75\n\n");
	SW_block_8_project(O74_SW,O75_SW,W75);
	printf("[C_verifier.cpp]Calculate BatchNormalization76\n\n");
	SW_block_8_project_BN(O75_SW,O76_SW, W76);
	printf("[C_verifier.cpp]Calculate Add77\n\n");
	SW_block_8_add(O68_SW,O76_SW,O77_SW);
	printf("[C_verifier.cpp]Calculate Conv2D78\n\n");
	SW_block_9_expand(O77_SW,O78_SW,W78);
	printf("[C_verifier.cpp]Calculate BatchNormalization79\n\n");
	SW_block_9_expand_BN(O78_SW,O79_SW, W79);
	printf("[C_verifier.cpp]Calculate Relu80\n\n");
	SW_block_9_expand_relu(O79_SW,O80_SW);
	printf("[C_verifier.cpp]Calculate DepthwiseConv2D81\n\n");
	SW_block_9_depthwise(O80_SW,O81_SW,W81);
	printf("[C_verifier.cpp]Calculate BatchNormalization82\n\n");
	SW_block_9_depthwise_BN(O81_SW,O82_SW, W82);
	printf("[C_verifier.cpp]Calculate Relu83\n\n");
	SW_block_9_depthwise_relu(O82_SW,O83_SW);
	printf("[C_verifier.cpp]Calculate Conv2D84\n\n");
	SW_block_9_project(O83_SW,O84_SW,W84);
	printf("[C_verifier.cpp]Calculate BatchNormalization85\n\n");
	SW_block_9_project_BN(O84_SW,O85_SW, W85);
	printf("[C_verifier.cpp]Calculate Add86\n\n");
	SW_block_9_add(O77_SW,O85_SW,O86_SW);
	printf("[C_verifier.cpp]Calculate Conv2D87\n\n");
	SW_block_10_expand(O86_SW,O87_SW,W87);
	printf("[C_verifier.cpp]Calculate BatchNormalization88\n\n");
	SW_block_10_expand_BN(O87_SW,O88_SW, W88);
	printf("[C_verifier.cpp]Calculate Relu89\n\n");
	SW_block_10_expand_relu(O88_SW,O89_SW);
	printf("[C_verifier.cpp]Calculate DepthwiseConv2D90\n\n");
	SW_block_10_depthwise(O89_SW,O90_SW,W90);
	printf("[C_verifier.cpp]Calculate BatchNormalization91\n\n");
	SW_block_10_depthwise_BN(O90_SW,O91_SW, W91);
	printf("[C_verifier.cpp]Calculate Relu92\n\n");
	SW_block_10_depthwise_relu(O91_SW,O92_SW);
	printf("[C_verifier.cpp]Calculate Conv2D93\n\n");
	SW_block_10_project(O92_SW,O93_SW,W93);
	printf("[C_verifier.cpp]Calculate BatchNormalization94\n\n");
	SW_block_10_project_BN(O93_SW,O94_SW, W94);
	printf("[C_verifier.cpp]Calculate Conv2D95\n\n");
	SW_block_11_expand(O94_SW,O95_SW,W95);
	printf("[C_verifier.cpp]Calculate BatchNormalization96\n\n");
	SW_block_11_expand_BN(O95_SW,O96_SW, W96);
	printf("[C_verifier.cpp]Calculate Relu97\n\n");
	SW_block_11_expand_relu(O96_SW,O97_SW);
	printf("[C_verifier.cpp]Calculate DepthwiseConv2D98\n\n");
	SW_block_11_depthwise(O97_SW,O98_SW,W98);
	printf("[C_verifier.cpp]Calculate BatchNormalization99\n\n");
	SW_block_11_depthwise_BN(O98_SW,O99_SW, W99);
	printf("[C_verifier.cpp]Calculate Relu100\n\n");
	SW_block_11_depthwise_relu(O99_SW,O100_SW);
	printf("[C_verifier.cpp]Calculate Conv2D101\n\n");
	SW_block_11_project(O100_SW,O101_SW,W101);
	printf("[C_verifier.cpp]Calculate BatchNormalization102\n\n");
	SW_block_11_project_BN(O101_SW,O102_SW, W102);
	printf("[C_verifier.cpp]Calculate Add103\n\n");
	SW_block_11_add(O94_SW,O102_SW,O103_SW);
	printf("[C_verifier.cpp]Calculate Conv2D104\n\n");
	SW_block_12_expand(O103_SW,O104_SW,W104);
	printf("[C_verifier.cpp]Calculate BatchNormalization105\n\n");
	SW_block_12_expand_BN(O104_SW,O105_SW, W105);
	printf("[C_verifier.cpp]Calculate Relu106\n\n");
	SW_block_12_expand_relu(O105_SW,O106_SW);
	printf("[C_verifier.cpp]Calculate DepthwiseConv2D107\n\n");
	SW_block_12_depthwise(O106_SW,O107_SW,W107);
	printf("[C_verifier.cpp]Calculate BatchNormalization108\n\n");
	SW_block_12_depthwise_BN(O107_SW,O108_SW, W108);
	printf("[C_verifier.cpp]Calculate Relu109\n\n");
	SW_block_12_depthwise_relu(O108_SW,O109_SW);
	printf("[C_verifier.cpp]Calculate Conv2D110\n\n");
	SW_block_12_project(O109_SW,O110_SW,W110);
	printf("[C_verifier.cpp]Calculate BatchNormalization111\n\n");
	SW_block_12_project_BN(O110_SW,O111_SW, W111);
	printf("[C_verifier.cpp]Calculate Add112\n\n");
	SW_block_12_add(O103_SW,O111_SW,O112_SW);
	printf("[C_verifier.cpp]Calculate Conv2D113\n\n");
	SW_block_13_expand(O112_SW,O113_SW,W113);
	printf("[C_verifier.cpp]Calculate BatchNormalization114\n\n");
	SW_block_13_expand_BN(O113_SW,O114_SW, W114);
	printf("[C_verifier.cpp]Calculate Relu115\n\n");
	SW_block_13_expand_relu(O114_SW,O115_SW);
	printf("[C_verifier.cpp]Calculate DepthwiseConv2D116\n\n");
	SW_block_13_depthwise(O115_SW,O116_SW,W116);
	printf("[C_verifier.cpp]Calculate BatchNormalization117\n\n");
	SW_block_13_depthwise_BN(O116_SW,O117_SW, W117);
	printf("[C_verifier.cpp]Calculate Relu118\n\n");
	SW_block_13_depthwise_relu(O117_SW,O118_SW);
	printf("[C_verifier.cpp]Calculate Conv2D119\n\n");
	SW_block_13_project(O118_SW,O119_SW,W119);
	printf("[C_verifier.cpp]Calculate BatchNormalization120\n\n");
	SW_block_13_project_BN(O119_SW,O120_SW, W120);
	printf("[C_verifier.cpp]Calculate Conv2D121\n\n");
	SW_block_14_expand(O120_SW,O121_SW,W121);
	printf("[C_verifier.cpp]Calculate BatchNormalization122\n\n");
	SW_block_14_expand_BN(O121_SW,O122_SW, W122);
	printf("[C_verifier.cpp]Calculate Relu123\n\n");
	SW_block_14_expand_relu(O122_SW,O123_SW);
	printf("[C_verifier.cpp]Calculate DepthwiseConv2D124\n\n");
	SW_block_14_depthwise(O123_SW,O124_SW,W124);
	printf("[C_verifier.cpp]Calculate BatchNormalization125\n\n");
	SW_block_14_depthwise_BN(O124_SW,O125_SW, W125);
	printf("[C_verifier.cpp]Calculate Relu126\n\n");
	SW_block_14_depthwise_relu(O125_SW,O126_SW);
	printf("[C_verifier.cpp]Calculate Conv2D127\n\n");
	SW_block_14_project(O126_SW,O127_SW,W127);
	printf("[C_verifier.cpp]Calculate BatchNormalization128\n\n");
	SW_block_14_project_BN(O127_SW,O128_SW, W128);
	printf("[C_verifier.cpp]Calculate Add129\n\n");
	SW_block_14_add(O120_SW,O128_SW,O129_SW);
	printf("[C_verifier.cpp]Calculate Conv2D130\n\n");
	SW_block_15_expand(O129_SW,O130_SW,W130);
	printf("[C_verifier.cpp]Calculate BatchNormalization131\n\n");
	SW_block_15_expand_BN(O130_SW,O131_SW, W131);
	printf("[C_verifier.cpp]Calculate Relu132\n\n");
	SW_block_15_expand_relu(O131_SW,O132_SW);
	printf("[C_verifier.cpp]Calculate DepthwiseConv2D133\n\n");
	SW_block_15_depthwise(O132_SW,O133_SW,W133);
	printf("[C_verifier.cpp]Calculate BatchNormalization134\n\n");
	SW_block_15_depthwise_BN(O133_SW,O134_SW, W134);
	printf("[C_verifier.cpp]Calculate Relu135\n\n");
	SW_block_15_depthwise_relu(O134_SW,O135_SW);
	printf("[C_verifier.cpp]Calculate Conv2D136\n\n");
	SW_block_15_project(O135_SW,O136_SW,W136);
	printf("[C_verifier.cpp]Calculate BatchNormalization137\n\n");
	SW_block_15_project_BN(O136_SW,O137_SW, W137);
	printf("[C_verifier.cpp]Calculate Add138\n\n");
	SW_block_15_add(O129_SW,O137_SW,O138_SW);
	printf("[C_verifier.cpp]Calculate Conv2D139\n\n");
	SW_block_16_expand(O138_SW,O139_SW,W139);
	printf("[C_verifier.cpp]Calculate BatchNormalization140\n\n");
	SW_block_16_expand_BN(O139_SW,O140_SW, W140);
	printf("[C_verifier.cpp]Calculate Relu141\n\n");
	SW_block_16_expand_relu(O140_SW,O141_SW);
	printf("[C_verifier.cpp]Calculate DepthwiseConv2D142\n\n");
	SW_block_16_depthwise(O141_SW,O142_SW,W142);
	printf("[C_verifier.cpp]Calculate BatchNormalization143\n\n");
	SW_block_16_depthwise_BN(O142_SW,O143_SW, W143);
	printf("[C_verifier.cpp]Calculate Relu144\n\n");
	SW_block_16_depthwise_relu(O143_SW,O144_SW);
	printf("[C_verifier.cpp]Calculate Conv2D145\n\n");
	SW_block_16_project(O144_SW,O145_SW,W145);
	printf("[C_verifier.cpp]Calculate BatchNormalization146\n\n");
	SW_block_16_project_BN(O145_SW,O146_SW, W146);
	printf("[C_verifier.cpp]Calculate Conv2D147\n\n");
	SW_Conv_1(O146_SW,O147_SW,W147);
	printf("[C_verifier.cpp]Calculate BatchNormalization148\n\n");
	SW_Conv_1_bn(O147_SW,O148_SW, W148);
	printf("[C_verifier.cpp]Calculate Relu149\n\n");
	SW_out_relu(O148_SW,O149_SW);
	printf("[C_verifier.cpp]Calculate GlobalAveragePooling2D150\n\n");
	SW_global_average_pooling2d_1(O149_SW,O150_SW);
	printf("[C_verifier.cpp]Calculate Dense151\n\n");
	SW_Logits(O150_SW,O151_SW,W151,B151);
	

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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 144 ; y++) {
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


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 144 ; y++) {
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


fprintf(o_stream,"%s","DepthwiseConv2D : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 144 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 144 ; y++) {
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


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 144 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
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
		for(y = 0; y < 192 ; y++) {
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
		for(y = 0; y < 192 ; y++) {
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
		for(y = 0; y < 192 ; y++) {
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


fprintf(o_stream,"%s","DepthwiseConv2D : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O37_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O37_SW[y][k][x]);
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


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
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


fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
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


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
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


fprintf(o_stream,"%s","DepthwiseConv2D : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
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
		for(y = 0; y < 192 ; y++) {
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


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
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


fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
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
		for(y = 0; y < 192 ; y++) {
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


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
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


fprintf(o_stream,"%s","DepthwiseConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
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


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
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


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
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


fprintf(o_stream,"%s","DepthwiseConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
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


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
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


fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
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


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
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


fprintf(o_stream,"%s","DepthwiseConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
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


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O74_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O74_SW[y][k][x]);
		}
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
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O75_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O75_SW[y][k][x]);
		}
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
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O76_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O76_SW[y][k][x]);
		}
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
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O77_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O77_SW[y][k][x]);
		}
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
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O78_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O78_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O79_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O79_SW[y][k][x]);
		}
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
		for(y = 0; y < 384 ; y++) {
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


fprintf(o_stream,"%s","DepthwiseConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
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


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
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


fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
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
		for(y = 0; y < 384 ; y++) {
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
		for(y = 0; y < 384 ; y++) {
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


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
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


fprintf(o_stream,"%s","DepthwiseConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
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


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 96 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 96 ; y++) {
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
		for(y = 0; y < 576 ; y++) {
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
		for(y = 0; y < 576 ; y++) {
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


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 576 ; y++) {
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


fprintf(o_stream,"%s","DepthwiseConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 576 ; y++) {
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
		for(y = 0; y < 576 ; y++) {
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


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 576 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 96 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 96 ; y++) {
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


fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 96 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 576 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 576 ; y++) {
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


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 576 ; y++) {
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


fprintf(o_stream,"%s","DepthwiseConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 576 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 576 ; y++) {
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


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 576 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 96 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 96 ; y++) {
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


fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 96 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 576 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 576 ; y++) {
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


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 576 ; y++) {
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


fprintf(o_stream,"%s","DepthwiseConv2D : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 576 ; y++) {
			fprintf(o_stream,"%.6f ",O116_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O116_SW[y][k][x]);
		}
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
		for(y = 0; y < 576 ; y++) {
			fprintf(o_stream,"%.6f ",O117_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O117_SW[y][k][x]);
		}
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
		for(y = 0; y < 576 ; y++) {
			fprintf(o_stream,"%.6f ",O118_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O118_SW[y][k][x]);
		}
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
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O119_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O119_SW[y][k][x]);
		}
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
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O120_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O120_SW[y][k][x]);
		}
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
		for(y = 0; y < 960 ; y++) {
			fprintf(o_stream,"%.6f ",O121_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O121_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O122_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O122_SW[y][k][x]);
		}
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
		for(y = 0; y < 960 ; y++) {
			fprintf(o_stream,"%.6f ",O123_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O123_SW[y][k][x]);
		}
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
		for(y = 0; y < 960 ; y++) {
			fprintf(o_stream,"%.6f ",O124_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O124_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O125_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O125_SW[y][k][x]);
		}
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
		for(y = 0; y < 960 ; y++) {
			fprintf(o_stream,"%.6f ",O126_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O126_SW[y][k][x]);
		}
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
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O127_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O127_SW[y][k][x]);
		}
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
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O128_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O128_SW[y][k][x]);
		}
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
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O129_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O129_SW[y][k][x]);
		}
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
		for(y = 0; y < 960 ; y++) {
			fprintf(o_stream,"%.6f ",O130_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O130_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O131_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O131_SW[y][k][x]);
		}
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
		for(y = 0; y < 960 ; y++) {
			fprintf(o_stream,"%.6f ",O132_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O132_SW[y][k][x]);
		}
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
		for(y = 0; y < 960 ; y++) {
			fprintf(o_stream,"%.6f ",O133_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O133_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O134_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O134_SW[y][k][x]);
		}
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
		for(y = 0; y < 960 ; y++) {
			fprintf(o_stream,"%.6f ",O135_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O135_SW[y][k][x]);
		}
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
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O136_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O136_SW[y][k][x]);
		}
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
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O137_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O137_SW[y][k][x]);
		}
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
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O138_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O138_SW[y][k][x]);
		}
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
		for(y = 0; y < 960 ; y++) {
			fprintf(o_stream,"%.6f ",O139_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O139_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O140_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O140_SW[y][k][x]);
		}
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
		for(y = 0; y < 960 ; y++) {
			fprintf(o_stream,"%.6f ",O141_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O141_SW[y][k][x]);
		}
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
		for(y = 0; y < 960 ; y++) {
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
		for(y = 0; y < 960 ; y++) {
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


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 960 ; y++) {
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
		for(y = 0; y < 320 ; y++) {
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
		for(y = 0; y < 320 ; y++) {
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


fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1280 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1280 ; y++) {
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


fprintf(o_stream,"%s","ReLU : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1280 ; y++) {
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


fprintf(o_stream,"%s","GlobalAveragePooling2D : [[");
for (k = 0; k <  1280 ; k++) {
	fprintf(o_stream,"%.6f ",O150_SW[k]);
	fprintf(c_num,"%.6f ",O150_SW[k]);
}
fprintf(o_stream,"%s","]]\n\n");


fprintf(o_stream,"%s","Dense : [[");
for (k = 0; k <  1000 ; k++) {
	fprintf(o_stream,"%.6f ",O151_SW[k]);
	fprintf(c_num,"%.6f ",O151_SW[k]);
}
fprintf(o_stream,"%s","]]\n\n");





    fclose(w_stream);
    fclose(i_stream);
    fclose(o_stream);
    fclose(c_num);

    return 0;
}