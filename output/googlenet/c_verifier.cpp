#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <math.h>

using namespace std;

typedef float DATA_T;

// define functions of each layer
void SW_block1_conv1(DATA_T I[3][224][224], DATA_T O[64][112][112], DATA_T W[64][3][7][7], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<64; m++) {
		for (x = 0; x<112; x++) {
			for (y = 0; y<112; y++) {
				ofm = B[m];
				for (k = 0; k<3; k++) {
					for (i = 0; i<7; i++) {
						for (j = 0; j<7; j++) {
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

void SW_block1_pool(DATA_T I[64][112][112], DATA_T O[64][56][56])
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
void SW_block2_conv1(DATA_T I[64][56][56], DATA_T O[64][56][56], DATA_T W[64][64][1][1], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(56 - 1) - 56 + 1)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				ofm = B[m];
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

void SW_block2_conv2(DATA_T I[64][56][56], DATA_T O[192][56][56], DATA_T W[192][64][3][3], DATA_T B[192]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(56 - 1) - 56 + 3)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				ofm = B[m];
				for (k = 0; k<64; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
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

void SW_block2_pool(DATA_T I[192][56][56], DATA_T O[192][28][28])
{
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<192; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
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
void SW_block3_conv1(DATA_T I[192][28][28], DATA_T O[64][28][28], DATA_T W[64][192][1][1], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 1)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = B[m];
				for (k = 0; k<192; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block3_conv2(DATA_T I[192][28][28], DATA_T O[96][28][28], DATA_T W[96][192][1][1], DATA_T B[96]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 1)/2;
	for (m = 0; m<96; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = B[m];
				for (k = 0; k<192; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block3_conv3(DATA_T I[192][28][28], DATA_T O[16][28][28], DATA_T W[16][192][1][1], DATA_T B[16]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 1)/2;
	for (m = 0; m<16; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = B[m];
				for (k = 0; k<192; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block3_pool(DATA_T I[192][28][28], DATA_T O[192][28][28])
{
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<192; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				max = I[m][x*1][y*1];
				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {
						if (I[m][x*1 + i][y*1 + j] > max) {
							max = I[m][x*1 + i][y*1 + j];
						}
					}
				}
				O[m][x][y] = max;
			}
		}
	}
}
void SW_block3_conv4(DATA_T I[96][28][28], DATA_T O[128][28][28], DATA_T W[128][96][3][3], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 3)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = B[m];
				for (k = 0; k<96; k++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block3_conv5(DATA_T I[16][28][28], DATA_T O[32][28][28], DATA_T W[32][16][5][5], DATA_T B[32]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 5)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = B[m];
				for (k = 0; k<16; k++) {
					for (i = 0; i<5; i++) {
						for (j = 0; j<5; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block3_conv6(DATA_T I[192][28][28], DATA_T O[32][28][28], DATA_T W[32][192][1][1], DATA_T B[32]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = B[m];
				for (k = 0; k<192; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_concat3(DATA_T I1[64][28][28], DATA_T I2[128][28][28], DATA_T I3[32][28][28], DATA_T I4[32][28][28], DATA_T O[256][28][28]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 28; x++) {
		for(y = 0; y < 28; y++) {
			ch=0;
			for(k = ch; k < ch+64; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=64;
			for(k = ch; k < ch+128; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
			ch+=128;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I3[k-ch][x][y];
			}
			ch+=32;
			for(k = ch; k < ch+32; k++){
				O[k][x][y] = I4[k-ch][x][y];
			}
		}
	}

}
void SW_block4_conv1(DATA_T I[256][28][28], DATA_T O[128][28][28], DATA_T W[128][256][1][1], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = B[m];
				for (k = 0; k<256; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block4_conv2(DATA_T I[256][28][28], DATA_T O[128][28][28], DATA_T W[128][256][1][1], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = B[m];
				for (k = 0; k<256; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block4_conv3(DATA_T I[256][28][28], DATA_T O[32][28][28], DATA_T W[32][256][1][1], DATA_T B[32]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = B[m];
				for (k = 0; k<256; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block4_pool(DATA_T I[256][28][28], DATA_T O[256][28][28])
{
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<256; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				max = I[m][x*1][y*1];
				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {
						if (I[m][x*1 + i][y*1 + j] > max) {
							max = I[m][x*1 + i][y*1 + j];
						}
					}
				}
				O[m][x][y] = max;
			}
		}
	}
}
void SW_block4_conv4(DATA_T I[128][28][28], DATA_T O[192][28][28], DATA_T W[192][128][3][3], DATA_T B[192]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 3)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = B[m];
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block4_conv5(DATA_T I[32][28][28], DATA_T O[96][28][28], DATA_T W[96][32][5][5], DATA_T B[96]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 5)/2;
	for (m = 0; m<96; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = B[m];
				for (k = 0; k<32; k++) {
					for (i = 0; i<5; i++) {
						for (j = 0; j<5; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block4_conv6(DATA_T I[256][28][28], DATA_T O[64][28][28], DATA_T W[64][256][1][1], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(28 - 1) - 28 + 1)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = B[m];
				for (k = 0; k<256; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_concat4(DATA_T I1[128][28][28], DATA_T I2[192][28][28], DATA_T I3[96][28][28], DATA_T I4[64][28][28], DATA_T O[480][28][28]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 28; x++) {
		for(y = 0; y < 28; y++) {
			ch=0;
			for(k = ch; k < ch+128; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=128;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
			ch+=192;
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
void SW_block5_pool(DATA_T I[480][28][28], DATA_T O[480][14][14])
{
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<480; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
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
void SW_block6_conv1(DATA_T I[480][14][14], DATA_T O[192][14][14], DATA_T W[192][480][1][1], DATA_T B[192]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = B[m];
				for (k = 0; k<480; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block6_conv2(DATA_T I[480][14][14], DATA_T O[96][14][14], DATA_T W[96][480][1][1], DATA_T B[96]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<96; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = B[m];
				for (k = 0; k<480; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block6_conv3(DATA_T I[480][14][14], DATA_T O[16][14][14], DATA_T W[16][480][1][1], DATA_T B[16]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<16; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = B[m];
				for (k = 0; k<480; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block6_pool(DATA_T I[480][14][14], DATA_T O[480][14][14])
{
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<480; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				max = I[m][x*1][y*1];
				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {
						if (I[m][x*1 + i][y*1 + j] > max) {
							max = I[m][x*1 + i][y*1 + j];
						}
					}
				}
				O[m][x][y] = max;
			}
		}
	}
}
void SW_block6_conv4(DATA_T I[96][14][14], DATA_T O[208][14][14], DATA_T W[208][96][3][3], DATA_T B[208]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<208; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = B[m];
				for (k = 0; k<96; k++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block6_conv5(DATA_T I[16][14][14], DATA_T O[48][14][14], DATA_T W[48][16][5][5], DATA_T B[48]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 5)/2;
	for (m = 0; m<48; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = B[m];
				for (k = 0; k<16; k++) {
					for (i = 0; i<5; i++) {
						for (j = 0; j<5; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block6_conv6(DATA_T I[480][14][14], DATA_T O[64][14][14], DATA_T W[64][480][1][1], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = B[m];
				for (k = 0; k<480; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_concat6(DATA_T I1[192][14][14], DATA_T I2[208][14][14], DATA_T I3[48][14][14], DATA_T I4[64][14][14], DATA_T O[512][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+192; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=192;
			for(k = ch; k < ch+208; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
			ch+=208;
			for(k = ch; k < ch+48; k++){
				O[k][x][y] = I3[k-ch][x][y];
			}
			ch+=48;
			for(k = ch; k < ch+64; k++){
				O[k][x][y] = I4[k-ch][x][y];
			}
		}
	}

}
void SW_block7_conv1(DATA_T I[512][14][14], DATA_T O[160][14][14], DATA_T W[160][512][1][1], DATA_T B[160]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = B[m];
				for (k = 0; k<512; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block7_conv2(DATA_T I[512][14][14], DATA_T O[112][14][14], DATA_T W[112][512][1][1], DATA_T B[112]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<112; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = B[m];
				for (k = 0; k<512; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block7_conv3(DATA_T I[512][14][14], DATA_T O[24][14][14], DATA_T W[24][512][1][1], DATA_T B[24]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<24; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = B[m];
				for (k = 0; k<512; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block7_pool(DATA_T I[512][14][14], DATA_T O[512][14][14])
{
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<512; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				max = I[m][x*1][y*1];
				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {
						if (I[m][x*1 + i][y*1 + j] > max) {
							max = I[m][x*1 + i][y*1 + j];
						}
					}
				}
				O[m][x][y] = max;
			}
		}
	}
}
void SW_block7_conv4(DATA_T I[112][14][14], DATA_T O[224][14][14], DATA_T W[224][112][3][3], DATA_T B[224]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<224; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = B[m];
				for (k = 0; k<112; k++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block7_conv5(DATA_T I[24][14][14], DATA_T O[64][14][14], DATA_T W[64][24][5][5], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 5)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = B[m];
				for (k = 0; k<24; k++) {
					for (i = 0; i<5; i++) {
						for (j = 0; j<5; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block7_conv6(DATA_T I[512][14][14], DATA_T O[64][14][14], DATA_T W[64][512][1][1], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = B[m];
				for (k = 0; k<512; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_concat7(DATA_T I1[160][14][14], DATA_T I2[224][14][14], DATA_T I3[64][14][14], DATA_T I4[64][14][14], DATA_T O[512][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+160; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=160;
			for(k = ch; k < ch+224; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
			ch+=224;
			for(k = ch; k < ch+64; k++){
				O[k][x][y] = I3[k-ch][x][y];
			}
			ch+=64;
			for(k = ch; k < ch+64; k++){
				O[k][x][y] = I4[k-ch][x][y];
			}
		}
	}

}
void SW_block8_conv1(DATA_T I[512][14][14], DATA_T O[128][14][14], DATA_T W[128][512][1][1], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = B[m];
				for (k = 0; k<512; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block8_conv2(DATA_T I[512][14][14], DATA_T O[128][14][14], DATA_T W[128][512][1][1], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = B[m];
				for (k = 0; k<512; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block8_conv3(DATA_T I[512][14][14], DATA_T O[24][14][14], DATA_T W[24][512][1][1], DATA_T B[24]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<24; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = B[m];
				for (k = 0; k<512; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block8_pool(DATA_T I[512][14][14], DATA_T O[512][14][14])
{
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<512; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				max = I[m][x*1][y*1];
				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {
						if (I[m][x*1 + i][y*1 + j] > max) {
							max = I[m][x*1 + i][y*1 + j];
						}
					}
				}
				O[m][x][y] = max;
			}
		}
	}
}
void SW_block8_conv4(DATA_T I[128][14][14], DATA_T O[256][14][14], DATA_T W[256][128][3][3], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = B[m];
				for (k = 0; k<128; k++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block8_conv5(DATA_T I[24][14][14], DATA_T O[64][14][14], DATA_T W[64][24][5][5], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 5)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = B[m];
				for (k = 0; k<24; k++) {
					for (i = 0; i<5; i++) {
						for (j = 0; j<5; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block8_conv6(DATA_T I[512][14][14], DATA_T O[64][14][14], DATA_T W[64][512][1][1], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = B[m];
				for (k = 0; k<512; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_concat8(DATA_T I1[128][14][14], DATA_T I2[256][14][14], DATA_T I3[64][14][14], DATA_T I4[64][14][14], DATA_T O[512][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+128; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=128;
			for(k = ch; k < ch+256; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
			ch+=256;
			for(k = ch; k < ch+64; k++){
				O[k][x][y] = I3[k-ch][x][y];
			}
			ch+=64;
			for(k = ch; k < ch+64; k++){
				O[k][x][y] = I4[k-ch][x][y];
			}
		}
	}

}
void SW_block9_conv1(DATA_T I[512][14][14], DATA_T O[112][14][14], DATA_T W[112][512][1][1], DATA_T B[112]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<112; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = B[m];
				for (k = 0; k<512; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block9_conv2(DATA_T I[512][14][14], DATA_T O[144][14][14], DATA_T W[144][512][1][1], DATA_T B[144]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<144; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = B[m];
				for (k = 0; k<512; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block9_conv3(DATA_T I[512][14][14], DATA_T O[32][14][14], DATA_T W[32][512][1][1], DATA_T B[32]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = B[m];
				for (k = 0; k<512; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block9_pool(DATA_T I[512][14][14], DATA_T O[512][14][14])
{
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<512; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				max = I[m][x*1][y*1];
				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {
						if (I[m][x*1 + i][y*1 + j] > max) {
							max = I[m][x*1 + i][y*1 + j];
						}
					}
				}
				O[m][x][y] = max;
			}
		}
	}
}
void SW_block9_conv4(DATA_T I[144][14][14], DATA_T O[288][14][14], DATA_T W[288][144][3][3], DATA_T B[288]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<288; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = B[m];
				for (k = 0; k<144; k++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block9_conv5(DATA_T I[32][14][14], DATA_T O[64][14][14], DATA_T W[64][32][5][5], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 5)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = B[m];
				for (k = 0; k<32; k++) {
					for (i = 0; i<5; i++) {
						for (j = 0; j<5; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block9_conv6(DATA_T I[512][14][14], DATA_T O[64][14][14], DATA_T W[64][512][1][1], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = B[m];
				for (k = 0; k<512; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_concat9(DATA_T I1[112][14][14], DATA_T I2[288][14][14], DATA_T I3[64][14][14], DATA_T I4[64][14][14], DATA_T O[528][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+112; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=112;
			for(k = ch; k < ch+288; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
			ch+=288;
			for(k = ch; k < ch+64; k++){
				O[k][x][y] = I3[k-ch][x][y];
			}
			ch+=64;
			for(k = ch; k < ch+64; k++){
				O[k][x][y] = I4[k-ch][x][y];
			}
		}
	}

}
void SW_block10_conv1(DATA_T I[528][14][14], DATA_T O[256][14][14], DATA_T W[256][528][1][1], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = B[m];
				for (k = 0; k<528; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block10_conv2(DATA_T I[528][14][14], DATA_T O[160][14][14], DATA_T W[160][528][1][1], DATA_T B[160]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = B[m];
				for (k = 0; k<528; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block10_conv3(DATA_T I[528][14][14], DATA_T O[32][14][14], DATA_T W[32][528][1][1], DATA_T B[32]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = B[m];
				for (k = 0; k<528; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block10_pool(DATA_T I[528][14][14], DATA_T O[528][14][14])
{
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<528; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				max = I[m][x*1][y*1];
				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {
						if (I[m][x*1 + i][y*1 + j] > max) {
							max = I[m][x*1 + i][y*1 + j];
						}
					}
				}
				O[m][x][y] = max;
			}
		}
	}
}
void SW_block10_conv4(DATA_T I[160][14][14], DATA_T O[320][14][14], DATA_T W[320][160][3][3], DATA_T B[320]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<320; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = B[m];
				for (k = 0; k<160; k++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block10_conv5(DATA_T I[32][14][14], DATA_T O[128][14][14], DATA_T W[128][32][5][5], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 5)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = B[m];
				for (k = 0; k<32; k++) {
					for (i = 0; i<5; i++) {
						for (j = 0; j<5; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block10_conv6(DATA_T I[528][14][14], DATA_T O[128][14][14], DATA_T W[128][528][1][1], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(14 - 1) - 14 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = B[m];
				for (k = 0; k<528; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_concat10(DATA_T I1[256][14][14], DATA_T I2[320][14][14], DATA_T I3[128][14][14], DATA_T I4[128][14][14], DATA_T O[832][14][14]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 14; x++) {
		for(y = 0; y < 14; y++) {
			ch=0;
			for(k = ch; k < ch+256; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=256;
			for(k = ch; k < ch+320; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
			ch+=320;
			for(k = ch; k < ch+128; k++){
				O[k][x][y] = I3[k-ch][x][y];
			}
			ch+=128;
			for(k = ch; k < ch+128; k++){
				O[k][x][y] = I4[k-ch][x][y];
			}
		}
	}

}
void SW_block11_pool(DATA_T I[832][14][14], DATA_T O[832][7][7])
{
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<832; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
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
void SW_block12_conv1(DATA_T I[832][7][7], DATA_T O[256][7][7], DATA_T W[256][832][1][1], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 1)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = B[m];
				for (k = 0; k<832; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block12_conv2(DATA_T I[832][7][7], DATA_T O[160][7][7], DATA_T W[160][832][1][1], DATA_T B[160]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 1)/2;
	for (m = 0; m<160; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = B[m];
				for (k = 0; k<832; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block12_conv3(DATA_T I[832][7][7], DATA_T O[32][7][7], DATA_T W[32][832][1][1], DATA_T B[32]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = B[m];
				for (k = 0; k<832; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block12_pool(DATA_T I[832][7][7], DATA_T O[832][7][7])
{
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<832; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				max = I[m][x*1][y*1];
				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {
						if (I[m][x*1 + i][y*1 + j] > max) {
							max = I[m][x*1 + i][y*1 + j];
						}
					}
				}
				O[m][x][y] = max;
			}
		}
	}
}
void SW_block12_conv4(DATA_T I[160][7][7], DATA_T O[320][7][7], DATA_T W[320][160][3][3], DATA_T B[320]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<320; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = B[m];
				for (k = 0; k<160; k++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block12_conv5(DATA_T I[32][7][7], DATA_T O[128][7][7], DATA_T W[128][32][5][5], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 5)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = B[m];
				for (k = 0; k<32; k++) {
					for (i = 0; i<5; i++) {
						for (j = 0; j<5; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block12_conv6(DATA_T I[832][7][7], DATA_T O[128][7][7], DATA_T W[128][832][1][1], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = B[m];
				for (k = 0; k<832; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_concat12(DATA_T I1[256][7][7], DATA_T I2[320][7][7], DATA_T I3[128][7][7], DATA_T I4[128][7][7], DATA_T O[832][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+256; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=256;
			for(k = ch; k < ch+320; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
			ch+=320;
			for(k = ch; k < ch+128; k++){
				O[k][x][y] = I3[k-ch][x][y];
			}
			ch+=128;
			for(k = ch; k < ch+128; k++){
				O[k][x][y] = I4[k-ch][x][y];
			}
		}
	}

}
void SW_block13_conv1(DATA_T I[832][7][7], DATA_T O[384][7][7], DATA_T W[384][832][1][1], DATA_T B[384]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 1)/2;
	for (m = 0; m<384; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = B[m];
				for (k = 0; k<832; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block13_conv2(DATA_T I[832][7][7], DATA_T O[192][7][7], DATA_T W[192][832][1][1], DATA_T B[192]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = B[m];
				for (k = 0; k<832; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block13_conv3(DATA_T I[832][7][7], DATA_T O[48][7][7], DATA_T W[48][832][1][1], DATA_T B[48]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 1)/2;
	for (m = 0; m<48; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = B[m];
				for (k = 0; k<832; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block13_pool(DATA_T I[832][7][7], DATA_T O[832][7][7])
{
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<832; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				max = I[m][x*1][y*1];
				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {
						if (I[m][x*1 + i][y*1 + j] > max) {
							max = I[m][x*1 + i][y*1 + j];
						}
					}
				}
				O[m][x][y] = max;
			}
		}
	}
}
void SW_block13_conv4(DATA_T I[192][7][7], DATA_T O[384][7][7], DATA_T W[384][192][3][3], DATA_T B[384]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<384; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = B[m];
				for (k = 0; k<192; k++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block13_conv5(DATA_T I[48][7][7], DATA_T O[128][7][7], DATA_T W[128][48][5][5], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 5)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = B[m];
				for (k = 0; k<48; k++) {
					for (i = 0; i<5; i++) {
						for (j = 0; j<5; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block13_conv6(DATA_T I[832][7][7], DATA_T O[128][7][7], DATA_T W[128][832][1][1], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(7 - 1) - 7 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = B[m];
				for (k = 0; k<832; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_concat13(DATA_T I1[384][7][7], DATA_T I2[384][7][7], DATA_T I3[128][7][7], DATA_T I4[128][7][7], DATA_T O[1024][7][7]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 7; x++) {
		for(y = 0; y < 7; y++) {
			ch=0;
			for(k = ch; k < ch+384; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=384;
			for(k = ch; k < ch+384; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
			ch+=384;
			for(k = ch; k < ch+128; k++){
				O[k][x][y] = I3[k-ch][x][y];
			}
			ch+=128;
			for(k = ch; k < ch+128; k++){
				O[k][x][y] = I4[k-ch][x][y];
			}
		}
	}

}
void SW_avg_pool1(DATA_T I[1024][7][7], DATA_T O[1024][1][1])
{
	int m, x, y, i, j;
	DATA_T sum;
	int p = (1*(1-1) - 7 + 6)/2;
    DATA_T div;
	for (m = 0; m<1024; m++) {
		for (x = 0; x<1; x++) {
			for (y = 0; y<1; y++) {
                sum = (DATA_T)0;
                if(x!=0 && x!=1-1 && y!=0 && y!=1-1)
                    div = 9;
                else if((x==0 || x==1-1) && (y==0 || y==1-1))
                    div = 4;
                else
                    div = 6;
				for (i = 0; i<6; i++) {
					for (j = 0; j<6; j++) {

						if (x + i < 7 + p && y + j < 7 + p && x + i -p >= 0 && y + j -p >= 0)
						    sum += I[m][x*1 + i -p][y*1 + j -p];
					}
				}
				O[m][x][y] = sum/div;
			}
		}
	}
}
void SW_flatten(DATA_T I[1024][1][1], DATA_T O[1024]){
	int i, j, x, y;
	i = 0;
	 for(x=0; x<1;x++)
		for(y=0; y<1; y++)
			for (j=0; j<1024; j++) {
				O[i] = I[j][x][y];
				i++;
			}
}

void SW_predictions(DATA_T I[1024], DATA_T O[1000], DATA_T W[1000][1024])
{
    //Dense
	int m, c;
	DATA_T denom=0;
	DATA_T maximum;
	for(m=0; m<1000; m++){
        O[m] = 0;
		for (c = 0; c < 1024; c++){
			O[m] += W[m][c] * I[c];
         }
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

    // declare array variables of input, weight and bias (static variables)
    static DATA_T I[3][224][224];
	static DATA_T W1[64][3][7][7];
	static DATA_T B1[64];
	static DATA_T W3[64][64][1][1];
	static DATA_T B3[64];
	static DATA_T W4[192][64][3][3];
	static DATA_T B4[192];
	static DATA_T W6[64][192][1][1];
	static DATA_T B6[64];
	static DATA_T W7[96][192][1][1];
	static DATA_T B7[96];
	static DATA_T W8[16][192][1][1];
	static DATA_T B8[16];
	static DATA_T W10[128][96][3][3];
	static DATA_T B10[128];
	static DATA_T W11[32][16][5][5];
	static DATA_T B11[32];
	static DATA_T W12[32][192][1][1];
	static DATA_T B12[32];
	static DATA_T W14[128][256][1][1];
	static DATA_T B14[128];
	static DATA_T W15[128][256][1][1];
	static DATA_T B15[128];
	static DATA_T W16[32][256][1][1];
	static DATA_T B16[32];
	static DATA_T W18[192][128][3][3];
	static DATA_T B18[192];
	static DATA_T W19[96][32][5][5];
	static DATA_T B19[96];
	static DATA_T W20[64][256][1][1];
	static DATA_T B20[64];
	static DATA_T W23[192][480][1][1];
	static DATA_T B23[192];
	static DATA_T W24[96][480][1][1];
	static DATA_T B24[96];
	static DATA_T W25[16][480][1][1];
	static DATA_T B25[16];
	static DATA_T W27[208][96][3][3];
	static DATA_T B27[208];
	static DATA_T W28[48][16][5][5];
	static DATA_T B28[48];
	static DATA_T W29[64][480][1][1];
	static DATA_T B29[64];
	static DATA_T W31[160][512][1][1];
	static DATA_T B31[160];
	static DATA_T W32[112][512][1][1];
	static DATA_T B32[112];
	static DATA_T W33[24][512][1][1];
	static DATA_T B33[24];
	static DATA_T W35[224][112][3][3];
	static DATA_T B35[224];
	static DATA_T W36[64][24][5][5];
	static DATA_T B36[64];
	static DATA_T W37[64][512][1][1];
	static DATA_T B37[64];
	static DATA_T W39[128][512][1][1];
	static DATA_T B39[128];
	static DATA_T W40[128][512][1][1];
	static DATA_T B40[128];
	static DATA_T W41[24][512][1][1];
	static DATA_T B41[24];
	static DATA_T W43[256][128][3][3];
	static DATA_T B43[256];
	static DATA_T W44[64][24][5][5];
	static DATA_T B44[64];
	static DATA_T W45[64][512][1][1];
	static DATA_T B45[64];
	static DATA_T W47[112][512][1][1];
	static DATA_T B47[112];
	static DATA_T W48[144][512][1][1];
	static DATA_T B48[144];
	static DATA_T W49[32][512][1][1];
	static DATA_T B49[32];
	static DATA_T W51[288][144][3][3];
	static DATA_T B51[288];
	static DATA_T W52[64][32][5][5];
	static DATA_T B52[64];
	static DATA_T W53[64][512][1][1];
	static DATA_T B53[64];
	static DATA_T W55[256][528][1][1];
	static DATA_T B55[256];
	static DATA_T W56[160][528][1][1];
	static DATA_T B56[160];
	static DATA_T W57[32][528][1][1];
	static DATA_T B57[32];
	static DATA_T W59[320][160][3][3];
	static DATA_T B59[320];
	static DATA_T W60[128][32][5][5];
	static DATA_T B60[128];
	static DATA_T W61[128][528][1][1];
	static DATA_T B61[128];
	static DATA_T W64[256][832][1][1];
	static DATA_T B64[256];
	static DATA_T W65[160][832][1][1];
	static DATA_T B65[160];
	static DATA_T W66[32][832][1][1];
	static DATA_T B66[32];
	static DATA_T W68[320][160][3][3];
	static DATA_T B68[320];
	static DATA_T W69[128][32][5][5];
	static DATA_T B69[128];
	static DATA_T W70[128][832][1][1];
	static DATA_T B70[128];
	static DATA_T W72[384][832][1][1];
	static DATA_T B72[384];
	static DATA_T W73[192][832][1][1];
	static DATA_T B73[192];
	static DATA_T W74[48][832][1][1];
	static DATA_T B74[48];
	static DATA_T W76[384][192][3][3];
	static DATA_T B76[384];
	static DATA_T W77[128][48][5][5];
	static DATA_T B77[128];
	static DATA_T W78[128][832][1][1];
	static DATA_T B78[128];
	static DATA_T W82[1000][1024];
	

    // declare array variables of output (static variables)
    static DATA_T O0_SW[3][224][224];
	static DATA_T O1_SW[64][112][112];
	static DATA_T O2_SW[64][56][56];
	static DATA_T O3_SW[64][56][56];
	static DATA_T O4_SW[192][56][56];
	static DATA_T O5_SW[192][28][28];
	static DATA_T O6_SW[64][28][28];
	static DATA_T O7_SW[96][28][28];
	static DATA_T O8_SW[16][28][28];
	static DATA_T O9_SW[192][28][28];
	static DATA_T O10_SW[128][28][28];
	static DATA_T O11_SW[32][28][28];
	static DATA_T O12_SW[32][28][28];
	static DATA_T O13_SW[256][28][28];
	static DATA_T O14_SW[128][28][28];
	static DATA_T O15_SW[128][28][28];
	static DATA_T O16_SW[32][28][28];
	static DATA_T O17_SW[256][28][28];
	static DATA_T O18_SW[192][28][28];
	static DATA_T O19_SW[96][28][28];
	static DATA_T O20_SW[64][28][28];
	static DATA_T O21_SW[480][28][28];
	static DATA_T O22_SW[480][14][14];
	static DATA_T O23_SW[192][14][14];
	static DATA_T O24_SW[96][14][14];
	static DATA_T O25_SW[16][14][14];
	static DATA_T O26_SW[480][14][14];
	static DATA_T O27_SW[208][14][14];
	static DATA_T O28_SW[48][14][14];
	static DATA_T O29_SW[64][14][14];
	static DATA_T O30_SW[512][14][14];
	static DATA_T O31_SW[160][14][14];
	static DATA_T O32_SW[112][14][14];
	static DATA_T O33_SW[24][14][14];
	static DATA_T O34_SW[512][14][14];
	static DATA_T O35_SW[224][14][14];
	static DATA_T O36_SW[64][14][14];
	static DATA_T O37_SW[64][14][14];
	static DATA_T O38_SW[512][14][14];
	static DATA_T O39_SW[128][14][14];
	static DATA_T O40_SW[128][14][14];
	static DATA_T O41_SW[24][14][14];
	static DATA_T O42_SW[512][14][14];
	static DATA_T O43_SW[256][14][14];
	static DATA_T O44_SW[64][14][14];
	static DATA_T O45_SW[64][14][14];
	static DATA_T O46_SW[512][14][14];
	static DATA_T O47_SW[112][14][14];
	static DATA_T O48_SW[144][14][14];
	static DATA_T O49_SW[32][14][14];
	static DATA_T O50_SW[512][14][14];
	static DATA_T O51_SW[288][14][14];
	static DATA_T O52_SW[64][14][14];
	static DATA_T O53_SW[64][14][14];
	static DATA_T O54_SW[528][14][14];
	static DATA_T O55_SW[256][14][14];
	static DATA_T O56_SW[160][14][14];
	static DATA_T O57_SW[32][14][14];
	static DATA_T O58_SW[528][14][14];
	static DATA_T O59_SW[320][14][14];
	static DATA_T O60_SW[128][14][14];
	static DATA_T O61_SW[128][14][14];
	static DATA_T O62_SW[832][14][14];
	static DATA_T O63_SW[832][7][7];
	static DATA_T O64_SW[256][7][7];
	static DATA_T O65_SW[160][7][7];
	static DATA_T O66_SW[32][7][7];
	static DATA_T O67_SW[832][7][7];
	static DATA_T O68_SW[320][7][7];
	static DATA_T O69_SW[128][7][7];
	static DATA_T O70_SW[128][7][7];
	static DATA_T O71_SW[832][7][7];
	static DATA_T O72_SW[384][7][7];
	static DATA_T O73_SW[192][7][7];
	static DATA_T O74_SW[48][7][7];
	static DATA_T O75_SW[832][7][7];
	static DATA_T O76_SW[384][7][7];
	static DATA_T O77_SW[128][7][7];
	static DATA_T O78_SW[128][7][7];
	static DATA_T O79_SW[1024][7][7];
	static DATA_T O80_SW[1024][1][1];
	static DATA_T O81_SW[1024];
	static DATA_T O82_SW[1000];
	

    FILE *w_stream = fopen(argv[1], "rb");
    if (w_stream == NULL) printf("weight file was not opened");
    FILE *i_stream = fopen(argv[2], "rb");
    if (i_stream == NULL) printf("input file was not opened");
    FILE *o_stream = fopen("./output/googlenet/output_value/c_output.txt", "w");
    if (o_stream == NULL) printf("Output file was not opened");
    FILE *c_num = fopen("./output/googlenet/output_value/c_output_num.txt", "w");
    if (c_num == NULL) printf("Output file was not opened");

    // initialize input, weight, bias variables using fread
    printf("[c_verifier.cpp]Start Initialzation\n");
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
                W1[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B1[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W3[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B3[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W4[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B4[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 192 ; i++) {
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

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 192 ; i++) {
			for (j = 0; j < 96 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W7[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 96 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B7[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 192 ; i++) {
			for (j = 0; j < 16 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W8[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 16 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B8[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 96 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W10[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B10[m] = (DATA_T) trash;
}

	for (m = 0; m <  5 ; m++) {
	for (k = 0; k < 5 ; k++) {
		for (i = 0; i < 16 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W11[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B11[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 192 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W12[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B12[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W14[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B14[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W15[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B15[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W16[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B16[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W18[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B18[m] = (DATA_T) trash;
}

	for (m = 0; m <  5 ; m++) {
	for (k = 0; k < 5 ; k++) {
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 96 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W19[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 96 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B19[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W20[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B20[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 480 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W23[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B23[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 480 ; i++) {
			for (j = 0; j < 96 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W24[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 96 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B24[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 480 ; i++) {
			for (j = 0; j < 16 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W25[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 16 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B25[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 96 ; i++) {
			for (j = 0; j < 208 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W27[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 208 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B27[m] = (DATA_T) trash;
}

	for (m = 0; m <  5 ; m++) {
	for (k = 0; k < 5 ; k++) {
		for (i = 0; i < 16 ; i++) {
			for (j = 0; j < 48 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W28[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 48 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B28[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 480 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W29[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B29[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W31[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B31[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 112 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W32[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 112 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B32[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 24 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W33[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 24 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B33[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 112 ; i++) {
			for (j = 0; j < 224 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W35[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 224 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B35[m] = (DATA_T) trash;
}

	for (m = 0; m <  5 ; m++) {
	for (k = 0; k < 5 ; k++) {
		for (i = 0; i < 24 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W36[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B36[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W37[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B37[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W39[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B39[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W40[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B40[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 24 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W41[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 24 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B41[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W43[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B43[m] = (DATA_T) trash;
}

	for (m = 0; m <  5 ; m++) {
	for (k = 0; k < 5 ; k++) {
		for (i = 0; i < 24 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W44[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B44[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W45[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B45[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 112 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W47[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 112 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B47[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 144 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W48[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 144 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B48[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W49[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B49[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 144 ; i++) {
			for (j = 0; j < 288 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W51[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 288 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B51[m] = (DATA_T) trash;
}

	for (m = 0; m <  5 ; m++) {
	for (k = 0; k < 5 ; k++) {
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W52[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B52[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W53[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B53[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 528 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W55[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B55[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 528 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W56[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B56[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 528 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W57[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B57[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 320 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W59[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 320 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B59[m] = (DATA_T) trash;
}

	for (m = 0; m <  5 ; m++) {
	for (k = 0; k < 5 ; k++) {
		for (i = 0; i < 32 ; i++) {
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

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 528 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W61[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B61[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 832 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W64[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B64[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 832 ; i++) {
			for (j = 0; j < 160 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W65[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 160 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B65[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 832 ; i++) {
			for (j = 0; j < 32 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W66[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 32 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B66[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 160 ; i++) {
			for (j = 0; j < 320 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W68[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 320 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B68[m] = (DATA_T) trash;
}

	for (m = 0; m <  5 ; m++) {
	for (k = 0; k < 5 ; k++) {
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W69[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B69[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 832 ; i++) {
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

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 832 ; i++) {
			for (j = 0; j < 384 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W72[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 384 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B72[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 832 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W73[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B73[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 832 ; i++) {
			for (j = 0; j < 48 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W74[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 48 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B74[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 192 ; i++) {
			for (j = 0; j < 384 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W76[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 384 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B76[m] = (DATA_T) trash;
}

	for (m = 0; m <  5 ; m++) {
	for (k = 0; k < 5 ; k++) {
		for (i = 0; i < 48 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W77[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B77[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 832 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W78[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B78[m] = (DATA_T) trash;
}

	for (m = 0; m <  1024 ; m++) {
	for (k = 0; k < 1000 ; k++) {
		fread(&trash, sizeof(int), 1, w_stream);
        W82[k][m] = (DATA_T) trash;
	}
}
/*
for (m = 0; m < 1000 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B82[m] = (DATA_T) trash;
}
*/
	
    printf("[c_verifier.cpp]Finish Initialization\n");

    // call function of each layer based on csv file which containts layer information
    printf("[c_verifier.cpp]InputLayer\n");
	printf("[c_verifier.cpp]Calculate Conv2D1\n");
	SW_block1_conv1(O0_SW,O1_SW,W1,B1);
	printf("[c_verifier.cpp]Calculate MaxPooling2D2\n");
	SW_block1_pool(O1_SW,O2_SW);
	printf("[c_verifier.cpp]Calculate Conv2D3\n");
	SW_block2_conv1(O2_SW,O3_SW,W3,B3);
	printf("[c_verifier.cpp]Calculate Conv2D4\n");
	SW_block2_conv2(O3_SW,O4_SW,W4,B4);
	printf("[c_verifier.cpp]Calculate MaxPooling2D5\n");
	SW_block2_pool(O4_SW,O5_SW);
	printf("[c_verifier.cpp]Calculate Conv2D6\n");
	SW_block3_conv1(O5_SW,O6_SW,W6,B6);
	printf("[c_verifier.cpp]Calculate Conv2D7\n");
	SW_block3_conv2(O5_SW,O7_SW,W7,B7);
	printf("[c_verifier.cpp]Calculate Conv2D8\n");
	SW_block3_conv3(O5_SW,O8_SW,W8,B8);
	printf("[c_verifier.cpp]Calculate MaxPooling2D9\n");
	SW_block3_pool(O5_SW,O9_SW);
	printf("[c_verifier.cpp]Calculate Conv2D10\n");
	SW_block3_conv4(O7_SW,O10_SW,W10,B10);
	printf("[c_verifier.cpp]Calculate Conv2D11\n");
	SW_block3_conv5(O8_SW,O11_SW,W11,B11);
	printf("[c_verifier.cpp]Calculate Conv2D12\n");
	SW_block3_conv6(O9_SW,O12_SW,W12,B12);
	printf("[c_verifier.cpp]Calculate Concatenate13\n");
	SW_concat3(O6_SW, O10_SW, O11_SW, O12_SW,O13_SW);
	printf("[c_verifier.cpp]Calculate Conv2D14\n");
	SW_block4_conv1(O13_SW,O14_SW,W14,B14);
	printf("[c_verifier.cpp]Calculate Conv2D15\n");
	SW_block4_conv2(O13_SW,O15_SW,W15,B15);
	printf("[c_verifier.cpp]Calculate Conv2D16\n");
	SW_block4_conv3(O13_SW,O16_SW,W16,B16);
	printf("[c_verifier.cpp]Calculate MaxPooling2D17\n");
	SW_block4_pool(O13_SW,O17_SW);
	printf("[c_verifier.cpp]Calculate Conv2D18\n");
	SW_block4_conv4(O15_SW,O18_SW,W18,B18);
	printf("[c_verifier.cpp]Calculate Conv2D19\n");
	SW_block4_conv5(O16_SW,O19_SW,W19,B19);
	printf("[c_verifier.cpp]Calculate Conv2D20\n");
	SW_block4_conv6(O17_SW,O20_SW,W20,B20);
	printf("[c_verifier.cpp]Calculate Concatenate21\n");
	SW_concat4(O14_SW, O18_SW, O19_SW, O20_SW,O21_SW);
	printf("[c_verifier.cpp]Calculate MaxPooling2D22\n");
	SW_block5_pool(O21_SW,O22_SW);
	printf("[c_verifier.cpp]Calculate Conv2D23\n");
	SW_block6_conv1(O22_SW,O23_SW,W23,B23);
	printf("[c_verifier.cpp]Calculate Conv2D24\n");
	SW_block6_conv2(O22_SW,O24_SW,W24,B24);
	printf("[c_verifier.cpp]Calculate Conv2D25\n");
	SW_block6_conv3(O22_SW,O25_SW,W25,B25);
	printf("[c_verifier.cpp]Calculate MaxPooling2D26\n");
	SW_block6_pool(O22_SW,O26_SW);
	printf("[c_verifier.cpp]Calculate Conv2D27\n");
	SW_block6_conv4(O24_SW,O27_SW,W27,B27);
	printf("[c_verifier.cpp]Calculate Conv2D28\n");
	SW_block6_conv5(O25_SW,O28_SW,W28,B28);
	printf("[c_verifier.cpp]Calculate Conv2D29\n");
	SW_block6_conv6(O26_SW,O29_SW,W29,B29);
	printf("[c_verifier.cpp]Calculate Concatenate30\n");
	SW_concat6(O23_SW, O27_SW, O28_SW, O29_SW,O30_SW);
	printf("[c_verifier.cpp]Calculate Conv2D31\n");
	SW_block7_conv1(O30_SW,O31_SW,W31,B31);
	printf("[c_verifier.cpp]Calculate Conv2D32\n");
	SW_block7_conv2(O30_SW,O32_SW,W32,B32);
	printf("[c_verifier.cpp]Calculate Conv2D33\n");
	SW_block7_conv3(O30_SW,O33_SW,W33,B33);
	printf("[c_verifier.cpp]Calculate MaxPooling2D34\n");
	SW_block7_pool(O30_SW,O34_SW);
	printf("[c_verifier.cpp]Calculate Conv2D35\n");
	SW_block7_conv4(O32_SW,O35_SW,W35,B35);
	printf("[c_verifier.cpp]Calculate Conv2D36\n");
	SW_block7_conv5(O33_SW,O36_SW,W36,B36);
	printf("[c_verifier.cpp]Calculate Conv2D37\n");
	SW_block7_conv6(O34_SW,O37_SW,W37,B37);
	printf("[c_verifier.cpp]Calculate Concatenate38\n");
	SW_concat7(O31_SW, O35_SW, O36_SW, O37_SW,O38_SW);
	printf("[c_verifier.cpp]Calculate Conv2D39\n");
	SW_block8_conv1(O38_SW,O39_SW,W39,B39);
	printf("[c_verifier.cpp]Calculate Conv2D40\n");
	SW_block8_conv2(O38_SW,O40_SW,W40,B40);
	printf("[c_verifier.cpp]Calculate Conv2D41\n");
	SW_block8_conv3(O38_SW,O41_SW,W41,B41);
	printf("[c_verifier.cpp]Calculate MaxPooling2D42\n");
	SW_block8_pool(O38_SW,O42_SW);
	printf("[c_verifier.cpp]Calculate Conv2D43\n");
	SW_block8_conv4(O40_SW,O43_SW,W43,B43);
	printf("[c_verifier.cpp]Calculate Conv2D44\n");
	SW_block8_conv5(O41_SW,O44_SW,W44,B44);
	printf("[c_verifier.cpp]Calculate Conv2D45\n");
	SW_block8_conv6(O42_SW,O45_SW,W45,B45);
	printf("[c_verifier.cpp]Calculate Concatenate46\n");
	SW_concat8(O39_SW, O43_SW, O44_SW, O45_SW,O46_SW);
	printf("[c_verifier.cpp]Calculate Conv2D47\n");
	SW_block9_conv1(O46_SW,O47_SW,W47,B47);
	printf("[c_verifier.cpp]Calculate Conv2D48\n");
	SW_block9_conv2(O46_SW,O48_SW,W48,B48);
	printf("[c_verifier.cpp]Calculate Conv2D49\n");
	SW_block9_conv3(O46_SW,O49_SW,W49,B49);
	printf("[c_verifier.cpp]Calculate MaxPooling2D50\n");
	SW_block9_pool(O46_SW,O50_SW);
	printf("[c_verifier.cpp]Calculate Conv2D51\n");
	SW_block9_conv4(O48_SW,O51_SW,W51,B51);
	printf("[c_verifier.cpp]Calculate Conv2D52\n");
	SW_block9_conv5(O49_SW,O52_SW,W52,B52);
	printf("[c_verifier.cpp]Calculate Conv2D53\n");
	SW_block9_conv6(O50_SW,O53_SW,W53,B53);
	printf("[c_verifier.cpp]Calculate Concatenate54\n");
	SW_concat9(O47_SW, O51_SW, O52_SW, O53_SW,O54_SW);
	printf("[c_verifier.cpp]Calculate Conv2D55\n");
	SW_block10_conv1(O54_SW,O55_SW,W55,B55);
	printf("[c_verifier.cpp]Calculate Conv2D56\n");
	SW_block10_conv2(O54_SW,O56_SW,W56,B56);
	printf("[c_verifier.cpp]Calculate Conv2D57\n");
	SW_block10_conv3(O54_SW,O57_SW,W57,B57);
	printf("[c_verifier.cpp]Calculate MaxPooling2D58\n");
	SW_block10_pool(O54_SW,O58_SW);
	printf("[c_verifier.cpp]Calculate Conv2D59\n");
	SW_block10_conv4(O56_SW,O59_SW,W59,B59);
	printf("[c_verifier.cpp]Calculate Conv2D60\n");
	SW_block10_conv5(O57_SW,O60_SW,W60,B60);
	printf("[c_verifier.cpp]Calculate Conv2D61\n");
	SW_block10_conv6(O58_SW,O61_SW,W61,B61);
	printf("[c_verifier.cpp]Calculate Concatenate62\n");
	SW_concat10(O55_SW, O59_SW, O60_SW, O61_SW,O62_SW);
	printf("[c_verifier.cpp]Calculate MaxPooling2D63\n");
	SW_block11_pool(O62_SW,O63_SW);
	printf("[c_verifier.cpp]Calculate Conv2D64\n");
	SW_block12_conv1(O63_SW,O64_SW,W64,B64);
	printf("[c_verifier.cpp]Calculate Conv2D65\n");
	SW_block12_conv2(O63_SW,O65_SW,W65,B65);
	printf("[c_verifier.cpp]Calculate Conv2D66\n");
	SW_block12_conv3(O63_SW,O66_SW,W66,B66);
	printf("[c_verifier.cpp]Calculate MaxPooling2D67\n");
	SW_block12_pool(O63_SW,O67_SW);
	printf("[c_verifier.cpp]Calculate Conv2D68\n");
	SW_block12_conv4(O65_SW,O68_SW,W68,B68);
	printf("[c_verifier.cpp]Calculate Conv2D69\n");
	SW_block12_conv5(O66_SW,O69_SW,W69,B69);
	printf("[c_verifier.cpp]Calculate Conv2D70\n");
	SW_block12_conv6(O67_SW,O70_SW,W70,B70);
	printf("[c_verifier.cpp]Calculate Concatenate71\n");
	SW_concat12(O64_SW, O68_SW, O69_SW, O70_SW,O71_SW);
	printf("[c_verifier.cpp]Calculate Conv2D72\n");
	SW_block13_conv1(O71_SW,O72_SW,W72,B72);
	printf("[c_verifier.cpp]Calculate Conv2D73\n");
	SW_block13_conv2(O71_SW,O73_SW,W73,B73);
	printf("[c_verifier.cpp]Calculate Conv2D74\n");
	SW_block13_conv3(O71_SW,O74_SW,W74,B74);
	printf("[c_verifier.cpp]Calculate MaxPooling2D75\n");
	SW_block13_pool(O71_SW,O75_SW);
	printf("[c_verifier.cpp]Calculate Conv2D76\n");
	SW_block13_conv4(O73_SW,O76_SW,W76,B76);
	printf("[c_verifier.cpp]Calculate Conv2D77\n");
	SW_block13_conv5(O74_SW,O77_SW,W77,B77);
	printf("[c_verifier.cpp]Calculate Conv2D78\n");
	SW_block13_conv6(O75_SW,O78_SW,W78,B78);
	printf("[c_verifier.cpp]Calculate Concatenate79\n");
	SW_concat13(O72_SW, O76_SW, O77_SW, O78_SW,O79_SW);
	printf("[c_verifier.cpp]Calculate AveragePooling2D80\n");
	SW_avg_pool1(O79_SW,O80_SW);
	printf("[c_verifier.cpp]Calculate Flatten81\n");
	SW_flatten(O80_SW,O81_SW);
	printf("[c_verifier.cpp]Calculate Dense82\n");
	SW_predictions(O81_SW,O82_SW,W82);
	

    // print each element of output variables
    printf("[c_verifier.cpp]Print Result\n");


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
		for(y = 0; y < 64 ; y++) {
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


	fprintf(o_stream,"%s","MaxPooling2D : [[");
for (k = 0; k < 56 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 56 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O2_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O2_SW[y][k][x]);
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
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O3_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O3_SW[y][k][x]);
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
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O4_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O4_SW[y][k][x]);
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


	fprintf(o_stream,"%s","MaxPooling2D : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O5_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O5_SW[y][k][x]);
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
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O6_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O6_SW[y][k][x]);
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
		for(y = 0; y < 96 ; y++) {
			fprintf(o_stream,"%.6f ",O7_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O7_SW[y][k][x]);
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
		for(y = 0; y < 16 ; y++) {
			fprintf(o_stream,"%.6f ",O8_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O8_SW[y][k][x]);
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


	fprintf(o_stream,"%s","MaxPooling2D : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O9_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O9_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O10_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O10_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O11_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O11_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O12_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O12_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O13_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O13_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O14_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O14_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O15_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O15_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O16_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O16_SW[y][k][x]);
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


	fprintf(o_stream,"%s","MaxPooling2D : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O17_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O17_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O18_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O18_SW[y][k][x]);
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
		for(y = 0; y < 96 ; y++) {
			fprintf(o_stream,"%.6f ",O19_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O19_SW[y][k][x]);
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
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",O20_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O20_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O21_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O21_SW[y][k][x]);
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


	fprintf(o_stream,"%s","MaxPooling2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 480 ; y++) {
			fprintf(o_stream,"%.6f ",O22_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O22_SW[y][k][x]);
		}
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
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O23_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O23_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O24_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O24_SW[y][k][x]);
		}
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
		for(y = 0; y < 16 ; y++) {
			fprintf(o_stream,"%.6f ",O25_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O25_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","MaxPooling2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 480 ; y++) {
			fprintf(o_stream,"%.6f ",O26_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O26_SW[y][k][x]);
		}
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
		for(y = 0; y < 208 ; y++) {
			fprintf(o_stream,"%.6f ",O27_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O27_SW[y][k][x]);
		}
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
		for(y = 0; y < 48 ; y++) {
			fprintf(o_stream,"%.6f ",O28_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O28_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O29_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O29_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O30_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O30_SW[y][k][x]);
		}
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
		for(y = 0; y < 160 ; y++) {
			fprintf(o_stream,"%.6f ",O31_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O31_SW[y][k][x]);
		}
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
		for(y = 0; y < 112 ; y++) {
			fprintf(o_stream,"%.6f ",O32_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O32_SW[y][k][x]);
		}
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
		for(y = 0; y < 24 ; y++) {
			fprintf(o_stream,"%.6f ",O33_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O33_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","MaxPooling2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",O34_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O34_SW[y][k][x]);
		}
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
		for(y = 0; y < 224 ; y++) {
			fprintf(o_stream,"%.6f ",O35_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O35_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O36_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O36_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O37_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O37_SW[y][k][x]);
		}
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


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
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


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
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
		for(y = 0; y < 24 ; y++) {
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


	fprintf(o_stream,"%s","MaxPooling2D : [[");
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


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
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


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
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


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
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


	fprintf(o_stream,"%s","Concatenate : [[");
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
		for(y = 0; y < 112 ; y++) {
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


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 144 ; y++) {
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


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
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


	fprintf(o_stream,"%s","MaxPooling2D : [[");
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


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 288 ; y++) {
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


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
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
		for(y = 0; y < 64 ; y++) {
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


	fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 528 ; y++) {
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


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
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


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 160 ; y++) {
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


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
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


	fprintf(o_stream,"%s","MaxPooling2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 528 ; y++) {
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
		for(y = 0; y < 320 ; y++) {
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
		for(y = 0; y < 128 ; y++) {
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


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
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


	fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 832 ; y++) {
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


	fprintf(o_stream,"%s","MaxPooling2D : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 832 ; y++) {
			fprintf(o_stream,"%.6f ",O63_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O63_SW[y][k][x]);
		}
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
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O64_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O64_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O65_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O65_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O66_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O66_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","MaxPooling2D : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 832 ; y++) {
			fprintf(o_stream,"%.6f ",O67_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O67_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O68_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O68_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O69_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O69_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O70_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O70_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O71_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O71_SW[y][k][x]);
		}
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
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O72_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O72_SW[y][k][x]);
		}
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
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",O73_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O73_SW[y][k][x]);
		}
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
		for(y = 0; y < 48 ; y++) {
			fprintf(o_stream,"%.6f ",O74_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O74_SW[y][k][x]);
		}
		if(x != 7 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 7 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","MaxPooling2D : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 832 ; y++) {
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


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
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


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
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
		for(y = 0; y < 128 ; y++) {
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


	fprintf(o_stream,"%s","Concatenate : [[");
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


	fprintf(o_stream,"%s","AveragePooling2D : [[");
for (k = 0; k < 1 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 1 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O80_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O80_SW[y][k][x]);
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
for (k = 0; k <  1024 ; k++) {
	fprintf(o_stream,"%.6f ",O81_SW[k]);
	fprintf(c_num,"%.6f ",O81_SW[k]);
}
fprintf(o_stream,"%s","]]\n\n");


	fprintf(o_stream,"%s","Dense : [[");
for (k = 0; k <  1000 ; k++) {
	fprintf(o_stream,"%.6f ",O82_SW[k]);
	fprintf(c_num,"%.6f ",O82_SW[k]);
}
fprintf(o_stream,"%s","]]\n\n");


	


    fclose(w_stream);
    fclose(i_stream);
    fclose(o_stream);
    fclose(c_num);

    return 0;
}
