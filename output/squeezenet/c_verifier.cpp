#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <math.h>

using namespace std;

typedef float DATA_T;

// define functions of each layer
void SW_conv1(DATA_T I[3][224][224], DATA_T O[64][112][112], DATA_T W[64][3][3][3], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (2 *(112 - 1) - 224 + 3)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<112; x++) {
			for (y = 0; y<112; y++) {
				ofm = B[m];
				for (k = 0; k<3; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*2 + i < 224 + p && y*2 + j < 224 + p && x*2 + i -p >= 0 && y*2 + j -p >= 0) {
                                    				ifm = I[k][x*2 + i - p][y*2 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				if (ofm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ofm;
			}
		}
	}
}

void SW_maxpool1(DATA_T I[64][112][112], DATA_T O[64][55][55])
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
void SW_fire2_squeeze1x1(DATA_T I[64][55][55], DATA_T O[16][55][55], DATA_T W[16][64][1][1], DATA_T B[16]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(55 - 1) - 55 + 1)/2;
	for (m = 0; m<16; m++) {
		for (x = 0; x<55; x++) {
			for (y = 0; y<55; y++) {
				ofm = B[m];
				for (k = 0; k<64; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 55 + p && y*1 + j < 55 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                                    				ifm = I[k][x*1 + i - p][y*1 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				if (ofm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ofm;
			}
		}
	}
}

void SW_fire2_expand1x1(DATA_T I[16][55][55], DATA_T O[64][55][55], DATA_T W[64][16][1][1], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(55 - 1) - 55 + 1)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<55; x++) {
			for (y = 0; y<55; y++) {
				ofm = B[m];
				for (k = 0; k<16; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 55 + p && y*1 + j < 55 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                                    				ifm = I[k][x*1 + i - p][y*1 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				if (ofm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ofm;
			}
		}
	}
}

void SW_fire2_expand3x3(DATA_T I[16][55][55], DATA_T O[64][55][55], DATA_T W[64][16][3][3], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(55 - 1) - 55 + 3)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<55; x++) {
			for (y = 0; y<55; y++) {
				ofm = B[m];
				for (k = 0; k<16; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 55 + p && y*1 + j < 55 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                                    				ifm = I[k][x*1 + i - p][y*1 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				if (ofm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ofm;
			}
		}
	}
}

void SW_fire2(DATA_T I1[64][55][55], DATA_T I2[64][55][55], DATA_T O[128][55][55]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 55; x++) {
		for(y = 0; y < 55; y++) {
			ch=0;
			for(k = ch; k < ch+64; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=64;
			for(k = ch; k < ch+64; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_fire3_squeeze1x1(DATA_T I[128][55][55], DATA_T O[16][55][55], DATA_T W[16][128][1][1], DATA_T B[16]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(55 - 1) - 55 + 1)/2;
	for (m = 0; m<16; m++) {
		for (x = 0; x<55; x++) {
			for (y = 0; y<55; y++) {
				ofm = B[m];
				for (k = 0; k<128; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 55 + p && y*1 + j < 55 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                                    				ifm = I[k][x*1 + i - p][y*1 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				if (ofm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ofm;
			}
		}
	}
}

void SW_fire3_expand1x1(DATA_T I[16][55][55], DATA_T O[64][55][55], DATA_T W[64][16][1][1], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(55 - 1) - 55 + 1)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<55; x++) {
			for (y = 0; y<55; y++) {
				ofm = B[m];
				for (k = 0; k<16; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 55 + p && y*1 + j < 55 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                                    				ifm = I[k][x*1 + i - p][y*1 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				if (ofm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ofm;
			}
		}
	}
}

void SW_fire3_expand3x3(DATA_T I[16][55][55], DATA_T O[64][55][55], DATA_T W[64][16][3][3], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(55 - 1) - 55 + 3)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<55; x++) {
			for (y = 0; y<55; y++) {
				ofm = B[m];
				for (k = 0; k<16; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 55 + p && y*1 + j < 55 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                                    				ifm = I[k][x*1 + i - p][y*1 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				if (ofm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ofm;
			}
		}
	}
}

void SW_fire3(DATA_T I1[64][55][55], DATA_T I2[64][55][55], DATA_T O[128][55][55]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 55; x++) {
		for(y = 0; y < 55; y++) {
			ch=0;
			for(k = ch; k < ch+64; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=64;
			for(k = ch; k < ch+64; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_maxpool3(DATA_T I[128][55][55], DATA_T O[128][27][27])
{
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<128; m++) {
		for (x = 0; x<27; x++) {
			for (y = 0; y<27; y++) {
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
void SW_fire4_squeeze1x1(DATA_T I[128][27][27], DATA_T O[32][27][27], DATA_T W[32][128][1][1], DATA_T B[32]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(27 - 1) - 27 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<27; x++) {
			for (y = 0; y<27; y++) {
				ofm = B[m];
				for (k = 0; k<128; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 27 + p && y*1 + j < 27 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                                    				ifm = I[k][x*1 + i - p][y*1 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				if (ofm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ofm;
			}
		}
	}
}

void SW_fire4_expand1x1(DATA_T I[32][27][27], DATA_T O[128][27][27], DATA_T W[128][32][1][1], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(27 - 1) - 27 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<27; x++) {
			for (y = 0; y<27; y++) {
				ofm = B[m];
				for (k = 0; k<32; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 27 + p && y*1 + j < 27 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                                    				ifm = I[k][x*1 + i - p][y*1 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				if (ofm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ofm;
			}
		}
	}
}

void SW_fire4_expand3x3(DATA_T I[32][27][27], DATA_T O[128][27][27], DATA_T W[128][32][3][3], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(27 - 1) - 27 + 3)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<27; x++) {
			for (y = 0; y<27; y++) {
				ofm = B[m];
				for (k = 0; k<32; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 27 + p && y*1 + j < 27 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                                    				ifm = I[k][x*1 + i - p][y*1 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				if (ofm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ofm;
			}
		}
	}
}

void SW_fire4(DATA_T I1[128][27][27], DATA_T I2[128][27][27], DATA_T O[256][27][27]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 27; x++) {
		for(y = 0; y < 27; y++) {
			ch=0;
			for(k = ch; k < ch+128; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=128;
			for(k = ch; k < ch+128; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_fire5_squeeze1x1(DATA_T I[256][27][27], DATA_T O[32][27][27], DATA_T W[32][256][1][1], DATA_T B[32]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(27 - 1) - 27 + 1)/2;
	for (m = 0; m<32; m++) {
		for (x = 0; x<27; x++) {
			for (y = 0; y<27; y++) {
				ofm = B[m];
				for (k = 0; k<256; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 27 + p && y*1 + j < 27 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                                    				ifm = I[k][x*1 + i - p][y*1 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				if (ofm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ofm;
			}
		}
	}
}

void SW_fire5_expand1x1(DATA_T I[32][27][27], DATA_T O[128][27][27], DATA_T W[128][32][1][1], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(27 - 1) - 27 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<27; x++) {
			for (y = 0; y<27; y++) {
				ofm = B[m];
				for (k = 0; k<32; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 27 + p && y*1 + j < 27 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                                    				ifm = I[k][x*1 + i - p][y*1 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				if (ofm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ofm;
			}
		}
	}
}

void SW_fire5_expand3x3(DATA_T I[32][27][27], DATA_T O[128][27][27], DATA_T W[128][32][3][3], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(27 - 1) - 27 + 3)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<27; x++) {
			for (y = 0; y<27; y++) {
				ofm = B[m];
				for (k = 0; k<32; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 27 + p && y*1 + j < 27 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                                    				ifm = I[k][x*1 + i - p][y*1 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				if (ofm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ofm;
			}
		}
	}
}

void SW_fire5(DATA_T I1[128][27][27], DATA_T I2[128][27][27], DATA_T O[256][27][27]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 27; x++) {
		for(y = 0; y < 27; y++) {
			ch=0;
			for(k = ch; k < ch+128; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=128;
			for(k = ch; k < ch+128; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_maxpool5(DATA_T I[256][27][27], DATA_T O[256][13][13])
{
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<256; m++) {
		for (x = 0; x<13; x++) {
			for (y = 0; y<13; y++) {
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
void SW_fire6_squeeze1x1(DATA_T I[256][13][13], DATA_T O[48][13][13], DATA_T W[48][256][1][1], DATA_T B[48]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(13 - 1) - 13 + 1)/2;
	for (m = 0; m<48; m++) {
		for (x = 0; x<13; x++) {
			for (y = 0; y<13; y++) {
				ofm = B[m];
				for (k = 0; k<256; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 13 + p && y*1 + j < 13 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                                    				ifm = I[k][x*1 + i - p][y*1 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				if (ofm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ofm;
			}
		}
	}
}

void SW_fire6_expand1x1(DATA_T I[48][13][13], DATA_T O[192][13][13], DATA_T W[192][48][1][1], DATA_T B[192]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(13 - 1) - 13 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<13; x++) {
			for (y = 0; y<13; y++) {
				ofm = B[m];
				for (k = 0; k<48; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 13 + p && y*1 + j < 13 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                                    				ifm = I[k][x*1 + i - p][y*1 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				if (ofm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ofm;
			}
		}
	}
}

void SW_fire6_expand3x3(DATA_T I[48][13][13], DATA_T O[192][13][13], DATA_T W[192][48][3][3], DATA_T B[192]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(13 - 1) - 13 + 3)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<13; x++) {
			for (y = 0; y<13; y++) {
				ofm = B[m];
				for (k = 0; k<48; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 13 + p && y*1 + j < 13 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                                    				ifm = I[k][x*1 + i - p][y*1 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				if (ofm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ofm;
			}
		}
	}
}

void SW_fire6(DATA_T I1[192][13][13], DATA_T I2[192][13][13], DATA_T O[384][13][13]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 13; x++) {
		for(y = 0; y < 13; y++) {
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
void SW_fire7_squeeze1x1(DATA_T I[384][13][13], DATA_T O[48][13][13], DATA_T W[48][384][1][1], DATA_T B[48]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(13 - 1) - 13 + 1)/2;
	for (m = 0; m<48; m++) {
		for (x = 0; x<13; x++) {
			for (y = 0; y<13; y++) {
				ofm = B[m];
				for (k = 0; k<384; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 13 + p && y*1 + j < 13 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                                    				ifm = I[k][x*1 + i - p][y*1 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				if (ofm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ofm;
			}
		}
	}
}

void SW_fire7_expand1x1(DATA_T I[48][13][13], DATA_T O[192][13][13], DATA_T W[192][48][1][1], DATA_T B[192]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(13 - 1) - 13 + 1)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<13; x++) {
			for (y = 0; y<13; y++) {
				ofm = B[m];
				for (k = 0; k<48; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 13 + p && y*1 + j < 13 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                                    				ifm = I[k][x*1 + i - p][y*1 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				if (ofm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ofm;
			}
		}
	}
}

void SW_fire7_expand3x3(DATA_T I[48][13][13], DATA_T O[192][13][13], DATA_T W[192][48][3][3], DATA_T B[192]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(13 - 1) - 13 + 3)/2;
	for (m = 0; m<192; m++) {
		for (x = 0; x<13; x++) {
			for (y = 0; y<13; y++) {
				ofm = B[m];
				for (k = 0; k<48; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 13 + p && y*1 + j < 13 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                                    				ifm = I[k][x*1 + i - p][y*1 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				if (ofm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ofm;
			}
		}
	}
}

void SW_fire7(DATA_T I1[192][13][13], DATA_T I2[192][13][13], DATA_T O[384][13][13]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 13; x++) {
		for(y = 0; y < 13; y++) {
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
void SW_fire8_squeeze1x1(DATA_T I[384][13][13], DATA_T O[64][13][13], DATA_T W[64][384][1][1], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(13 - 1) - 13 + 1)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<13; x++) {
			for (y = 0; y<13; y++) {
				ofm = B[m];
				for (k = 0; k<384; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 13 + p && y*1 + j < 13 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                                    				ifm = I[k][x*1 + i - p][y*1 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				if (ofm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ofm;
			}
		}
	}
}

void SW_fire8_expand1x1(DATA_T I[64][13][13], DATA_T O[256][13][13], DATA_T W[256][64][1][1], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(13 - 1) - 13 + 1)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<13; x++) {
			for (y = 0; y<13; y++) {
				ofm = B[m];
				for (k = 0; k<64; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 13 + p && y*1 + j < 13 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                                    				ifm = I[k][x*1 + i - p][y*1 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				if (ofm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ofm;
			}
		}
	}
}

void SW_fire8_expand3x3(DATA_T I[64][13][13], DATA_T O[256][13][13], DATA_T W[256][64][3][3], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(13 - 1) - 13 + 3)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<13; x++) {
			for (y = 0; y<13; y++) {
				ofm = B[m];
				for (k = 0; k<64; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 13 + p && y*1 + j < 13 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                                    				ifm = I[k][x*1 + i - p][y*1 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				if (ofm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ofm;
			}
		}
	}
}

void SW_fire8(DATA_T I1[256][13][13], DATA_T I2[256][13][13], DATA_T O[512][13][13]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 13; x++) {
		for(y = 0; y < 13; y++) {
			ch=0;
			for(k = ch; k < ch+256; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=256;
			for(k = ch; k < ch+256; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_fire9_squeeze1x1(DATA_T I[512][13][13], DATA_T O[64][13][13], DATA_T W[64][512][1][1], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(13 - 1) - 13 + 1)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<13; x++) {
			for (y = 0; y<13; y++) {
				ofm = B[m];
				for (k = 0; k<512; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 13 + p && y*1 + j < 13 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                                    				ifm = I[k][x*1 + i - p][y*1 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				if (ofm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ofm;
			}
		}
	}
}

void SW_fire9_expand1x1(DATA_T I[64][13][13], DATA_T O[256][13][13], DATA_T W[256][64][1][1], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(13 - 1) - 13 + 1)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<13; x++) {
			for (y = 0; y<13; y++) {
				ofm = B[m];
				for (k = 0; k<64; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*1 + i < 13 + p && y*1 + j < 13 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                                    				ifm = I[k][x*1 + i - p][y*1 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				if (ofm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ofm;
			}
		}
	}
}

void SW_fire9_expand3x3(DATA_T I[64][13][13], DATA_T O[256][13][13], DATA_T W[256][64][3][3], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(13 - 1) - 13 + 3)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<13; x++) {
			for (y = 0; y<13; y++) {
				ofm = B[m];
				for (k = 0; k<64; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 13 + p && y*1 + j < 13 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                                    				ifm = I[k][x*1 + i - p][y*1 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				if (ofm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ofm;
			}
		}
	}
}

void SW_fire9(DATA_T I1[256][13][13], DATA_T I2[256][13][13], DATA_T O[512][13][13]) {
	int x, y, k;
	int ch=0;
	
	for (x = 0; x < 13; x++) {
		for(y = 0; y < 13; y++) {
			ch=0;
			for(k = ch; k < ch+256; k++){
				O[k][x][y] = I1[k][x][y];
			}
			ch+=256;
			for(k = ch; k < ch+256; k++){
				O[k][x][y] = I2[k-ch][x][y];
			}
		}
	}

}
void SW_conv10(DATA_T I[512][13][13], DATA_T O[1000][13][13], DATA_T W[1000][512][1][1], DATA_T B[1000]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<1000; m++) {
		for (x = 0; x<13; x++) {
			for (y = 0; y<13; y++) {
				ofm = B[m];
				for (k = 0; k<512; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 13 && y + j <= 13) {
								ifm = I[k][x*1 + i][y*1 + j];
							}

							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				if (ofm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ofm;
			}
		}
	}
}

void SW_avgpool10(DATA_T I[1000][13][13], DATA_T O[1000][1][1])
{
	int m, x, y, i, j;
	DATA_T sum;
	int p = (13*(1-1) - 13 + 13)/2;
    DATA_T div;
	for (m = 0; m<1000; m++) {
		for (x = 0; x<1; x++) {
			for (y = 0; y<1; y++) {
                sum = (DATA_T)0;
                if(x!=0 && x!=1-1 && y!=0 && y!=1-1)
                    div = 9;
                else if((x==0 || x==1-1) && (y==0 || y==1-1))
                    div = 4;
                else
                    div = 6;
				for (i = 0; i<13; i++) {
					for (j = 0; j<13; j++) {

						if (x + i < 13 + p && y + j < 13 + p && x + i -p >= 0 && y + j -p >= 0)
						    sum += I[m][x*13 + i -p][y*13 + j -p];
					}
				}
				O[m][x][y] = sum/div;
			}
		}
	}
}
void SW_flatten10(DATA_T I[1000][1][1], DATA_T O[1000]){
	int i, j, x, y;
	i = 0;
	 for(x=0; x<1;x++)
		for(y=0; y<1; y++)
			for (j=0; j<1000; j++) {
				O[i] = I[j][x][y];
				i++;
			}
}

void SW_softmax(DATA_T I[1000], DATA_T O[1000]) {
	int i;
	DATA_T maximum;
	DATA_T denom = I[0];
	for (i = 1; i < 1000; i++)
		denom += I[i];
	for (i = 0; i < 1000; i++){
		O[i] = I[i] / denom; //float
		if(i==0)
		    maximum=O[i];
		else{
		    if(maximum<O[i])
		        maximum=O[i];

		}
	}
	for (i = 0; i < 1000; i++){
	    if(maximum!=O[i])
	        O[i]=0;
	    else
	        O[i]=1;
	}

}


//argv[1] = init_weight.txt, argv[2] = init_input.txt
int main(int argc, char *argv[]){

    DATA_T temp;
    int m, x, y, i, j, k, l;
    int trash;

    // declare array variables of input, weight and bias (static variables)
    static DATA_T I[3][224][224];
	static DATA_T W1[64][3][3][3];
	static DATA_T B1[64];
	static DATA_T W3[16][64][1][1];
	static DATA_T B3[16];
	static DATA_T W4[64][16][1][1];
	static DATA_T B4[64];
	static DATA_T W5[64][16][3][3];
	static DATA_T B5[64];
	static DATA_T W7[16][128][1][1];
	static DATA_T B7[16];
	static DATA_T W8[64][16][1][1];
	static DATA_T B8[64];
	static DATA_T W9[64][16][3][3];
	static DATA_T B9[64];
	static DATA_T W12[32][128][1][1];
	static DATA_T B12[32];
	static DATA_T W13[128][32][1][1];
	static DATA_T B13[128];
	static DATA_T W14[128][32][3][3];
	static DATA_T B14[128];
	static DATA_T W16[32][256][1][1];
	static DATA_T B16[32];
	static DATA_T W17[128][32][1][1];
	static DATA_T B17[128];
	static DATA_T W18[128][32][3][3];
	static DATA_T B18[128];
	static DATA_T W21[48][256][1][1];
	static DATA_T B21[48];
	static DATA_T W22[192][48][1][1];
	static DATA_T B22[192];
	static DATA_T W23[192][48][3][3];
	static DATA_T B23[192];
	static DATA_T W25[48][384][1][1];
	static DATA_T B25[48];
	static DATA_T W26[192][48][1][1];
	static DATA_T B26[192];
	static DATA_T W27[192][48][3][3];
	static DATA_T B27[192];
	static DATA_T W29[64][384][1][1];
	static DATA_T B29[64];
	static DATA_T W30[256][64][1][1];
	static DATA_T B30[256];
	static DATA_T W31[256][64][3][3];
	static DATA_T B31[256];
	static DATA_T W33[64][512][1][1];
	static DATA_T B33[64];
	static DATA_T W34[256][64][1][1];
	static DATA_T B34[256];
	static DATA_T W35[256][64][3][3];
	static DATA_T B35[256];
	static DATA_T W37[1000][512][1][1];
	static DATA_T B37[1000];
	

    // declare array variables of output (static variables)
    static DATA_T O0_SW[3][224][224];
	static DATA_T O1_SW[64][112][112];
	static DATA_T O2_SW[64][55][55];
	static DATA_T O3_SW[16][55][55];
	static DATA_T O4_SW[64][55][55];
	static DATA_T O5_SW[64][55][55];
	static DATA_T O6_SW[128][55][55];
	static DATA_T O7_SW[16][55][55];
	static DATA_T O8_SW[64][55][55];
	static DATA_T O9_SW[64][55][55];
	static DATA_T O10_SW[128][55][55];
	static DATA_T O11_SW[128][27][27];
	static DATA_T O12_SW[32][27][27];
	static DATA_T O13_SW[128][27][27];
	static DATA_T O14_SW[128][27][27];
	static DATA_T O15_SW[256][27][27];
	static DATA_T O16_SW[32][27][27];
	static DATA_T O17_SW[128][27][27];
	static DATA_T O18_SW[128][27][27];
	static DATA_T O19_SW[256][27][27];
	static DATA_T O20_SW[256][13][13];
	static DATA_T O21_SW[48][13][13];
	static DATA_T O22_SW[192][13][13];
	static DATA_T O23_SW[192][13][13];
	static DATA_T O24_SW[384][13][13];
	static DATA_T O25_SW[48][13][13];
	static DATA_T O26_SW[192][13][13];
	static DATA_T O27_SW[192][13][13];
	static DATA_T O28_SW[384][13][13];
	static DATA_T O29_SW[64][13][13];
	static DATA_T O30_SW[256][13][13];
	static DATA_T O31_SW[256][13][13];
	static DATA_T O32_SW[512][13][13];
	static DATA_T O33_SW[64][13][13];
	static DATA_T O34_SW[256][13][13];
	static DATA_T O35_SW[256][13][13];
	static DATA_T O36_SW[512][13][13];
	static DATA_T O37_SW[1000][13][13];
	static DATA_T O38_SW[1000][1][1];
	static DATA_T O39_SW[1000];
	static DATA_T O40_SW[1000];
	

    FILE *w_stream = fopen(argv[1], "rb");
    if (w_stream == NULL) printf("weight file was not opened");
    FILE *i_stream = fopen(argv[2], "rb");
    if (i_stream == NULL) printf("input file was not opened");
    FILE *o_stream = fopen("./output/squeezenet/output_value/c_output.txt", "w");
    if (o_stream == NULL) printf("Output file was not opened");
    FILE *c_num = fopen("./output/squeezenet/output_value/c_output_num.txt", "w");
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
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
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
			for (j = 0; j < 16 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W3[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 16 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B3[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 16 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W4[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B4[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 16 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W5[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B5[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 16 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W7[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 16 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B7[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 16 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W8[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B8[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 16 ; i++) {
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

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 128 ; i++) {
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
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W13[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B13[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 32 ; i++) {
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

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W17[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B17[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W18[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B18[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 48 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W21[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 48 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B21[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 48 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W22[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B22[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 48 ; i++) {
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
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 48 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W25[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 48 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B25[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 48 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W26[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B26[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 48 ; i++) {
			for (j = 0; j < 192 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W27[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 192 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B27[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 384 ; i++) {
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
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W30[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B30[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W31[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B31[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W33[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B33[m] = (DATA_T) trash;
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

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W35[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B35[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 1000 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W37[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 1000 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B37[m] = (DATA_T) trash;
}

	
    printf("[c_verifier.cpp]Finish Initialization\n");

    // call function of each layer based on csv file which containts layer information
    printf("[c_verifier.cpp]InputLayer\n");
	printf("[c_verifier.cpp]Calculate Conv2D1\n");
	SW_conv1(O0_SW,O1_SW,W1,B1);
	printf("[c_verifier.cpp]Calculate MaxPooling2D2\n");
	SW_maxpool1(O1_SW,O2_SW);
	printf("[c_verifier.cpp]Calculate Conv2D3\n");
	SW_fire2_squeeze1x1(O2_SW,O3_SW,W3,B3);
	printf("[c_verifier.cpp]Calculate Conv2D4\n");
	SW_fire2_expand1x1(O3_SW,O4_SW,W4,B4);
	printf("[c_verifier.cpp]Calculate Conv2D5\n");
	SW_fire2_expand3x3(O3_SW,O5_SW,W5,B5);
	printf("[c_verifier.cpp]Calculate Concatenate6\n");
	SW_fire2(O4_SW, O5_SW, O6_SW);
	printf("[c_verifier.cpp]Calculate Conv2D7\n");
	SW_fire3_squeeze1x1(O6_SW,O7_SW,W7,B7);
	printf("[c_verifier.cpp]Calculate Conv2D8\n");
	SW_fire3_expand1x1(O7_SW,O8_SW,W8,B8);
	printf("[c_verifier.cpp]Calculate Conv2D9\n");
	SW_fire3_expand3x3(O7_SW,O9_SW,W9,B9);
	printf("[c_verifier.cpp]Calculate Concatenate10\n");
	SW_fire3(O8_SW, O9_SW, O10_SW);
	printf("[c_verifier.cpp]Calculate MaxPooling2D11\n");
	SW_maxpool3(O10_SW,O11_SW);
	printf("[c_verifier.cpp]Calculate Conv2D12\n");
	SW_fire4_squeeze1x1(O11_SW,O12_SW,W12,B12);
	printf("[c_verifier.cpp]Calculate Conv2D13\n");
	SW_fire4_expand1x1(O12_SW,O13_SW,W13,B13);
	printf("[c_verifier.cpp]Calculate Conv2D14\n");
	SW_fire4_expand3x3(O12_SW,O14_SW,W14,B14);
	printf("[c_verifier.cpp]Calculate Concatenate15\n");
	SW_fire4(O13_SW, O14_SW, O15_SW);
	printf("[c_verifier.cpp]Calculate Conv2D16\n");
	SW_fire5_squeeze1x1(O15_SW,O16_SW,W16,B16);
	printf("[c_verifier.cpp]Calculate Conv2D17\n");
	SW_fire5_expand1x1(O16_SW,O17_SW,W17,B17);
	printf("[c_verifier.cpp]Calculate Conv2D18\n");
	SW_fire5_expand3x3(O16_SW,O18_SW,W18,B18);
	printf("[c_verifier.cpp]Calculate Concatenate19\n");
	SW_fire5(O17_SW, O18_SW, O19_SW);
	printf("[c_verifier.cpp]Calculate MaxPooling2D20\n");
	SW_maxpool5(O19_SW,O20_SW);
	printf("[c_verifier.cpp]Calculate Conv2D21\n");
	SW_fire6_squeeze1x1(O20_SW,O21_SW,W21,B21);
	printf("[c_verifier.cpp]Calculate Conv2D22\n");
	SW_fire6_expand1x1(O21_SW,O22_SW,W22,B22);
	printf("[c_verifier.cpp]Calculate Conv2D23\n");
	SW_fire6_expand3x3(O21_SW,O23_SW,W23,B23);
	printf("[c_verifier.cpp]Calculate Concatenate24\n");
	SW_fire6(O22_SW, O23_SW, O24_SW);
	printf("[c_verifier.cpp]Calculate Conv2D25\n");
	SW_fire7_squeeze1x1(O24_SW,O25_SW,W25,B25);
	printf("[c_verifier.cpp]Calculate Conv2D26\n");
	SW_fire7_expand1x1(O25_SW,O26_SW,W26,B26);
	printf("[c_verifier.cpp]Calculate Conv2D27\n");
	SW_fire7_expand3x3(O25_SW,O27_SW,W27,B27);
	printf("[c_verifier.cpp]Calculate Concatenate28\n");
	SW_fire7(O26_SW, O27_SW, O28_SW);
	printf("[c_verifier.cpp]Calculate Conv2D29\n");
	SW_fire8_squeeze1x1(O28_SW,O29_SW,W29,B29);
	printf("[c_verifier.cpp]Calculate Conv2D30\n");
	SW_fire8_expand1x1(O29_SW,O30_SW,W30,B30);
	printf("[c_verifier.cpp]Calculate Conv2D31\n");
	SW_fire8_expand3x3(O29_SW,O31_SW,W31,B31);
	printf("[c_verifier.cpp]Calculate Concatenate32\n");
	SW_fire8(O30_SW, O31_SW, O32_SW);
	printf("[c_verifier.cpp]Calculate Conv2D33\n");
	SW_fire9_squeeze1x1(O32_SW,O33_SW,W33,B33);
	printf("[c_verifier.cpp]Calculate Conv2D34\n");
	SW_fire9_expand1x1(O33_SW,O34_SW,W34,B34);
	printf("[c_verifier.cpp]Calculate Conv2D35\n");
	SW_fire9_expand3x3(O33_SW,O35_SW,W35,B35);
	printf("[c_verifier.cpp]Calculate Concatenate36\n");
	SW_fire9(O34_SW, O35_SW, O36_SW);
	printf("[c_verifier.cpp]Calculate Conv2D37\n");
	SW_conv10(O36_SW,O37_SW,W37,B37);
	printf("[c_verifier.cpp]Calculate AveragePooling2D38\n");
	SW_avgpool10(O37_SW,O38_SW);
	printf("[c_verifier.cpp]Calculate Flatten39\n");
	SW_flatten10(O38_SW,O39_SW);
	printf("[c_verifier.cpp]Calculate Activation40\n");
	SW_softmax(O39_SW,O40_SW);
	

    // print each element of output variables
    printf("[c_verifier.cpp]Print Result\n");


    fprintf(o_stream,"%s","InputLayer : [[");
for (k = 0; k < 224 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 224 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 3 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O0_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O0_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",(float)O1_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O1_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",(float)O2_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O2_SW[y][k][x]);
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
		for(y = 0; y < 16 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O3_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O3_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",(float)O4_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O4_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",(float)O5_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O5_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 55 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 55 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O6_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O6_SW[y][k][x]);
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
		for(y = 0; y < 16 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O7_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O7_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",(float)O8_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O8_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",(float)O9_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O9_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 55 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 55 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O10_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O10_SW[y][k][x]);
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


	fprintf(o_stream,"%s","MaxPooling2D : [[");
for (k = 0; k < 27 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 27 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O11_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O11_SW[y][k][x]);
		}
		if(x != 27 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 27 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 27 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 27 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O12_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O12_SW[y][k][x]);
		}
		if(x != 27 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 27 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 27 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 27 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O13_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O13_SW[y][k][x]);
		}
		if(x != 27 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 27 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 27 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 27 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O14_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O14_SW[y][k][x]);
		}
		if(x != 27 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 27 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 27 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 27 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O15_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O15_SW[y][k][x]);
		}
		if(x != 27 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 27 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 27 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 27 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 32 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O16_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O16_SW[y][k][x]);
		}
		if(x != 27 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 27 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 27 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 27 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O17_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O17_SW[y][k][x]);
		}
		if(x != 27 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 27 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 27 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 27 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O18_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O18_SW[y][k][x]);
		}
		if(x != 27 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 27 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 27 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 27 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O19_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O19_SW[y][k][x]);
		}
		if(x != 27 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 27 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","MaxPooling2D : [[");
for (k = 0; k < 13 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 13 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O20_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O20_SW[y][k][x]);
		}
		if(x != 13 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 13 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 13 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 13 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 48 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O21_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O21_SW[y][k][x]);
		}
		if(x != 13 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 13 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 13 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 13 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O22_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O22_SW[y][k][x]);
		}
		if(x != 13 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 13 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 13 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 13 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O23_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O23_SW[y][k][x]);
		}
		if(x != 13 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 13 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 13 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 13 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O24_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O24_SW[y][k][x]);
		}
		if(x != 13 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 13 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 13 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 13 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 48 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O25_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O25_SW[y][k][x]);
		}
		if(x != 13 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 13 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 13 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 13 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O26_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O26_SW[y][k][x]);
		}
		if(x != 13 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 13 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 13 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 13 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 192 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O27_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O27_SW[y][k][x]);
		}
		if(x != 13 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 13 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 13 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 13 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O28_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O28_SW[y][k][x]);
		}
		if(x != 13 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 13 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 13 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 13 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O29_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O29_SW[y][k][x]);
		}
		if(x != 13 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 13 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 13 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 13 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O30_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O30_SW[y][k][x]);
		}
		if(x != 13 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 13 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 13 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 13 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O31_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O31_SW[y][k][x]);
		}
		if(x != 13 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 13 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 13 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 13 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O32_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O32_SW[y][k][x]);
		}
		if(x != 13 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 13 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 13 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 13 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O33_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O33_SW[y][k][x]);
		}
		if(x != 13 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 13 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 13 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 13 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O34_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O34_SW[y][k][x]);
		}
		if(x != 13 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 13 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 13 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 13 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O35_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O35_SW[y][k][x]);
		}
		if(x != 13 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 13 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Concatenate : [[");
for (k = 0; k < 13 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 13 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O36_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O36_SW[y][k][x]);
		}
		if(x != 13 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 13 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 13 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 13 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1000 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O37_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O37_SW[y][k][x]);
		}
		if(x != 13 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 13 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","AveragePooling2D : [[");
for (k = 0; k < 1 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 1 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1000 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O38_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O38_SW[y][k][x]);
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
for (k = 0; k <  1000 ; k++) {
	fprintf(o_stream,"%.6f ",(float)O39_SW[k]);
	fprintf(c_num,"%.6f ",(float)O39_SW[k]);
}
fprintf(o_stream,"%s","]]\n\n");


	fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k <  1000 ; k++) {
	fprintf(o_stream,"%.6f ",(float)O40_SW[k]);
	fprintf(c_num,"%.6f ",(float)O40_SW[k]);
}
fprintf(o_stream,"%s","]]\n\n");


	


    fclose(w_stream);
    fclose(i_stream);
    fclose(o_stream);
    fclose(c_num);

    return 0;
}
