#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <math.h>

using namespace std;

typedef int DATA_T;

// define functions of each layer
void SW_conv2d_1(DATA_T I[3][32][32], DATA_T O[64][32][32], DATA_T W[64][3][3][3], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(32 - 1) - 32 + 3)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<32; x++) {
			for (y = 0; y<32; y++) {
				ofm = B[m];
				for (k = 0; k<3; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 32 + p && y*1 + j < 32 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_activation_1(DATA_T I[64][32][32], DATA_T O[64][32][32]) {
	int m, x, y, i, j, k;
	for (m = 0; m<64; m++) {
		for (x = 0; x<32; x++) {
			for (y = 0; y<32; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_res0a_branch2a(DATA_T I[64][32][32], DATA_T O[64][16][16], DATA_T W[64][64][3][3], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (2 *(16 - 1) - 32 + 3)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<16; x++) {
			for (y = 0; y<16; y++) {
				ofm = B[m];
				for (k = 0; k<64; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*2 + i < 32 + p && y*2 + j < 32 + p && x*2 + i -p >= 0 && y*2 + j -p >= 0) {
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

void SW_conv2d_2(DATA_T I[64][16][16], DATA_T O[64][16][16], DATA_T W[64][64][3][3], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(16 - 1) - 16 + 3)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<16; x++) {
			for (y = 0; y<16; y++) {
				ofm = B[m];
				for (k = 0; k<64; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 16 + p && y*1 + j < 16 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_3(DATA_T I[64][32][32], DATA_T O[64][16][16], DATA_T W[64][64][1][1], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<64; m++) {
		for (x = 0; x<16; x++) {
			for (y = 0; y<16; y++) {
				ofm = B[m];
				for (k = 0; k<64; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 32 && y + j <= 32) {
								ifm = I[k][x*2 + i][y*2 + j];
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

void SW_activation_2(DATA_T I[64][16][16], DATA_T O[64][16][16]) {
	int m, x, y, i, j, k;
	for (m = 0; m<64; m++) {
		for (x = 0; x<16; x++) {
			for (y = 0; y<16; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_add_1(DATA_T I1[64][16][16], DATA_T I2[64][16][16], DATA_T O[64][16][16]) {
	int m, x, y;
	for (m = 0; m< 64; m++) {
		for (x = 0; x < 16; x++) {
			for(y = 0; y < 16; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_conv2d_4(DATA_T I[64][16][16], DATA_T O[64][16][16], DATA_T W[64][64][3][3], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(16 - 1) - 16 + 3)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<16; x++) {
			for (y = 0; y<16; y++) {
				ofm = B[m];
				for (k = 0; k<64; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 16 + p && y*1 + j < 16 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_activation_3(DATA_T I[64][16][16], DATA_T O[64][16][16]) {
	int m, x, y, i, j, k;
	for (m = 0; m<64; m++) {
		for (x = 0; x<16; x++) {
			for (y = 0; y<16; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_5(DATA_T I[64][16][16], DATA_T O[64][16][16], DATA_T W[64][64][3][3], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(16 - 1) - 16 + 3)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<16; x++) {
			for (y = 0; y<16; y++) {
				ofm = B[m];
				for (k = 0; k<64; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 16 + p && y*1 + j < 16 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_activation_4(DATA_T I[64][16][16], DATA_T O[64][16][16]) {
	int m, x, y, i, j, k;
	for (m = 0; m<64; m++) {
		for (x = 0; x<16; x++) {
			for (y = 0; y<16; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_add_2(DATA_T I1[64][16][16], DATA_T I2[64][16][16], DATA_T O[64][16][16]) {
	int m, x, y;
	for (m = 0; m< 64; m++) {
		for (x = 0; x < 16; x++) {
			for(y = 0; y < 16; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_conv2d_6(DATA_T I[64][16][16], DATA_T O[128][8][8], DATA_T W[128][64][3][3], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (2 *(8 - 1) - 16 + 3)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<8; x++) {
			for (y = 0; y<8; y++) {
				ofm = B[m];
				for (k = 0; k<64; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*2 + i < 16 + p && y*2 + j < 16 + p && x*2 + i -p >= 0 && y*2 + j -p >= 0) {
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

void SW_activation_5(DATA_T I[128][8][8], DATA_T O[128][8][8]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<8; x++) {
			for (y = 0; y<8; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_7(DATA_T I[128][8][8], DATA_T O[128][8][8], DATA_T W[128][128][3][3], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(8 - 1) - 8 + 3)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<8; x++) {
			for (y = 0; y<8; y++) {
				ofm = B[m];
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 8 + p && y*1 + j < 8 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_8(DATA_T I[64][16][16], DATA_T O[128][8][8], DATA_T W[128][64][1][1], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<8; x++) {
			for (y = 0; y<8; y++) {
				ofm = B[m];
				for (k = 0; k<64; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 16 && y + j <= 16) {
								ifm = I[k][x*2 + i][y*2 + j];
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

void SW_activation_6(DATA_T I[128][8][8], DATA_T O[128][8][8]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<8; x++) {
			for (y = 0; y<8; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_add_3(DATA_T I1[128][8][8], DATA_T I2[128][8][8], DATA_T O[128][8][8]) {
	int m, x, y;
	for (m = 0; m< 128; m++) {
		for (x = 0; x < 8; x++) {
			for(y = 0; y < 8; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_conv2d_9(DATA_T I[128][8][8], DATA_T O[128][8][8], DATA_T W[128][128][3][3], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(8 - 1) - 8 + 3)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<8; x++) {
			for (y = 0; y<8; y++) {
				ofm = B[m];
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 8 + p && y*1 + j < 8 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_activation_7(DATA_T I[128][8][8], DATA_T O[128][8][8]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<8; x++) {
			for (y = 0; y<8; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_10(DATA_T I[128][8][8], DATA_T O[128][8][8], DATA_T W[128][128][3][3], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(8 - 1) - 8 + 3)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<8; x++) {
			for (y = 0; y<8; y++) {
				ofm = B[m];
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 8 + p && y*1 + j < 8 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_activation_8(DATA_T I[128][8][8], DATA_T O[128][8][8]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<8; x++) {
			for (y = 0; y<8; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_add_4(DATA_T I1[128][8][8], DATA_T I2[128][8][8], DATA_T O[128][8][8]) {
	int m, x, y;
	for (m = 0; m< 128; m++) {
		for (x = 0; x < 8; x++) {
			for(y = 0; y < 8; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_conv2d_11(DATA_T I[128][8][8], DATA_T O[256][4][4], DATA_T W[256][128][3][3], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (2 *(4 - 1) - 8 + 3)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<4; x++) {
			for (y = 0; y<4; y++) {
				ofm = B[m];
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*2 + i < 8 + p && y*2 + j < 8 + p && x*2 + i -p >= 0 && y*2 + j -p >= 0) {
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

void SW_activation_9(DATA_T I[256][4][4], DATA_T O[256][4][4]) {
	int m, x, y, i, j, k;
	for (m = 0; m<256; m++) {
		for (x = 0; x<4; x++) {
			for (y = 0; y<4; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_12(DATA_T I[256][4][4], DATA_T O[256][4][4], DATA_T W[256][256][3][3], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(4 - 1) - 4 + 3)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<4; x++) {
			for (y = 0; y<4; y++) {
				ofm = B[m];
				for (k = 0; k<256; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 4 + p && y*1 + j < 4 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_13(DATA_T I[128][8][8], DATA_T O[256][4][4], DATA_T W[256][128][1][1], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<4; x++) {
			for (y = 0; y<4; y++) {
				ofm = B[m];
				for (k = 0; k<128; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 8 && y + j <= 8) {
								ifm = I[k][x*2 + i][y*2 + j];
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

void SW_activation_10(DATA_T I[256][4][4], DATA_T O[256][4][4]) {
	int m, x, y, i, j, k;
	for (m = 0; m<256; m++) {
		for (x = 0; x<4; x++) {
			for (y = 0; y<4; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_add_5(DATA_T I1[256][4][4], DATA_T I2[256][4][4], DATA_T O[256][4][4]) {
	int m, x, y;
	for (m = 0; m< 256; m++) {
		for (x = 0; x < 4; x++) {
			for(y = 0; y < 4; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_conv2d_14(DATA_T I[256][4][4], DATA_T O[256][4][4], DATA_T W[256][256][3][3], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(4 - 1) - 4 + 3)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<4; x++) {
			for (y = 0; y<4; y++) {
				ofm = B[m];
				for (k = 0; k<256; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 4 + p && y*1 + j < 4 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_activation_11(DATA_T I[256][4][4], DATA_T O[256][4][4]) {
	int m, x, y, i, j, k;
	for (m = 0; m<256; m++) {
		for (x = 0; x<4; x++) {
			for (y = 0; y<4; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_15(DATA_T I[256][4][4], DATA_T O[256][4][4], DATA_T W[256][256][3][3], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(4 - 1) - 4 + 3)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<4; x++) {
			for (y = 0; y<4; y++) {
				ofm = B[m];
				for (k = 0; k<256; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 4 + p && y*1 + j < 4 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_activation_12(DATA_T I[256][4][4], DATA_T O[256][4][4]) {
	int m, x, y, i, j, k;
	for (m = 0; m<256; m++) {
		for (x = 0; x<4; x++) {
			for (y = 0; y<4; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_add_6(DATA_T I1[256][4][4], DATA_T I2[256][4][4], DATA_T O[256][4][4]) {
	int m, x, y;
	for (m = 0; m< 256; m++) {
		for (x = 0; x < 4; x++) {
			for(y = 0; y < 4; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_conv2d_16(DATA_T I[256][4][4], DATA_T O[512][2][2], DATA_T W[512][256][3][3], DATA_T B[512]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (2 *(2 - 1) - 4 + 3)/2;
	for (m = 0; m<512; m++) {
		for (x = 0; x<2; x++) {
			for (y = 0; y<2; y++) {
				ofm = B[m];
				for (k = 0; k<256; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*2 + i < 4 + p && y*2 + j < 4 + p && x*2 + i -p >= 0 && y*2 + j -p >= 0) {
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

void SW_activation_13(DATA_T I[512][2][2], DATA_T O[512][2][2]) {
	int m, x, y, i, j, k;
	for (m = 0; m<512; m++) {
		for (x = 0; x<2; x++) {
			for (y = 0; y<2; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_17(DATA_T I[512][2][2], DATA_T O[512][2][2], DATA_T W[512][512][3][3], DATA_T B[512]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(2 - 1) - 2 + 3)/2;
	for (m = 0; m<512; m++) {
		for (x = 0; x<2; x++) {
			for (y = 0; y<2; y++) {
				ofm = B[m];
				for (k = 0; k<512; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 2 + p && y*1 + j < 2 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_conv2d_18(DATA_T I[256][4][4], DATA_T O[512][2][2], DATA_T W[512][256][1][1], DATA_T B[512]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<512; m++) {
		for (x = 0; x<2; x++) {
			for (y = 0; y<2; y++) {
				ofm = B[m];
				for (k = 0; k<256; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 4 && y + j <= 4) {
								ifm = I[k][x*2 + i][y*2 + j];
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

void SW_activation_14(DATA_T I[512][2][2], DATA_T O[512][2][2]) {
	int m, x, y, i, j, k;
	for (m = 0; m<512; m++) {
		for (x = 0; x<2; x++) {
			for (y = 0; y<2; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_add_7(DATA_T I1[512][2][2], DATA_T I2[512][2][2], DATA_T O[512][2][2]) {
	int m, x, y;
	for (m = 0; m< 512; m++) {
		for (x = 0; x < 2; x++) {
			for(y = 0; y < 2; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_conv2d_19(DATA_T I[512][2][2], DATA_T O[512][2][2], DATA_T W[512][512][3][3], DATA_T B[512]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(2 - 1) - 2 + 3)/2;
	for (m = 0; m<512; m++) {
		for (x = 0; x<2; x++) {
			for (y = 0; y<2; y++) {
				ofm = B[m];
				for (k = 0; k<512; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 2 + p && y*1 + j < 2 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_activation_15(DATA_T I[512][2][2], DATA_T O[512][2][2]) {
	int m, x, y, i, j, k;
	for (m = 0; m<512; m++) {
		for (x = 0; x<2; x++) {
			for (y = 0; y<2; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_conv2d_20(DATA_T I[512][2][2], DATA_T O[512][2][2], DATA_T W[512][512][3][3], DATA_T B[512]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(2 - 1) - 2 + 3)/2;
	for (m = 0; m<512; m++) {
		for (x = 0; x<2; x++) {
			for (y = 0; y<2; y++) {
				ofm = B[m];
				for (k = 0; k<512; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 2 + p && y*1 + j < 2 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_activation_16(DATA_T I[512][2][2], DATA_T O[512][2][2]) {
	int m, x, y, i, j, k;
	for (m = 0; m<512; m++) {
		for (x = 0; x<2; x++) {
			for (y = 0; y<2; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_add_8(DATA_T I1[512][2][2], DATA_T I2[512][2][2], DATA_T O[512][2][2]) {
	int m, x, y;
	for (m = 0; m< 512; m++) {
		for (x = 0; x < 2; x++) {
			for(y = 0; y < 2; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_activation_17(DATA_T I[512][2][2], DATA_T O[512][2][2]) {
	int m, x, y, i, j, k;
	for (m = 0; m<512; m++) {
		for (x = 0; x<2; x++) {
			for (y = 0; y<2; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_global_average_pooling2d_1(DATA_T I[512][2][2], DATA_T O[512]) {
	int m, x, y;
	double avg;
	int div = 2 * 2;
	for (m = 0; m < 512; m++){
		avg = 0;
		for (x = 0; x < 2; x++) {
			for (y = 0; y < 2; y++) {
				avg += I[m][x][y];
			}
		}
		O[m] = avg/div;
	}
}
void SW_dense_1(DATA_T I[512], DATA_T O[10], DATA_T W[10][512], DATA_T B[10])
{
    //Dense
	int m, c;
	DATA_T maximum = 0;

	for(m=0; m<10; m++){
        O[m] = B[m];
		for (c = 0; c < 512; c++){
			O[m] += W[m][c] * I[c];
        }
    }
    //Find max
    for (m = 0; m < 10; m++){
		if(maximum<O[m])
		    maximum=O[m];
    }
    //One hot key
    for (m = 0; m < 10; m++){
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
    DATA_T trash;

    // declare array variables of input, weight and bias (static variables)
    static DATA_T I[3][32][32];
	static DATA_T W1[64][3][3][3];
	static DATA_T B1[64];
	static DATA_T W3[64][64][3][3];
	static DATA_T B3[64];
	static DATA_T W4[64][64][3][3];
	static DATA_T B4[64];
	static DATA_T W5[64][64][1][1];
	static DATA_T B5[64];
	static DATA_T W8[64][64][3][3];
	static DATA_T B8[64];
	static DATA_T W10[64][64][3][3];
	static DATA_T B10[64];
	static DATA_T W13[128][64][3][3];
	static DATA_T B13[128];
	static DATA_T W15[128][128][3][3];
	static DATA_T B15[128];
	static DATA_T W16[128][64][1][1];
	static DATA_T B16[128];
	static DATA_T W19[128][128][3][3];
	static DATA_T B19[128];
	static DATA_T W21[128][128][3][3];
	static DATA_T B21[128];
	static DATA_T W24[256][128][3][3];
	static DATA_T B24[256];
	static DATA_T W26[256][256][3][3];
	static DATA_T B26[256];
	static DATA_T W27[256][128][1][1];
	static DATA_T B27[256];
	static DATA_T W30[256][256][3][3];
	static DATA_T B30[256];
	static DATA_T W32[256][256][3][3];
	static DATA_T B32[256];
	static DATA_T W35[512][256][3][3];
	static DATA_T B35[512];
	static DATA_T W37[512][512][3][3];
	static DATA_T B37[512];
	static DATA_T W38[512][256][1][1];
	static DATA_T B38[512];
	static DATA_T W41[512][512][3][3];
	static DATA_T B41[512];
	static DATA_T W43[512][512][3][3];
	static DATA_T B43[512];
	static DATA_T W48[10][512];
	static DATA_T B48[10];
	

    // declare array variables of output (static variables)
    static DATA_T O0_SW[3][32][32];
	static DATA_T O1_SW[64][32][32];
	static DATA_T O2_SW[64][32][32];
	static DATA_T O3_SW[64][16][16];
	static DATA_T O4_SW[64][16][16];
	static DATA_T O5_SW[64][16][16];
	static DATA_T O6_SW[64][16][16];
	static DATA_T O7_SW[64][16][16];
	static DATA_T O8_SW[64][16][16];
	static DATA_T O9_SW[64][16][16];
	static DATA_T O10_SW[64][16][16];
	static DATA_T O11_SW[64][16][16];
	static DATA_T O12_SW[64][16][16];
	static DATA_T O13_SW[128][8][8];
	static DATA_T O14_SW[128][8][8];
	static DATA_T O15_SW[128][8][8];
	static DATA_T O16_SW[128][8][8];
	static DATA_T O17_SW[128][8][8];
	static DATA_T O18_SW[128][8][8];
	static DATA_T O19_SW[128][8][8];
	static DATA_T O20_SW[128][8][8];
	static DATA_T O21_SW[128][8][8];
	static DATA_T O22_SW[128][8][8];
	static DATA_T O23_SW[128][8][8];
	static DATA_T O24_SW[256][4][4];
	static DATA_T O25_SW[256][4][4];
	static DATA_T O26_SW[256][4][4];
	static DATA_T O27_SW[256][4][4];
	static DATA_T O28_SW[256][4][4];
	static DATA_T O29_SW[256][4][4];
	static DATA_T O30_SW[256][4][4];
	static DATA_T O31_SW[256][4][4];
	static DATA_T O32_SW[256][4][4];
	static DATA_T O33_SW[256][4][4];
	static DATA_T O34_SW[256][4][4];
	static DATA_T O35_SW[512][2][2];
	static DATA_T O36_SW[512][2][2];
	static DATA_T O37_SW[512][2][2];
	static DATA_T O38_SW[512][2][2];
	static DATA_T O39_SW[512][2][2];
	static DATA_T O40_SW[512][2][2];
	static DATA_T O41_SW[512][2][2];
	static DATA_T O42_SW[512][2][2];
	static DATA_T O43_SW[512][2][2];
	static DATA_T O44_SW[512][2][2];
	static DATA_T O45_SW[512][2][2];
	static DATA_T O46_SW[512][2][2];
	static DATA_T O47_SW[512];
	static DATA_T O48_SW[10];
	

    FILE *w_stream = fopen(argv[1], "rb");
    if (w_stream == NULL) printf("weight file was not opened");
    FILE *i_stream = fopen(argv[2], "rb");
    if (i_stream == NULL) printf("input file was not opened");
    FILE *o_stream = fopen("./output/resnet18/output_value/c_output.txt", "w");
    if (o_stream == NULL) printf("Output file was not opened");
    FILE *c_num = fopen("./output/resnet18/output_value/c_output_num.txt", "w");
    if (c_num == NULL) printf("Output file was not opened");

    // initialize input, weight, bias variables using fread
    printf("[c_verifier.cpp]Start Initialzation\n");
    for (k = 0; k <  32 ; k++) {
	for (x = 0; x < 32 ; x++) {
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
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W1[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B1[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W3[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B3[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W4[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B4[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W5[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B5[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W8[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B8[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W10[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B10[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W13[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B13[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W15[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B15[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W16[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B16[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W19[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B19[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W21[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B21[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W24[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B24[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W26[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B26[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W27[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B27[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W30[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B30[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W32[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B32[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W35[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B35[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W37[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B37[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W38[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B38[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W41[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B41[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W43[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B43[m] = (DATA_T) trash;
}

	for (m = 0; m <  512 ; m++) {
	for (k = 0; k < 10 ; k++) {
		fread(&trash, sizeof(DATA_T), 1, w_stream);
        W48[k][m] = (DATA_T) trash;
	}
}

for (m = 0; m < 10 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B48[m] = (DATA_T) trash;
}

	
    printf("[c_verifier.cpp]Finish Initialization\n");

    // call function of each layer based on csv file which containts layer information
    printf("[c_verifier.cpp]InputLayer\n");
	printf("[c_verifier.cpp]Calculate Conv2D1\n");
	SW_conv2d_1(O0_SW,O1_SW,W1,B1);
	printf("[c_verifier.cpp]Calculate Activation2\n");
	SW_activation_1(O1_SW,O2_SW);
	printf("[c_verifier.cpp]Calculate Conv2D3\n");
	SW_res0a_branch2a(O2_SW,O3_SW,W3,B3);
	printf("[c_verifier.cpp]Calculate Conv2D4\n");
	SW_conv2d_2(O3_SW,O4_SW,W4,B4);
	printf("[c_verifier.cpp]Calculate Conv2D5\n");
	SW_conv2d_3(O2_SW,O5_SW,W5,B5);
	printf("[c_verifier.cpp]Calculate Activation6\n");
	SW_activation_2(O4_SW,O6_SW);
	printf("[c_verifier.cpp]Calculate Add7\n");
	SW_add_1(O5_SW,O6_SW,O7_SW);
	printf("[c_verifier.cpp]Calculate Conv2D8\n");
	SW_conv2d_4(O7_SW,O8_SW,W8,B8);
	printf("[c_verifier.cpp]Calculate Activation9\n");
	SW_activation_3(O8_SW,O9_SW);
	printf("[c_verifier.cpp]Calculate Conv2D10\n");
	SW_conv2d_5(O9_SW,O10_SW,W10,B10);
	printf("[c_verifier.cpp]Calculate Activation11\n");
	SW_activation_4(O10_SW,O11_SW);
	printf("[c_verifier.cpp]Calculate Add12\n");
	SW_add_2(O7_SW,O11_SW,O12_SW);
	printf("[c_verifier.cpp]Calculate Conv2D13\n");
	SW_conv2d_6(O12_SW,O13_SW,W13,B13);
	printf("[c_verifier.cpp]Calculate Activation14\n");
	SW_activation_5(O13_SW,O14_SW);
	printf("[c_verifier.cpp]Calculate Conv2D15\n");
	SW_conv2d_7(O14_SW,O15_SW,W15,B15);
	printf("[c_verifier.cpp]Calculate Conv2D16\n");
	SW_conv2d_8(O12_SW,O16_SW,W16,B16);
	printf("[c_verifier.cpp]Calculate Activation17\n");
	SW_activation_6(O15_SW,O17_SW);
	printf("[c_verifier.cpp]Calculate Add18\n");
	SW_add_3(O16_SW,O17_SW,O18_SW);
	printf("[c_verifier.cpp]Calculate Conv2D19\n");
	SW_conv2d_9(O18_SW,O19_SW,W19,B19);
	printf("[c_verifier.cpp]Calculate Activation20\n");
	SW_activation_7(O19_SW,O20_SW);
	printf("[c_verifier.cpp]Calculate Conv2D21\n");
	SW_conv2d_10(O20_SW,O21_SW,W21,B21);
	printf("[c_verifier.cpp]Calculate Activation22\n");
	SW_activation_8(O21_SW,O22_SW);
	printf("[c_verifier.cpp]Calculate Add23\n");
	SW_add_4(O18_SW,O22_SW,O23_SW);
	printf("[c_verifier.cpp]Calculate Conv2D24\n");
	SW_conv2d_11(O23_SW,O24_SW,W24,B24);
	printf("[c_verifier.cpp]Calculate Activation25\n");
	SW_activation_9(O24_SW,O25_SW);
	printf("[c_verifier.cpp]Calculate Conv2D26\n");
	SW_conv2d_12(O25_SW,O26_SW,W26,B26);
	printf("[c_verifier.cpp]Calculate Conv2D27\n");
	SW_conv2d_13(O23_SW,O27_SW,W27,B27);
	printf("[c_verifier.cpp]Calculate Activation28\n");
	SW_activation_10(O26_SW,O28_SW);
	printf("[c_verifier.cpp]Calculate Add29\n");
	SW_add_5(O27_SW,O28_SW,O29_SW);
	printf("[c_verifier.cpp]Calculate Conv2D30\n");
	SW_conv2d_14(O29_SW,O30_SW,W30,B30);
	printf("[c_verifier.cpp]Calculate Activation31\n");
	SW_activation_11(O30_SW,O31_SW);
	printf("[c_verifier.cpp]Calculate Conv2D32\n");
	SW_conv2d_15(O31_SW,O32_SW,W32,B32);
	printf("[c_verifier.cpp]Calculate Activation33\n");
	SW_activation_12(O32_SW,O33_SW);
	printf("[c_verifier.cpp]Calculate Add34\n");
	SW_add_6(O29_SW,O33_SW,O34_SW);
	printf("[c_verifier.cpp]Calculate Conv2D35\n");
	SW_conv2d_16(O34_SW,O35_SW,W35,B35);
	printf("[c_verifier.cpp]Calculate Activation36\n");
	SW_activation_13(O35_SW,O36_SW);
	printf("[c_verifier.cpp]Calculate Conv2D37\n");
	SW_conv2d_17(O36_SW,O37_SW,W37,B37);
	printf("[c_verifier.cpp]Calculate Conv2D38\n");
	SW_conv2d_18(O34_SW,O38_SW,W38,B38);
	printf("[c_verifier.cpp]Calculate Activation39\n");
	SW_activation_14(O37_SW,O39_SW);
	printf("[c_verifier.cpp]Calculate Add40\n");
	SW_add_7(O38_SW,O39_SW,O40_SW);
	printf("[c_verifier.cpp]Calculate Conv2D41\n");
	SW_conv2d_19(O40_SW,O41_SW,W41,B41);
	printf("[c_verifier.cpp]Calculate Activation42\n");
	SW_activation_15(O41_SW,O42_SW);
	printf("[c_verifier.cpp]Calculate Conv2D43\n");
	SW_conv2d_20(O42_SW,O43_SW,W43,B43);
	printf("[c_verifier.cpp]Calculate Activation44\n");
	SW_activation_16(O43_SW,O44_SW);
	printf("[c_verifier.cpp]Calculate Add45\n");
	SW_add_8(O40_SW,O44_SW,O45_SW);
	printf("[c_verifier.cpp]Calculate Activation46\n");
	SW_activation_17(O45_SW,O46_SW);
	printf("[c_verifier.cpp]Calculate GlobalAveragePooling2D47\n");
	SW_global_average_pooling2d_1(O46_SW,O47_SW);
	printf("[c_verifier.cpp]Calculate Dense48\n");
	SW_dense_1(O47_SW,O48_SW,W48,B48);
	

    // print each element of output variables
    printf("[c_verifier.cpp]Print Result\n");


    fprintf(o_stream,"%s","InputLayer : [[");
for (k = 0; k < 32 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 32 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 3 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O0_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O0_SW[y][k][x]);
		}
		if(x != 32 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 32 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 32 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 32 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O1_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O1_SW[y][k][x]);
		}
		if(x != 32 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 32 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 32 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 32 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O2_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O2_SW[y][k][x]);
		}
		if(x != 32 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 32 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 16 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 16 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O3_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O3_SW[y][k][x]);
		}
		if(x != 16 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 16 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 16 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 16 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O4_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O4_SW[y][k][x]);
		}
		if(x != 16 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 16 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 16 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 16 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O5_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O5_SW[y][k][x]);
		}
		if(x != 16 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 16 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 16 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 16 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O6_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O6_SW[y][k][x]);
		}
		if(x != 16 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 16 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 16 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 16 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O7_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O7_SW[y][k][x]);
		}
		if(x != 16 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 16 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 16 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 16 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O8_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O8_SW[y][k][x]);
		}
		if(x != 16 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 16 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 16 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 16 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O9_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O9_SW[y][k][x]);
		}
		if(x != 16 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 16 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 16 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 16 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O10_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O10_SW[y][k][x]);
		}
		if(x != 16 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 16 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 16 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 16 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O11_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O11_SW[y][k][x]);
		}
		if(x != 16 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 16 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 16 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 16 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O12_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O12_SW[y][k][x]);
		}
		if(x != 16 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 16 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 8 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 8 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O13_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O13_SW[y][k][x]);
		}
		if(x != 8 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 8 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 8 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 8 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O14_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O14_SW[y][k][x]);
		}
		if(x != 8 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 8 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 8 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 8 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O15_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O15_SW[y][k][x]);
		}
		if(x != 8 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 8 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 8 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 8 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O16_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O16_SW[y][k][x]);
		}
		if(x != 8 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 8 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 8 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 8 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O17_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O17_SW[y][k][x]);
		}
		if(x != 8 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 8 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 8 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 8 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O18_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O18_SW[y][k][x]);
		}
		if(x != 8 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 8 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 8 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 8 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O19_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O19_SW[y][k][x]);
		}
		if(x != 8 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 8 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 8 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 8 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O20_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O20_SW[y][k][x]);
		}
		if(x != 8 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 8 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 8 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 8 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O21_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O21_SW[y][k][x]);
		}
		if(x != 8 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 8 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 8 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 8 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O22_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O22_SW[y][k][x]);
		}
		if(x != 8 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 8 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 8 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 8 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O23_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O23_SW[y][k][x]);
		}
		if(x != 8 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 8 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O24_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O24_SW[y][k][x]);
		}
		if(x != 4 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 4 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O25_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O25_SW[y][k][x]);
		}
		if(x != 4 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 4 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O26_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O26_SW[y][k][x]);
		}
		if(x != 4 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 4 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O27_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O27_SW[y][k][x]);
		}
		if(x != 4 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 4 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O28_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O28_SW[y][k][x]);
		}
		if(x != 4 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 4 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O29_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O29_SW[y][k][x]);
		}
		if(x != 4 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 4 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O30_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O30_SW[y][k][x]);
		}
		if(x != 4 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 4 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O31_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O31_SW[y][k][x]);
		}
		if(x != 4 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 4 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O32_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O32_SW[y][k][x]);
		}
		if(x != 4 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 4 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O33_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O33_SW[y][k][x]);
		}
		if(x != 4 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 4 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O34_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O34_SW[y][k][x]);
		}
		if(x != 4 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 4 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 2 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 2 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O35_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O35_SW[y][k][x]);
		}
		if(x != 2 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 2 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 2 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 2 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O36_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O36_SW[y][k][x]);
		}
		if(x != 2 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 2 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 2 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 2 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O37_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O37_SW[y][k][x]);
		}
		if(x != 2 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 2 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 2 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 2 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O38_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O38_SW[y][k][x]);
		}
		if(x != 2 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 2 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 2 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 2 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O39_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O39_SW[y][k][x]);
		}
		if(x != 2 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 2 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 2 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 2 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O40_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O40_SW[y][k][x]);
		}
		if(x != 2 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 2 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 2 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 2 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O41_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O41_SW[y][k][x]);
		}
		if(x != 2 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 2 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 2 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 2 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O42_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O42_SW[y][k][x]);
		}
		if(x != 2 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 2 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 2 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 2 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O43_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O43_SW[y][k][x]);
		}
		if(x != 2 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 2 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 2 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 2 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O44_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O44_SW[y][k][x]);
		}
		if(x != 2 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 2 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 2 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 2 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O45_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O45_SW[y][k][x]);
		}
		if(x != 2 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 2 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 2 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 2 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",(float)O46_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(float)O46_SW[y][k][x]);
		}
		if(x != 2 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 2 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","GlobalAveragePooling2D : [[");
for (k = 0; k <  512 ; k++) {
	fprintf(o_stream,"%.6f ",(float)O47_SW[k]);
	fprintf(c_num,"%.6f ",(float)O47_SW[k]);
}
fprintf(o_stream,"%s","]]\n\n");


	fprintf(o_stream,"%s","Dense : [[");
for (k = 0; k <  10 ; k++) {
	fprintf(o_stream,"%.6f ",(float)O48_SW[k]);
	fprintf(c_num,"%.6f ",(float)O48_SW[k]);
}
fprintf(o_stream,"%s","]]\n\n");


	


    fclose(w_stream);
    fclose(i_stream);
    fclose(o_stream);
    fclose(c_num);

    return 0;
}
