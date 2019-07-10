#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <math.h>

using namespace std;

typedef int DATA_IN;
typedef double DATA_T;

// define functions of each layer
void SW_conv2d_1(DATA_T I[3][32][32], DATA_T O[64][32][32], DATA_T W[64][3][3][3], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<64; m++) {
		for (x = 0; x<32; x++) {
			for (y = 0; y<32; y++) {
				ofm = B[m];
				for (k = 0; k<3; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 32 + 1 && y*1 + j < 32 + 1 && x*1 + i -1 >= 0 && y*1 + j -1 >= 0) {
                                    				ifm = I[k][x*1 + i - 1][y*1 + j -1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
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
	for (m = 0; m<64; m++) {
		for (x = 0; x<16; x++) {
			for (y = 0; y<16; y++) {
				ofm = B[m];
				for (k = 0; k<64; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*2 + i < 32 + 0 && y*2 + j < 32 + 0 && x*2 + i -0 >= 0 && y*2 + j -0 >= 0) {
                                    				ifm = I[k][x*2 + i - 0][y*2 + j -0];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_conv2d_2(DATA_T I[64][16][16], DATA_T O[64][16][16], DATA_T W[64][64][3][3], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<64; m++) {
		for (x = 0; x<16; x++) {
			for (y = 0; y<16; y++) {
				ofm = B[m];
				for (k = 0; k<64; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 16 + 1 && y*1 + j < 16 + 1 && x*1 + i -1 >= 0 && y*1 + j -1 >= 0) {
                                    				ifm = I[k][x*1 + i - 1][y*1 + j -1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
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
	for (m = 0; m<64; m++) {
		for (x = 0; x<16; x++) {
			for (y = 0; y<16; y++) {
				ofm = B[m];
				for (k = 0; k<64; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 16 + 1 && y*1 + j < 16 + 1 && x*1 + i -1 >= 0 && y*1 + j -1 >= 0) {
                                    				ifm = I[k][x*1 + i - 1][y*1 + j -1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
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
	for (m = 0; m<64; m++) {
		for (x = 0; x<16; x++) {
			for (y = 0; y<16; y++) {
				ofm = B[m];
				for (k = 0; k<64; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 16 + 1 && y*1 + j < 16 + 1 && x*1 + i -1 >= 0 && y*1 + j -1 >= 0) {
                                    				ifm = I[k][x*1 + i - 1][y*1 + j -1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
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
void SW_conv2d_6(DATA_T I[64][16][16], DATA_T O[64][16][16], DATA_T W[64][64][3][3], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<64; m++) {
		for (x = 0; x<16; x++) {
			for (y = 0; y<16; y++) {
				ofm = B[m];
				for (k = 0; k<64; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 16 + 1 && y*1 + j < 16 + 1 && x*1 + i -1 >= 0 && y*1 + j -1 >= 0) {
                                    				ifm = I[k][x*1 + i - 1][y*1 + j -1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_activation_5(DATA_T I[64][16][16], DATA_T O[64][16][16]) {
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

void SW_conv2d_7(DATA_T I[64][16][16], DATA_T O[64][16][16], DATA_T W[64][64][3][3], DATA_T B[64]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<64; m++) {
		for (x = 0; x<16; x++) {
			for (y = 0; y<16; y++) {
				ofm = B[m];
				for (k = 0; k<64; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 16 + 1 && y*1 + j < 16 + 1 && x*1 + i -1 >= 0 && y*1 + j -1 >= 0) {
                                    				ifm = I[k][x*1 + i - 1][y*1 + j -1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_activation_6(DATA_T I[64][16][16], DATA_T O[64][16][16]) {
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

void SW_add_3(DATA_T I1[64][16][16], DATA_T I2[64][16][16], DATA_T O[64][16][16]) {
	int m, x, y;
	for (m = 0; m< 64; m++) {
		for (x = 0; x < 16; x++) {
			for(y = 0; y < 16; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_conv2d_8(DATA_T I[64][16][16], DATA_T O[128][8][8], DATA_T W[128][64][3][3], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<8; x++) {
			for (y = 0; y<8; y++) {
				ofm = B[m];
				for (k = 0; k<64; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*2 + i < 16 + 0 && y*2 + j < 16 + 0 && x*2 + i -0 >= 0 && y*2 + j -0 >= 0) {
                                    				ifm = I[k][x*2 + i - 0][y*2 + j -0];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
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

void SW_conv2d_9(DATA_T I[128][8][8], DATA_T O[128][8][8], DATA_T W[128][128][3][3], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<8; x++) {
			for (y = 0; y<8; y++) {
				ofm = B[m];
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 8 + 1 && y*1 + j < 8 + 1 && x*1 + i -1 >= 0 && y*1 + j -1 >= 0) {
                                    				ifm = I[k][x*1 + i - 1][y*1 + j -1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_conv2d_10(DATA_T I[64][16][16], DATA_T O[128][8][8], DATA_T W[128][64][1][1], DATA_T B[128]) {
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
void SW_conv2d_11(DATA_T I[128][8][8], DATA_T O[128][8][8], DATA_T W[128][128][3][3], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<8; x++) {
			for (y = 0; y<8; y++) {
				ofm = B[m];
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 8 + 1 && y*1 + j < 8 + 1 && x*1 + i -1 >= 0 && y*1 + j -1 >= 0) {
                                    				ifm = I[k][x*1 + i - 1][y*1 + j -1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_activation_9(DATA_T I[128][8][8], DATA_T O[128][8][8]) {
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

void SW_conv2d_12(DATA_T I[128][8][8], DATA_T O[128][8][8], DATA_T W[128][128][3][3], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<8; x++) {
			for (y = 0; y<8; y++) {
				ofm = B[m];
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 8 + 1 && y*1 + j < 8 + 1 && x*1 + i -1 >= 0 && y*1 + j -1 >= 0) {
                                    				ifm = I[k][x*1 + i - 1][y*1 + j -1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_activation_10(DATA_T I[128][8][8], DATA_T O[128][8][8]) {
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

void SW_add_5(DATA_T I1[128][8][8], DATA_T I2[128][8][8], DATA_T O[128][8][8]) {
	int m, x, y;
	for (m = 0; m< 128; m++) {
		for (x = 0; x < 8; x++) {
			for(y = 0; y < 8; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_conv2d_13(DATA_T I[128][8][8], DATA_T O[128][8][8], DATA_T W[128][128][3][3], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<8; x++) {
			for (y = 0; y<8; y++) {
				ofm = B[m];
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 8 + 1 && y*1 + j < 8 + 1 && x*1 + i -1 >= 0 && y*1 + j -1 >= 0) {
                                    				ifm = I[k][x*1 + i - 1][y*1 + j -1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_activation_11(DATA_T I[128][8][8], DATA_T O[128][8][8]) {
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

void SW_conv2d_14(DATA_T I[128][8][8], DATA_T O[128][8][8], DATA_T W[128][128][3][3], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<8; x++) {
			for (y = 0; y<8; y++) {
				ofm = B[m];
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 8 + 1 && y*1 + j < 8 + 1 && x*1 + i -1 >= 0 && y*1 + j -1 >= 0) {
                                    				ifm = I[k][x*1 + i - 1][y*1 + j -1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_activation_12(DATA_T I[128][8][8], DATA_T O[128][8][8]) {
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

void SW_add_6(DATA_T I1[128][8][8], DATA_T I2[128][8][8], DATA_T O[128][8][8]) {
	int m, x, y;
	for (m = 0; m< 128; m++) {
		for (x = 0; x < 8; x++) {
			for(y = 0; y < 8; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_conv2d_15(DATA_T I[128][8][8], DATA_T O[128][8][8], DATA_T W[128][128][3][3], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<8; x++) {
			for (y = 0; y<8; y++) {
				ofm = B[m];
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 8 + 1 && y*1 + j < 8 + 1 && x*1 + i -1 >= 0 && y*1 + j -1 >= 0) {
                                    				ifm = I[k][x*1 + i - 1][y*1 + j -1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_activation_13(DATA_T I[128][8][8], DATA_T O[128][8][8]) {
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

void SW_conv2d_16(DATA_T I[128][8][8], DATA_T O[128][8][8], DATA_T W[128][128][3][3], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<8; x++) {
			for (y = 0; y<8; y++) {
				ofm = B[m];
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 8 + 1 && y*1 + j < 8 + 1 && x*1 + i -1 >= 0 && y*1 + j -1 >= 0) {
                                    				ifm = I[k][x*1 + i - 1][y*1 + j -1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_activation_14(DATA_T I[128][8][8], DATA_T O[128][8][8]) {
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

void SW_add_7(DATA_T I1[128][8][8], DATA_T I2[128][8][8], DATA_T O[128][8][8]) {
	int m, x, y;
	for (m = 0; m< 128; m++) {
		for (x = 0; x < 8; x++) {
			for(y = 0; y < 8; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_conv2d_17(DATA_T I[128][8][8], DATA_T O[256][4][4], DATA_T W[256][128][3][3], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<4; x++) {
			for (y = 0; y<4; y++) {
				ofm = B[m];
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*2 + i < 8 + 0 && y*2 + j < 8 + 0 && x*2 + i -0 >= 0 && y*2 + j -0 >= 0) {
                                    				ifm = I[k][x*2 + i - 0][y*2 + j -0];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_activation_15(DATA_T I[256][4][4], DATA_T O[256][4][4]) {
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

void SW_conv2d_18(DATA_T I[256][4][4], DATA_T O[256][4][4], DATA_T W[256][256][3][3], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<4; x++) {
			for (y = 0; y<4; y++) {
				ofm = B[m];
				for (k = 0; k<256; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 4 + 1 && y*1 + j < 4 + 1 && x*1 + i -1 >= 0 && y*1 + j -1 >= 0) {
                                    				ifm = I[k][x*1 + i - 1][y*1 + j -1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_conv2d_19(DATA_T I[128][8][8], DATA_T O[256][4][4], DATA_T W[256][128][1][1], DATA_T B[256]) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_activation_16(DATA_T I[256][4][4], DATA_T O[256][4][4]) {
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

void SW_add_8(DATA_T I1[256][4][4], DATA_T I2[256][4][4], DATA_T O[256][4][4]) {
	int m, x, y;
	for (m = 0; m< 256; m++) {
		for (x = 0; x < 4; x++) {
			for(y = 0; y < 4; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_conv2d_20(DATA_T I[256][4][4], DATA_T O[256][4][4], DATA_T W[256][256][3][3], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<4; x++) {
			for (y = 0; y<4; y++) {
				ofm = B[m];
				for (k = 0; k<256; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 4 + 1 && y*1 + j < 4 + 1 && x*1 + i -1 >= 0 && y*1 + j -1 >= 0) {
                                    				ifm = I[k][x*1 + i - 1][y*1 + j -1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_activation_17(DATA_T I[256][4][4], DATA_T O[256][4][4]) {
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

void SW_conv2d_21(DATA_T I[256][4][4], DATA_T O[256][4][4], DATA_T W[256][256][3][3], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<4; x++) {
			for (y = 0; y<4; y++) {
				ofm = B[m];
				for (k = 0; k<256; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 4 + 1 && y*1 + j < 4 + 1 && x*1 + i -1 >= 0 && y*1 + j -1 >= 0) {
                                    				ifm = I[k][x*1 + i - 1][y*1 + j -1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_activation_18(DATA_T I[256][4][4], DATA_T O[256][4][4]) {
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

void SW_add_9(DATA_T I1[256][4][4], DATA_T I2[256][4][4], DATA_T O[256][4][4]) {
	int m, x, y;
	for (m = 0; m< 256; m++) {
		for (x = 0; x < 4; x++) {
			for(y = 0; y < 4; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_conv2d_22(DATA_T I[256][4][4], DATA_T O[256][4][4], DATA_T W[256][256][3][3], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<4; x++) {
			for (y = 0; y<4; y++) {
				ofm = B[m];
				for (k = 0; k<256; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 4 + 1 && y*1 + j < 4 + 1 && x*1 + i -1 >= 0 && y*1 + j -1 >= 0) {
                                    				ifm = I[k][x*1 + i - 1][y*1 + j -1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_activation_19(DATA_T I[256][4][4], DATA_T O[256][4][4]) {
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

void SW_conv2d_23(DATA_T I[256][4][4], DATA_T O[256][4][4], DATA_T W[256][256][3][3], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<4; x++) {
			for (y = 0; y<4; y++) {
				ofm = B[m];
				for (k = 0; k<256; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 4 + 1 && y*1 + j < 4 + 1 && x*1 + i -1 >= 0 && y*1 + j -1 >= 0) {
                                    				ifm = I[k][x*1 + i - 1][y*1 + j -1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_activation_20(DATA_T I[256][4][4], DATA_T O[256][4][4]) {
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

void SW_add_10(DATA_T I1[256][4][4], DATA_T I2[256][4][4], DATA_T O[256][4][4]) {
	int m, x, y;
	for (m = 0; m< 256; m++) {
		for (x = 0; x < 4; x++) {
			for(y = 0; y < 4; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_conv2d_24(DATA_T I[256][4][4], DATA_T O[256][4][4], DATA_T W[256][256][3][3], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<4; x++) {
			for (y = 0; y<4; y++) {
				ofm = B[m];
				for (k = 0; k<256; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 4 + 1 && y*1 + j < 4 + 1 && x*1 + i -1 >= 0 && y*1 + j -1 >= 0) {
                                    				ifm = I[k][x*1 + i - 1][y*1 + j -1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_activation_21(DATA_T I[256][4][4], DATA_T O[256][4][4]) {
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

void SW_conv2d_25(DATA_T I[256][4][4], DATA_T O[256][4][4], DATA_T W[256][256][3][3], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<4; x++) {
			for (y = 0; y<4; y++) {
				ofm = B[m];
				for (k = 0; k<256; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 4 + 1 && y*1 + j < 4 + 1 && x*1 + i -1 >= 0 && y*1 + j -1 >= 0) {
                                    				ifm = I[k][x*1 + i - 1][y*1 + j -1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_activation_22(DATA_T I[256][4][4], DATA_T O[256][4][4]) {
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

void SW_add_11(DATA_T I1[256][4][4], DATA_T I2[256][4][4], DATA_T O[256][4][4]) {
	int m, x, y;
	for (m = 0; m< 256; m++) {
		for (x = 0; x < 4; x++) {
			for(y = 0; y < 4; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_conv2d_26(DATA_T I[256][4][4], DATA_T O[256][4][4], DATA_T W[256][256][3][3], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<4; x++) {
			for (y = 0; y<4; y++) {
				ofm = B[m];
				for (k = 0; k<256; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 4 + 1 && y*1 + j < 4 + 1 && x*1 + i -1 >= 0 && y*1 + j -1 >= 0) {
                                    				ifm = I[k][x*1 + i - 1][y*1 + j -1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_activation_23(DATA_T I[256][4][4], DATA_T O[256][4][4]) {
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

void SW_conv2d_27(DATA_T I[256][4][4], DATA_T O[256][4][4], DATA_T W[256][256][3][3], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<4; x++) {
			for (y = 0; y<4; y++) {
				ofm = B[m];
				for (k = 0; k<256; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 4 + 1 && y*1 + j < 4 + 1 && x*1 + i -1 >= 0 && y*1 + j -1 >= 0) {
                                    				ifm = I[k][x*1 + i - 1][y*1 + j -1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_activation_24(DATA_T I[256][4][4], DATA_T O[256][4][4]) {
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

void SW_add_12(DATA_T I1[256][4][4], DATA_T I2[256][4][4], DATA_T O[256][4][4]) {
	int m, x, y;
	for (m = 0; m< 256; m++) {
		for (x = 0; x < 4; x++) {
			for(y = 0; y < 4; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_conv2d_28(DATA_T I[256][4][4], DATA_T O[256][4][4], DATA_T W[256][256][3][3], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<4; x++) {
			for (y = 0; y<4; y++) {
				ofm = B[m];
				for (k = 0; k<256; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 4 + 1 && y*1 + j < 4 + 1 && x*1 + i -1 >= 0 && y*1 + j -1 >= 0) {
                                    				ifm = I[k][x*1 + i - 1][y*1 + j -1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_activation_25(DATA_T I[256][4][4], DATA_T O[256][4][4]) {
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

void SW_conv2d_29(DATA_T I[256][4][4], DATA_T O[256][4][4], DATA_T W[256][256][3][3], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<4; x++) {
			for (y = 0; y<4; y++) {
				ofm = B[m];
				for (k = 0; k<256; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 4 + 1 && y*1 + j < 4 + 1 && x*1 + i -1 >= 0 && y*1 + j -1 >= 0) {
                                    				ifm = I[k][x*1 + i - 1][y*1 + j -1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_activation_26(DATA_T I[256][4][4], DATA_T O[256][4][4]) {
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

void SW_add_13(DATA_T I1[256][4][4], DATA_T I2[256][4][4], DATA_T O[256][4][4]) {
	int m, x, y;
	for (m = 0; m< 256; m++) {
		for (x = 0; x < 4; x++) {
			for(y = 0; y < 4; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_conv2d_30(DATA_T I[256][4][4], DATA_T O[512][2][2], DATA_T W[512][256][3][3], DATA_T B[512]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<512; m++) {
		for (x = 0; x<2; x++) {
			for (y = 0; y<2; y++) {
				ofm = B[m];
				for (k = 0; k<256; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*2 + i < 4 + 0 && y*2 + j < 4 + 0 && x*2 + i -0 >= 0 && y*2 + j -0 >= 0) {
                                    				ifm = I[k][x*2 + i - 0][y*2 + j -0];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_activation_27(DATA_T I[512][2][2], DATA_T O[512][2][2]) {
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

void SW_conv2d_31(DATA_T I[512][2][2], DATA_T O[512][2][2], DATA_T W[512][512][3][3], DATA_T B[512]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<512; m++) {
		for (x = 0; x<2; x++) {
			for (y = 0; y<2; y++) {
				ofm = B[m];
				for (k = 0; k<512; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 2 + 1 && y*1 + j < 2 + 1 && x*1 + i -1 >= 0 && y*1 + j -1 >= 0) {
                                    				ifm = I[k][x*1 + i - 1][y*1 + j -1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_conv2d_32(DATA_T I[256][4][4], DATA_T O[512][2][2], DATA_T W[512][256][1][1], DATA_T B[512]) {
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
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_activation_28(DATA_T I[512][2][2], DATA_T O[512][2][2]) {
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

void SW_add_14(DATA_T I1[512][2][2], DATA_T I2[512][2][2], DATA_T O[512][2][2]) {
	int m, x, y;
	for (m = 0; m< 512; m++) {
		for (x = 0; x < 2; x++) {
			for(y = 0; y < 2; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_conv2d_33(DATA_T I[512][2][2], DATA_T O[512][2][2], DATA_T W[512][512][3][3], DATA_T B[512]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<512; m++) {
		for (x = 0; x<2; x++) {
			for (y = 0; y<2; y++) {
				ofm = B[m];
				for (k = 0; k<512; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 2 + 1 && y*1 + j < 2 + 1 && x*1 + i -1 >= 0 && y*1 + j -1 >= 0) {
                                    				ifm = I[k][x*1 + i - 1][y*1 + j -1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_activation_29(DATA_T I[512][2][2], DATA_T O[512][2][2]) {
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

void SW_conv2d_34(DATA_T I[512][2][2], DATA_T O[512][2][2], DATA_T W[512][512][3][3], DATA_T B[512]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<512; m++) {
		for (x = 0; x<2; x++) {
			for (y = 0; y<2; y++) {
				ofm = B[m];
				for (k = 0; k<512; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 2 + 1 && y*1 + j < 2 + 1 && x*1 + i -1 >= 0 && y*1 + j -1 >= 0) {
                                    				ifm = I[k][x*1 + i - 1][y*1 + j -1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_activation_30(DATA_T I[512][2][2], DATA_T O[512][2][2]) {
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

void SW_add_15(DATA_T I1[512][2][2], DATA_T I2[512][2][2], DATA_T O[512][2][2]) {
	int m, x, y;
	for (m = 0; m< 512; m++) {
		for (x = 0; x < 2; x++) {
			for(y = 0; y < 2; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_conv2d_35(DATA_T I[512][2][2], DATA_T O[512][2][2], DATA_T W[512][512][3][3], DATA_T B[512]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<512; m++) {
		for (x = 0; x<2; x++) {
			for (y = 0; y<2; y++) {
				ofm = B[m];
				for (k = 0; k<512; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 2 + 1 && y*1 + j < 2 + 1 && x*1 + i -1 >= 0 && y*1 + j -1 >= 0) {
                                    				ifm = I[k][x*1 + i - 1][y*1 + j -1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_activation_31(DATA_T I[512][2][2], DATA_T O[512][2][2]) {
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

void SW_conv2d_36(DATA_T I[512][2][2], DATA_T O[512][2][2], DATA_T W[512][512][3][3], DATA_T B[512]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<512; m++) {
		for (x = 0; x<2; x++) {
			for (y = 0; y<2; y++) {
				ofm = B[m];
				for (k = 0; k<512; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 2 + 1 && y*1 + j < 2 + 1 && x*1 + i -1 >= 0 && y*1 + j -1 >= 0) {
                                    				ifm = I[k][x*1 + i - 1][y*1 + j -1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_activation_32(DATA_T I[512][2][2], DATA_T O[512][2][2]) {
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

void SW_add_16(DATA_T I1[512][2][2], DATA_T I2[512][2][2], DATA_T O[512][2][2]) {
	int m, x, y;
	for (m = 0; m< 512; m++) {
		for (x = 0; x < 2; x++) {
			for(y = 0; y < 2; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_activation_33(DATA_T I[512][2][2], DATA_T O[512][2][2]) {
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
	DATA_T avg;
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

    int m, x, y, i, j, k, l;
    DATA_IN trash;

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
	static DATA_T W13[64][64][3][3];
	static DATA_T B13[64];
	static DATA_T W15[64][64][3][3];
	static DATA_T B15[64];
	static DATA_T W18[128][64][3][3];
	static DATA_T B18[128];
	static DATA_T W20[128][128][3][3];
	static DATA_T B20[128];
	static DATA_T W21[128][64][1][1];
	static DATA_T B21[128];
	static DATA_T W24[128][128][3][3];
	static DATA_T B24[128];
	static DATA_T W26[128][128][3][3];
	static DATA_T B26[128];
	static DATA_T W29[128][128][3][3];
	static DATA_T B29[128];
	static DATA_T W31[128][128][3][3];
	static DATA_T B31[128];
	static DATA_T W34[128][128][3][3];
	static DATA_T B34[128];
	static DATA_T W36[128][128][3][3];
	static DATA_T B36[128];
	static DATA_T W39[256][128][3][3];
	static DATA_T B39[256];
	static DATA_T W41[256][256][3][3];
	static DATA_T B41[256];
	static DATA_T W42[256][128][1][1];
	static DATA_T B42[256];
	static DATA_T W45[256][256][3][3];
	static DATA_T B45[256];
	static DATA_T W47[256][256][3][3];
	static DATA_T B47[256];
	static DATA_T W50[256][256][3][3];
	static DATA_T B50[256];
	static DATA_T W52[256][256][3][3];
	static DATA_T B52[256];
	static DATA_T W55[256][256][3][3];
	static DATA_T B55[256];
	static DATA_T W57[256][256][3][3];
	static DATA_T B57[256];
	static DATA_T W60[256][256][3][3];
	static DATA_T B60[256];
	static DATA_T W62[256][256][3][3];
	static DATA_T B62[256];
	static DATA_T W65[256][256][3][3];
	static DATA_T B65[256];
	static DATA_T W67[256][256][3][3];
	static DATA_T B67[256];
	static DATA_T W70[512][256][3][3];
	static DATA_T B70[512];
	static DATA_T W72[512][512][3][3];
	static DATA_T B72[512];
	static DATA_T W73[512][256][1][1];
	static DATA_T B73[512];
	static DATA_T W76[512][512][3][3];
	static DATA_T B76[512];
	static DATA_T W78[512][512][3][3];
	static DATA_T B78[512];
	static DATA_T W81[512][512][3][3];
	static DATA_T B81[512];
	static DATA_T W83[512][512][3][3];
	static DATA_T B83[512];
	static DATA_T W88[10][512];
	static DATA_T B88[10];
	

    // declare array variables of output (static variables)
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
	static DATA_T O13_SW[64][16][16];
	static DATA_T O14_SW[64][16][16];
	static DATA_T O15_SW[64][16][16];
	static DATA_T O16_SW[64][16][16];
	static DATA_T O17_SW[64][16][16];
	static DATA_T O18_SW[128][8][8];
	static DATA_T O19_SW[128][8][8];
	static DATA_T O20_SW[128][8][8];
	static DATA_T O21_SW[128][8][8];
	static DATA_T O22_SW[128][8][8];
	static DATA_T O23_SW[128][8][8];
	static DATA_T O24_SW[128][8][8];
	static DATA_T O25_SW[128][8][8];
	static DATA_T O26_SW[128][8][8];
	static DATA_T O27_SW[128][8][8];
	static DATA_T O28_SW[128][8][8];
	static DATA_T O29_SW[128][8][8];
	static DATA_T O30_SW[128][8][8];
	static DATA_T O31_SW[128][8][8];
	static DATA_T O32_SW[128][8][8];
	static DATA_T O33_SW[128][8][8];
	static DATA_T O34_SW[128][8][8];
	static DATA_T O35_SW[128][8][8];
	static DATA_T O36_SW[128][8][8];
	static DATA_T O37_SW[128][8][8];
	static DATA_T O38_SW[128][8][8];
	static DATA_T O39_SW[256][4][4];
	static DATA_T O40_SW[256][4][4];
	static DATA_T O41_SW[256][4][4];
	static DATA_T O42_SW[256][4][4];
	static DATA_T O43_SW[256][4][4];
	static DATA_T O44_SW[256][4][4];
	static DATA_T O45_SW[256][4][4];
	static DATA_T O46_SW[256][4][4];
	static DATA_T O47_SW[256][4][4];
	static DATA_T O48_SW[256][4][4];
	static DATA_T O49_SW[256][4][4];
	static DATA_T O50_SW[256][4][4];
	static DATA_T O51_SW[256][4][4];
	static DATA_T O52_SW[256][4][4];
	static DATA_T O53_SW[256][4][4];
	static DATA_T O54_SW[256][4][4];
	static DATA_T O55_SW[256][4][4];
	static DATA_T O56_SW[256][4][4];
	static DATA_T O57_SW[256][4][4];
	static DATA_T O58_SW[256][4][4];
	static DATA_T O59_SW[256][4][4];
	static DATA_T O60_SW[256][4][4];
	static DATA_T O61_SW[256][4][4];
	static DATA_T O62_SW[256][4][4];
	static DATA_T O63_SW[256][4][4];
	static DATA_T O64_SW[256][4][4];
	static DATA_T O65_SW[256][4][4];
	static DATA_T O66_SW[256][4][4];
	static DATA_T O67_SW[256][4][4];
	static DATA_T O68_SW[256][4][4];
	static DATA_T O69_SW[256][4][4];
	static DATA_T O70_SW[512][2][2];
	static DATA_T O71_SW[512][2][2];
	static DATA_T O72_SW[512][2][2];
	static DATA_T O73_SW[512][2][2];
	static DATA_T O74_SW[512][2][2];
	static DATA_T O75_SW[512][2][2];
	static DATA_T O76_SW[512][2][2];
	static DATA_T O77_SW[512][2][2];
	static DATA_T O78_SW[512][2][2];
	static DATA_T O79_SW[512][2][2];
	static DATA_T O80_SW[512][2][2];
	static DATA_T O81_SW[512][2][2];
	static DATA_T O82_SW[512][2][2];
	static DATA_T O83_SW[512][2][2];
	static DATA_T O84_SW[512][2][2];
	static DATA_T O85_SW[512][2][2];
	static DATA_T O86_SW[512][2][2];
	static DATA_T O87_SW[512];
	static DATA_T O88_SW[10];
	

    FILE *w_stream = fopen(argv[1], "rb");
    if (w_stream == NULL) printf("weight file was not opened");
    FILE *i_stream = fopen(argv[2], "rb");
    if (i_stream == NULL) printf("input file was not opened");
    FILE *o_stream = fopen("./output/resnet34/output_value/c_output.txt", "w");
    if (o_stream == NULL) printf("Output file was not opened");
    FILE *c_num = fopen("./output/resnet34/output_value/c_output_num.txt", "w");
    if (c_num == NULL) printf("Output file was not opened");

    // initialize input, weight, bias variables using fread
    printf("[c_verifier.cpp]Start Initialzation\n");
    for (k = 0; k <  32 ; k++) {
	for (x = 0; x < 32 ; x++) {
		for(y = 0; y < 3 ; y++) {
			fread(&trash, sizeof(DATA_IN), 1, i_stream);
            I[y][k][x] = (DATA_T) trash;
		}
	}
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 3 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W1[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B1[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W3[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B3[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W4[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B4[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W5[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B5[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W8[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B8[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W10[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B10[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W13[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B13[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W15[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B15[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W18[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B18[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W20[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B20[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W21[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B21[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W24[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B24[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W26[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B26[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W29[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B29[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W31[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B31[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W34[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B34[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W36[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B36[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W39[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B39[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W41[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B41[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W42[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B42[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W45[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B45[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W47[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B47[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W50[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B50[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W52[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B52[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W55[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B55[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W57[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B57[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W60[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B60[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W62[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B62[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W65[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B65[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W67[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B67[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W70[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B70[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W72[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B72[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W73[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B73[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W76[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B76[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W78[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B78[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W81[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B81[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(DATA_IN), 1, w_stream);
                W83[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B83[m] = (DATA_T) trash;
}

	for (m = 0; m <  512 ; m++) {
	for (k = 0; k < 10 ; k++) {
		fread(&trash, sizeof(DATA_IN), 1, w_stream);
        W88[k][m] = (DATA_T) trash;
	}
}

for (m = 0; m < 10 ; m++) {
    fread(&trash, sizeof(DATA_IN), 1, w_stream);
    B88[m] = (DATA_T) trash;
}

	
    printf("[c_verifier.cpp]Finish Initialization\n");

    // call function of each layer based on csv file which containts layer information
    printf("[c_verifier.cpp]Calculate Conv2D1\n");
	SW_conv2d_1(I,O1_SW,W1,B1);
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
	printf("[c_verifier.cpp]Calculate Activation16\n");
	SW_activation_6(O15_SW,O16_SW);
	printf("[c_verifier.cpp]Calculate Add17\n");
	SW_add_3(O12_SW,O16_SW,O17_SW);
	printf("[c_verifier.cpp]Calculate Conv2D18\n");
	SW_conv2d_8(O17_SW,O18_SW,W18,B18);
	printf("[c_verifier.cpp]Calculate Activation19\n");
	SW_activation_7(O18_SW,O19_SW);
	printf("[c_verifier.cpp]Calculate Conv2D20\n");
	SW_conv2d_9(O19_SW,O20_SW,W20,B20);
	printf("[c_verifier.cpp]Calculate Conv2D21\n");
	SW_conv2d_10(O17_SW,O21_SW,W21,B21);
	printf("[c_verifier.cpp]Calculate Activation22\n");
	SW_activation_8(O20_SW,O22_SW);
	printf("[c_verifier.cpp]Calculate Add23\n");
	SW_add_4(O21_SW,O22_SW,O23_SW);
	printf("[c_verifier.cpp]Calculate Conv2D24\n");
	SW_conv2d_11(O23_SW,O24_SW,W24,B24);
	printf("[c_verifier.cpp]Calculate Activation25\n");
	SW_activation_9(O24_SW,O25_SW);
	printf("[c_verifier.cpp]Calculate Conv2D26\n");
	SW_conv2d_12(O25_SW,O26_SW,W26,B26);
	printf("[c_verifier.cpp]Calculate Activation27\n");
	SW_activation_10(O26_SW,O27_SW);
	printf("[c_verifier.cpp]Calculate Add28\n");
	SW_add_5(O23_SW,O27_SW,O28_SW);
	printf("[c_verifier.cpp]Calculate Conv2D29\n");
	SW_conv2d_13(O28_SW,O29_SW,W29,B29);
	printf("[c_verifier.cpp]Calculate Activation30\n");
	SW_activation_11(O29_SW,O30_SW);
	printf("[c_verifier.cpp]Calculate Conv2D31\n");
	SW_conv2d_14(O30_SW,O31_SW,W31,B31);
	printf("[c_verifier.cpp]Calculate Activation32\n");
	SW_activation_12(O31_SW,O32_SW);
	printf("[c_verifier.cpp]Calculate Add33\n");
	SW_add_6(O28_SW,O32_SW,O33_SW);
	printf("[c_verifier.cpp]Calculate Conv2D34\n");
	SW_conv2d_15(O33_SW,O34_SW,W34,B34);
	printf("[c_verifier.cpp]Calculate Activation35\n");
	SW_activation_13(O34_SW,O35_SW);
	printf("[c_verifier.cpp]Calculate Conv2D36\n");
	SW_conv2d_16(O35_SW,O36_SW,W36,B36);
	printf("[c_verifier.cpp]Calculate Activation37\n");
	SW_activation_14(O36_SW,O37_SW);
	printf("[c_verifier.cpp]Calculate Add38\n");
	SW_add_7(O33_SW,O37_SW,O38_SW);
	printf("[c_verifier.cpp]Calculate Conv2D39\n");
	SW_conv2d_17(O38_SW,O39_SW,W39,B39);
	printf("[c_verifier.cpp]Calculate Activation40\n");
	SW_activation_15(O39_SW,O40_SW);
	printf("[c_verifier.cpp]Calculate Conv2D41\n");
	SW_conv2d_18(O40_SW,O41_SW,W41,B41);
	printf("[c_verifier.cpp]Calculate Conv2D42\n");
	SW_conv2d_19(O38_SW,O42_SW,W42,B42);
	printf("[c_verifier.cpp]Calculate Activation43\n");
	SW_activation_16(O41_SW,O43_SW);
	printf("[c_verifier.cpp]Calculate Add44\n");
	SW_add_8(O42_SW,O43_SW,O44_SW);
	printf("[c_verifier.cpp]Calculate Conv2D45\n");
	SW_conv2d_20(O44_SW,O45_SW,W45,B45);
	printf("[c_verifier.cpp]Calculate Activation46\n");
	SW_activation_17(O45_SW,O46_SW);
	printf("[c_verifier.cpp]Calculate Conv2D47\n");
	SW_conv2d_21(O46_SW,O47_SW,W47,B47);
	printf("[c_verifier.cpp]Calculate Activation48\n");
	SW_activation_18(O47_SW,O48_SW);
	printf("[c_verifier.cpp]Calculate Add49\n");
	SW_add_9(O44_SW,O48_SW,O49_SW);
	printf("[c_verifier.cpp]Calculate Conv2D50\n");
	SW_conv2d_22(O49_SW,O50_SW,W50,B50);
	printf("[c_verifier.cpp]Calculate Activation51\n");
	SW_activation_19(O50_SW,O51_SW);
	printf("[c_verifier.cpp]Calculate Conv2D52\n");
	SW_conv2d_23(O51_SW,O52_SW,W52,B52);
	printf("[c_verifier.cpp]Calculate Activation53\n");
	SW_activation_20(O52_SW,O53_SW);
	printf("[c_verifier.cpp]Calculate Add54\n");
	SW_add_10(O49_SW,O53_SW,O54_SW);
	printf("[c_verifier.cpp]Calculate Conv2D55\n");
	SW_conv2d_24(O54_SW,O55_SW,W55,B55);
	printf("[c_verifier.cpp]Calculate Activation56\n");
	SW_activation_21(O55_SW,O56_SW);
	printf("[c_verifier.cpp]Calculate Conv2D57\n");
	SW_conv2d_25(O56_SW,O57_SW,W57,B57);
	printf("[c_verifier.cpp]Calculate Activation58\n");
	SW_activation_22(O57_SW,O58_SW);
	printf("[c_verifier.cpp]Calculate Add59\n");
	SW_add_11(O54_SW,O58_SW,O59_SW);
	printf("[c_verifier.cpp]Calculate Conv2D60\n");
	SW_conv2d_26(O59_SW,O60_SW,W60,B60);
	printf("[c_verifier.cpp]Calculate Activation61\n");
	SW_activation_23(O60_SW,O61_SW);
	printf("[c_verifier.cpp]Calculate Conv2D62\n");
	SW_conv2d_27(O61_SW,O62_SW,W62,B62);
	printf("[c_verifier.cpp]Calculate Activation63\n");
	SW_activation_24(O62_SW,O63_SW);
	printf("[c_verifier.cpp]Calculate Add64\n");
	SW_add_12(O59_SW,O63_SW,O64_SW);
	printf("[c_verifier.cpp]Calculate Conv2D65\n");
	SW_conv2d_28(O64_SW,O65_SW,W65,B65);
	printf("[c_verifier.cpp]Calculate Activation66\n");
	SW_activation_25(O65_SW,O66_SW);
	printf("[c_verifier.cpp]Calculate Conv2D67\n");
	SW_conv2d_29(O66_SW,O67_SW,W67,B67);
	printf("[c_verifier.cpp]Calculate Activation68\n");
	SW_activation_26(O67_SW,O68_SW);
	printf("[c_verifier.cpp]Calculate Add69\n");
	SW_add_13(O64_SW,O68_SW,O69_SW);
	printf("[c_verifier.cpp]Calculate Conv2D70\n");
	SW_conv2d_30(O69_SW,O70_SW,W70,B70);
	printf("[c_verifier.cpp]Calculate Activation71\n");
	SW_activation_27(O70_SW,O71_SW);
	printf("[c_verifier.cpp]Calculate Conv2D72\n");
	SW_conv2d_31(O71_SW,O72_SW,W72,B72);
	printf("[c_verifier.cpp]Calculate Conv2D73\n");
	SW_conv2d_32(O69_SW,O73_SW,W73,B73);
	printf("[c_verifier.cpp]Calculate Activation74\n");
	SW_activation_28(O72_SW,O74_SW);
	printf("[c_verifier.cpp]Calculate Add75\n");
	SW_add_14(O73_SW,O74_SW,O75_SW);
	printf("[c_verifier.cpp]Calculate Conv2D76\n");
	SW_conv2d_33(O75_SW,O76_SW,W76,B76);
	printf("[c_verifier.cpp]Calculate Activation77\n");
	SW_activation_29(O76_SW,O77_SW);
	printf("[c_verifier.cpp]Calculate Conv2D78\n");
	SW_conv2d_34(O77_SW,O78_SW,W78,B78);
	printf("[c_verifier.cpp]Calculate Activation79\n");
	SW_activation_30(O78_SW,O79_SW);
	printf("[c_verifier.cpp]Calculate Add80\n");
	SW_add_15(O75_SW,O79_SW,O80_SW);
	printf("[c_verifier.cpp]Calculate Conv2D81\n");
	SW_conv2d_35(O80_SW,O81_SW,W81,B81);
	printf("[c_verifier.cpp]Calculate Activation82\n");
	SW_activation_31(O81_SW,O82_SW);
	printf("[c_verifier.cpp]Calculate Conv2D83\n");
	SW_conv2d_36(O82_SW,O83_SW,W83,B83);
	printf("[c_verifier.cpp]Calculate Activation84\n");
	SW_activation_32(O83_SW,O84_SW);
	printf("[c_verifier.cpp]Calculate Add85\n");
	SW_add_16(O80_SW,O84_SW,O85_SW);
	printf("[c_verifier.cpp]Calculate Activation86\n");
	SW_activation_33(O85_SW,O86_SW);
	printf("[c_verifier.cpp]Calculate GlobalAveragePooling2D87\n");
	SW_global_average_pooling2d_1(O86_SW,O87_SW);
	printf("[c_verifier.cpp]Calculate Dense88\n");
	SW_dense_1(O87_SW,O88_SW,W88,B88);
	

    // print each element of output variables
    printf("[c_verifier.cpp]Print Result\n");
    fprintf(o_stream,"%s","InputLayerinput_1 : [[");
for (k = 0; k < 32 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 32 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 3 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)I[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)I[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_1 : [[");
for (k = 0; k < 32 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 32 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O1_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O1_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Activationactivation_1 : [[");
for (k = 0; k < 32 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 32 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O2_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O2_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dres0a_branch2a : [[");
for (k = 0; k < 16 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 16 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O3_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O3_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_2 : [[");
for (k = 0; k < 16 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 16 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O4_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O4_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_3 : [[");
for (k = 0; k < 16 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 16 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O5_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O5_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Activationactivation_2 : [[");
for (k = 0; k < 16 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 16 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O6_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O6_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Addadd_1 : [[");
for (k = 0; k < 16 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 16 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O7_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O7_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_4 : [[");
for (k = 0; k < 16 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 16 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O8_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O8_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Activationactivation_3 : [[");
for (k = 0; k < 16 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 16 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O9_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O9_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_5 : [[");
for (k = 0; k < 16 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 16 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O10_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O10_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Activationactivation_4 : [[");
for (k = 0; k < 16 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 16 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O11_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O11_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Addadd_2 : [[");
for (k = 0; k < 16 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 16 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O12_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O12_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_6 : [[");
for (k = 0; k < 16 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 16 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O13_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O13_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Activationactivation_5 : [[");
for (k = 0; k < 16 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 16 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O14_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O14_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_7 : [[");
for (k = 0; k < 16 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 16 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O15_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O15_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Activationactivation_6 : [[");
for (k = 0; k < 16 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 16 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O16_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O16_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Addadd_3 : [[");
for (k = 0; k < 16 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 16 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 64 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O17_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O17_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_8 : [[");
for (k = 0; k < 8 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 8 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O18_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O18_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Activationactivation_7 : [[");
for (k = 0; k < 8 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 8 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O19_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O19_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_9 : [[");
for (k = 0; k < 8 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 8 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O20_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O20_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_10 : [[");
for (k = 0; k < 8 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 8 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O21_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O21_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Activationactivation_8 : [[");
for (k = 0; k < 8 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 8 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O22_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O22_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Addadd_4 : [[");
for (k = 0; k < 8 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 8 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O23_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O23_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_11 : [[");
for (k = 0; k < 8 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 8 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O24_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O24_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Activationactivation_9 : [[");
for (k = 0; k < 8 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 8 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O25_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O25_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_12 : [[");
for (k = 0; k < 8 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 8 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O26_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O26_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Activationactivation_10 : [[");
for (k = 0; k < 8 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 8 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O27_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O27_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Addadd_5 : [[");
for (k = 0; k < 8 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 8 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O28_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O28_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_13 : [[");
for (k = 0; k < 8 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 8 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O29_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O29_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Activationactivation_11 : [[");
for (k = 0; k < 8 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 8 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O30_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O30_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_14 : [[");
for (k = 0; k < 8 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 8 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O31_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O31_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Activationactivation_12 : [[");
for (k = 0; k < 8 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 8 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O32_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O32_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Addadd_6 : [[");
for (k = 0; k < 8 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 8 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O33_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O33_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_15 : [[");
for (k = 0; k < 8 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 8 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O34_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O34_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Activationactivation_13 : [[");
for (k = 0; k < 8 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 8 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O35_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O35_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_16 : [[");
for (k = 0; k < 8 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 8 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O36_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O36_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Activationactivation_14 : [[");
for (k = 0; k < 8 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 8 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O37_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O37_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Addadd_7 : [[");
for (k = 0; k < 8 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 8 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O38_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O38_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_17 : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O39_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O39_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Activationactivation_15 : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O40_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O40_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_18 : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O41_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O41_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_19 : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O42_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O42_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Activationactivation_16 : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O43_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O43_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Addadd_8 : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O44_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O44_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_20 : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O45_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O45_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Activationactivation_17 : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O46_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O46_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_21 : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O47_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O47_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Activationactivation_18 : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O48_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O48_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Addadd_9 : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O49_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O49_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_22 : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O50_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O50_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Activationactivation_19 : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O51_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O51_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_23 : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O52_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O52_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Activationactivation_20 : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O53_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O53_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Addadd_10 : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O54_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O54_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_24 : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O55_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O55_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Activationactivation_21 : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O56_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O56_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_25 : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O57_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O57_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Activationactivation_22 : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O58_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O58_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Addadd_11 : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O59_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O59_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_26 : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O60_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O60_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Activationactivation_23 : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O61_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O61_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_27 : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O62_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O62_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Activationactivation_24 : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O63_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O63_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Addadd_12 : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O64_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O64_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_28 : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O65_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O65_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Activationactivation_25 : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O66_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O66_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_29 : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O67_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O67_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Activationactivation_26 : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O68_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O68_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Addadd_13 : [[");
for (k = 0; k < 4 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 4 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O69_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O69_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_30 : [[");
for (k = 0; k < 2 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 2 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O70_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O70_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Activationactivation_27 : [[");
for (k = 0; k < 2 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 2 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O71_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O71_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_31 : [[");
for (k = 0; k < 2 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 2 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O72_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O72_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_32 : [[");
for (k = 0; k < 2 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 2 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O73_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O73_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Activationactivation_28 : [[");
for (k = 0; k < 2 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 2 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O74_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O74_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Addadd_14 : [[");
for (k = 0; k < 2 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 2 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O75_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O75_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_33 : [[");
for (k = 0; k < 2 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 2 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O76_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O76_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Activationactivation_29 : [[");
for (k = 0; k < 2 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 2 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O77_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O77_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_34 : [[");
for (k = 0; k < 2 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 2 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O78_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O78_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Activationactivation_30 : [[");
for (k = 0; k < 2 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 2 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O79_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O79_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Addadd_15 : [[");
for (k = 0; k < 2 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 2 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O80_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O80_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_35 : [[");
for (k = 0; k < 2 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 2 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O81_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O81_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Activationactivation_31 : [[");
for (k = 0; k < 2 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 2 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O82_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O82_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Conv2Dconv2d_36 : [[");
for (k = 0; k < 2 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 2 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O83_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O83_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Activationactivation_32 : [[");
for (k = 0; k < 2 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 2 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O84_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O84_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Addadd_16 : [[");
for (k = 0; k < 2 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 2 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O85_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O85_SW[y][k][x]);
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


	fprintf(o_stream,"%s","Activationactivation_33 : [[");
for (k = 0; k < 2 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 2 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 512 ; y++) {
			fprintf(o_stream,"%.6f ",(DATA_T)O86_SW[y][k][x]);
			fprintf(c_num,"%.6f ",(DATA_T)O86_SW[y][k][x]);
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


	fprintf(o_stream,"%s","GlobalAveragePooling2Dglobal_average_pooling2d_1 : [[");
for (k = 0; k <  512 ; k++) {
	fprintf(o_stream,"%.6f ",(DATA_T)O87_SW[k]);
	fprintf(c_num,"%.6f ",(DATA_T)O87_SW[k]);
}
fprintf(o_stream,"%s","]]\n\n");


	fprintf(o_stream,"%s","Densedense_1 : [[");
for (k = 0; k <  10 ; k++) {
	fprintf(o_stream,"%.6f ",(DATA_T)O88_SW[k]);
	fprintf(c_num,"%.6f ",(DATA_T)O88_SW[k]);
}
fprintf(o_stream,"%s","]]\n\n");


	

    fclose(w_stream);
    fclose(i_stream);
    fclose(o_stream);
    fclose(c_num);

    return 0;
}
