#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <math.h>
    
using namespace std;

typedef float DATA_T;

void SW_block1_conv1(DATA_T I[3][224][224], DATA_T O[32][111][111], DATA_T W[32][3][3][3]) {
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

void SW_block1_conv1_bn(DATA_T I[32][111][111], DATA_T O[32][111][111], DATA_T W[4][32]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 32; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 111; x++){
			for (y = 0; y < 111; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_block1_conv1_act(DATA_T I[32][111][111], DATA_T O[32][111][111]) {
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

void SW_block1_conv2(DATA_T I[32][111][111], DATA_T O[64][109][109], DATA_T W[64][32][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<64; m++) {
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

void SW_block1_conv2_bn(DATA_T I[64][109][109], DATA_T O[64][109][109], DATA_T W[4][64]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 64; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 109; x++){
			for (y = 0; y < 109; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_block1_conv2_act(DATA_T I[64][109][109], DATA_T O[64][109][109]) {
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

void SW_block2_sepconv1(DATA_T I[64][109][109], DATA_T O[128][109][109], DATA_T W1[64][3][3], DATA_T W2[128][64]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
    int p = (1 *(109 - 1) - 109 + 3)/2;
	for (m = 0; m<64; m++) {
		for (x = 0; x<109; x++) {
			for (y = 0; y<109; y++) {
				ofm = 0;

				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {
						if (x*1 + i < 109 + p && y*1 + j < 109 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                            ifm = I[m][x*1 + i - p][y*1 + j -p];
						}
						else {
							ifm = 0; // zero padding
						}
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<128;m++){
        for(i=0;i<109;i++){
            for(j=0;j<109;j++){
                ofm1=0;
                for(k=0;k<64;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block2_sepconv1_bn(DATA_T I[128][109][109], DATA_T O[128][109][109], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 128; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 109; x++){
			for (y = 0; y < 109; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_block2_sepconv2_act(DATA_T I[128][109][109], DATA_T O[128][109][109]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
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

void SW_block2_sepconv2(DATA_T I[128][109][109], DATA_T O[128][109][109], DATA_T W1[128][3][3], DATA_T W2[128][128]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
    int p = (1 *(109 - 1) - 109 + 3)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<109; x++) {
			for (y = 0; y<109; y++) {
				ofm = 0;

				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {
						if (x*1 + i < 109 + p && y*1 + j < 109 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                            ifm = I[m][x*1 + i - p][y*1 + j -p];
						}
						else {
							ifm = 0; // zero padding
						}
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<128;m++){
        for(i=0;i<109;i++){
            for(j=0;j<109;j++){
                ofm1=0;
                for(k=0;k<128;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block2_sepconv2_bn(DATA_T I[128][109][109], DATA_T O[128][109][109], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 128; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 109; x++){
			for (y = 0; y < 109; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_conv2d_204(DATA_T I[64][109][109], DATA_T O[128][55][55], DATA_T W[128][64][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (2 *(55 - 1) - 109 + 1)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<55; x++) {
			for (y = 0; y<55; y++) {
				ofm = 0;
				for (k = 0; k<64; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*2 + i < 109 + p && y*2 + j < 109 + p && x*2 + i -p >= 0 && y*2 + j -p >= 0) {
                                    ifm = I[k][x*2 + i - p][y*2 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}

				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block2_pool(DATA_T I[128][109][109], DATA_T O[128][55][55])
{
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<128; m++) {
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
void SW_batch_normalization_204(DATA_T I[128][55][55], DATA_T O[128][55][55], DATA_T W[4][128]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 128; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 55; x++){
			for (y = 0; y < 55; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_add_1(DATA_T I1[128][55][55], DATA_T I2[128][55][55], DATA_T O[128][55][55]) {
	int m, x, y;
	for (m = 0; m< 128; m++) {
		for (x = 0; x < 55; x++) {
			for(y = 0; y < 55; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_block3_sepconv1_act(DATA_T I[128][55][55], DATA_T O[128][55][55]) {
	int m, x, y, i, j, k;
	for (m = 0; m<128; m++) {
		for (x = 0; x<55; x++) {
			for (y = 0; y<55; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block3_sepconv1(DATA_T I[128][55][55], DATA_T O[256][55][55], DATA_T W1[128][3][3], DATA_T W2[256][128]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
    int p = (1 *(55 - 1) - 55 + 3)/2;
	for (m = 0; m<128; m++) {
		for (x = 0; x<55; x++) {
			for (y = 0; y<55; y++) {
				ofm = 0;

				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {
						if (x*1 + i < 55 + p && y*1 + j < 55 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                            ifm = I[m][x*1 + i - p][y*1 + j -p];
						}
						else {
							ifm = 0; // zero padding
						}
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<256;m++){
        for(i=0;i<55;i++){
            for(j=0;j<55;j++){
                ofm1=0;
                for(k=0;k<128;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block3_sepconv1_bn(DATA_T I[256][55][55], DATA_T O[256][55][55], DATA_T W[4][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 256; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 55; x++){
			for (y = 0; y < 55; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_block3_sepconv2_act(DATA_T I[256][55][55], DATA_T O[256][55][55]) {
	int m, x, y, i, j, k;
	for (m = 0; m<256; m++) {
		for (x = 0; x<55; x++) {
			for (y = 0; y<55; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_block3_sepconv2(DATA_T I[256][55][55], DATA_T O[256][55][55], DATA_T W1[256][3][3], DATA_T W2[256][256]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
    int p = (1 *(55 - 1) - 55 + 3)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<55; x++) {
			for (y = 0; y<55; y++) {
				ofm = 0;

				for (i = 0; i<3; i++) {
					for (j = 0; j<3; j++) {
						if (x*1 + i < 55 + p && y*1 + j < 55 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
                            ifm = I[m][x*1 + i - p][y*1 + j -p];
						}
						else {
							ifm = 0; // zero padding
						}
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<256;m++){
        for(i=0;i<55;i++){
            for(j=0;j<55;j++){
                ofm1=0;
                for(k=0;k<256;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block3_sepconv2_bn(DATA_T I[256][55][55], DATA_T O[256][55][55], DATA_T W[4][256]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 256; m++){
        DATA_T gamma = W[0][m];
        DATA_T beta = W[1][m];
        DATA_T mean = W[2][m];
        DATA_T var = W[3][m];

		for (x = 0; x < 55; x++){
			for (y = 0; y < 55; y++){
				O[m][x][y] = ((gamma * (I[m][x][y]-mean))/sqrt(var+epsilon)) + beta;
			}
		}
	}
}
void SW_conv2d_205(DATA_T I[128][55][55], DATA_T O[256][28][28], DATA_T W[256][128][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (2 *(28 - 1) - 55 + 1)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<28; x++) {
			for (y = 0; y<28; y++) {
				ofm = 0;
				for (k = 0; k<128; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*2 + i < 55 + p && y*2 + j < 55 + p && x*2 + i -p >= 0 && y*2 + j -p >= 0) {
                                    ifm = I[k][x*2 + i - p][y*2 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}

				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block3_pool(DATA_T I[256][55][55], DATA_T O[256][28][28])
{
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<256; m++) {
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
void SW_batch_normalization_205(DATA_T I[256][28][28], DATA_T O[256][28][28], DATA_T W[4][256]) {
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
void SW_add_2(DATA_T I1[256][28][28], DATA_T I2[256][28][28], DATA_T O[256][28][28]) {
	int m, x, y;
	for (m = 0; m< 256; m++) {
		for (x = 0; x < 28; x++) {
			for(y = 0; y < 28; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_block4_sepconv1_act(DATA_T I[256][28][28], DATA_T O[256][28][28]) {
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

void SW_block4_sepconv1(DATA_T I[256][28][28], DATA_T O[728][28][28], DATA_T W1[256][3][3], DATA_T W2[728][256]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
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
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<728;m++){
        for(i=0;i<28;i++){
            for(j=0;j<28;j++){
                ofm1=0;
                for(k=0;k<256;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block4_sepconv1_bn(DATA_T I[728][28][28], DATA_T O[728][28][28], DATA_T W[4][728]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 728; m++){
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
void SW_block4_sepconv2_act(DATA_T I[728][28][28], DATA_T O[728][28][28]) {
	int m, x, y, i, j, k;
	for (m = 0; m<728; m++) {
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

void SW_block4_sepconv2(DATA_T I[728][28][28], DATA_T O[728][28][28], DATA_T W1[728][3][3], DATA_T W2[728][728]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
    int p = (1 *(28 - 1) - 28 + 3)/2;
	for (m = 0; m<728; m++) {
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
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<728;m++){
        for(i=0;i<28;i++){
            for(j=0;j<28;j++){
                ofm1=0;
                for(k=0;k<728;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block4_sepconv2_bn(DATA_T I[728][28][28], DATA_T O[728][28][28], DATA_T W[4][728]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 728; m++){
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
void SW_conv2d_206(DATA_T I[256][28][28], DATA_T O[728][14][14], DATA_T W[728][256][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (2 *(14 - 1) - 28 + 1)/2;
	for (m = 0; m<728; m++) {
		for (x = 0; x<14; x++) {
			for (y = 0; y<14; y++) {
				ofm = 0;
				for (k = 0; k<256; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*2 + i < 28 + p && y*2 + j < 28 + p && x*2 + i -p >= 0 && y*2 + j -p >= 0) {
                                    ifm = I[k][x*2 + i - p][y*2 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}

				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block4_pool(DATA_T I[728][28][28], DATA_T O[728][14][14])
{
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<728; m++) {
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
void SW_batch_normalization_206(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W[4][728]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 728; m++){
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
void SW_add_3(DATA_T I1[728][14][14], DATA_T I2[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y;
	for (m = 0; m< 728; m++) {
		for (x = 0; x < 14; x++) {
			for(y = 0; y < 14; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_block5_sepconv1_act(DATA_T I[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<728; m++) {
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

void SW_block5_sepconv1(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W1[728][3][3], DATA_T W2[728][728]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<728; m++) {
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
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<728;m++){
        for(i=0;i<14;i++){
            for(j=0;j<14;j++){
                ofm1=0;
                for(k=0;k<728;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block5_sepconv1_bn(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W[4][728]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 728; m++){
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
void SW_block5_sepconv2_act(DATA_T I[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<728; m++) {
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

void SW_block5_sepconv2(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W1[728][3][3], DATA_T W2[728][728]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<728; m++) {
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
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<728;m++){
        for(i=0;i<14;i++){
            for(j=0;j<14;j++){
                ofm1=0;
                for(k=0;k<728;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block5_sepconv2_bn(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W[4][728]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 728; m++){
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
void SW_block5_sepconv3_act(DATA_T I[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<728; m++) {
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

void SW_block5_sepconv3(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W1[728][3][3], DATA_T W2[728][728]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<728; m++) {
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
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<728;m++){
        for(i=0;i<14;i++){
            for(j=0;j<14;j++){
                ofm1=0;
                for(k=0;k<728;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block5_sepconv3_bn(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W[4][728]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 728; m++){
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
void SW_add_4(DATA_T I1[728][14][14], DATA_T I2[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y;
	for (m = 0; m< 728; m++) {
		for (x = 0; x < 14; x++) {
			for(y = 0; y < 14; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_block6_sepconv1_act(DATA_T I[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<728; m++) {
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

void SW_block6_sepconv1(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W1[728][3][3], DATA_T W2[728][728]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<728; m++) {
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
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<728;m++){
        for(i=0;i<14;i++){
            for(j=0;j<14;j++){
                ofm1=0;
                for(k=0;k<728;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block6_sepconv1_bn(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W[4][728]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 728; m++){
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
void SW_block6_sepconv2_act(DATA_T I[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<728; m++) {
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

void SW_block6_sepconv2(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W1[728][3][3], DATA_T W2[728][728]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<728; m++) {
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
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<728;m++){
        for(i=0;i<14;i++){
            for(j=0;j<14;j++){
                ofm1=0;
                for(k=0;k<728;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block6_sepconv2_bn(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W[4][728]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 728; m++){
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
void SW_block6_sepconv3_act(DATA_T I[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<728; m++) {
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

void SW_block6_sepconv3(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W1[728][3][3], DATA_T W2[728][728]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<728; m++) {
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
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<728;m++){
        for(i=0;i<14;i++){
            for(j=0;j<14;j++){
                ofm1=0;
                for(k=0;k<728;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block6_sepconv3_bn(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W[4][728]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 728; m++){
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
void SW_add_5(DATA_T I1[728][14][14], DATA_T I2[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y;
	for (m = 0; m< 728; m++) {
		for (x = 0; x < 14; x++) {
			for(y = 0; y < 14; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_block7_sepconv1_act(DATA_T I[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<728; m++) {
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

void SW_block7_sepconv1(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W1[728][3][3], DATA_T W2[728][728]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<728; m++) {
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
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<728;m++){
        for(i=0;i<14;i++){
            for(j=0;j<14;j++){
                ofm1=0;
                for(k=0;k<728;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block7_sepconv1_bn(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W[4][728]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 728; m++){
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
void SW_block7_sepconv2_act(DATA_T I[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<728; m++) {
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

void SW_block7_sepconv2(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W1[728][3][3], DATA_T W2[728][728]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<728; m++) {
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
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<728;m++){
        for(i=0;i<14;i++){
            for(j=0;j<14;j++){
                ofm1=0;
                for(k=0;k<728;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block7_sepconv2_bn(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W[4][728]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 728; m++){
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
void SW_block7_sepconv3_act(DATA_T I[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<728; m++) {
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

void SW_block7_sepconv3(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W1[728][3][3], DATA_T W2[728][728]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<728; m++) {
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
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<728;m++){
        for(i=0;i<14;i++){
            for(j=0;j<14;j++){
                ofm1=0;
                for(k=0;k<728;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block7_sepconv3_bn(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W[4][728]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 728; m++){
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
void SW_add_6(DATA_T I1[728][14][14], DATA_T I2[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y;
	for (m = 0; m< 728; m++) {
		for (x = 0; x < 14; x++) {
			for(y = 0; y < 14; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_block8_sepconv1_act(DATA_T I[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<728; m++) {
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

void SW_block8_sepconv1(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W1[728][3][3], DATA_T W2[728][728]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<728; m++) {
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
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<728;m++){
        for(i=0;i<14;i++){
            for(j=0;j<14;j++){
                ofm1=0;
                for(k=0;k<728;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block8_sepconv1_bn(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W[4][728]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 728; m++){
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
void SW_block8_sepconv2_act(DATA_T I[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<728; m++) {
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

void SW_block8_sepconv2(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W1[728][3][3], DATA_T W2[728][728]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<728; m++) {
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
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<728;m++){
        for(i=0;i<14;i++){
            for(j=0;j<14;j++){
                ofm1=0;
                for(k=0;k<728;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block8_sepconv2_bn(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W[4][728]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 728; m++){
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
void SW_block8_sepconv3_act(DATA_T I[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<728; m++) {
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

void SW_block8_sepconv3(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W1[728][3][3], DATA_T W2[728][728]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<728; m++) {
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
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<728;m++){
        for(i=0;i<14;i++){
            for(j=0;j<14;j++){
                ofm1=0;
                for(k=0;k<728;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block8_sepconv3_bn(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W[4][728]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 728; m++){
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
void SW_add_7(DATA_T I1[728][14][14], DATA_T I2[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y;
	for (m = 0; m< 728; m++) {
		for (x = 0; x < 14; x++) {
			for(y = 0; y < 14; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_block9_sepconv1_act(DATA_T I[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<728; m++) {
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

void SW_block9_sepconv1(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W1[728][3][3], DATA_T W2[728][728]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<728; m++) {
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
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<728;m++){
        for(i=0;i<14;i++){
            for(j=0;j<14;j++){
                ofm1=0;
                for(k=0;k<728;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block9_sepconv1_bn(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W[4][728]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 728; m++){
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
void SW_block9_sepconv2_act(DATA_T I[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<728; m++) {
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

void SW_block9_sepconv2(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W1[728][3][3], DATA_T W2[728][728]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<728; m++) {
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
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<728;m++){
        for(i=0;i<14;i++){
            for(j=0;j<14;j++){
                ofm1=0;
                for(k=0;k<728;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block9_sepconv2_bn(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W[4][728]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 728; m++){
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
void SW_block9_sepconv3_act(DATA_T I[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<728; m++) {
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

void SW_block9_sepconv3(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W1[728][3][3], DATA_T W2[728][728]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<728; m++) {
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
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<728;m++){
        for(i=0;i<14;i++){
            for(j=0;j<14;j++){
                ofm1=0;
                for(k=0;k<728;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block9_sepconv3_bn(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W[4][728]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 728; m++){
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
void SW_add_8(DATA_T I1[728][14][14], DATA_T I2[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y;
	for (m = 0; m< 728; m++) {
		for (x = 0; x < 14; x++) {
			for(y = 0; y < 14; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_block10_sepconv1_act(DATA_T I[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<728; m++) {
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

void SW_block10_sepconv1(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W1[728][3][3], DATA_T W2[728][728]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<728; m++) {
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
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<728;m++){
        for(i=0;i<14;i++){
            for(j=0;j<14;j++){
                ofm1=0;
                for(k=0;k<728;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block10_sepconv1_bn(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W[4][728]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 728; m++){
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
void SW_block10_sepconv2_act(DATA_T I[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<728; m++) {
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

void SW_block10_sepconv2(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W1[728][3][3], DATA_T W2[728][728]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<728; m++) {
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
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<728;m++){
        for(i=0;i<14;i++){
            for(j=0;j<14;j++){
                ofm1=0;
                for(k=0;k<728;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block10_sepconv2_bn(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W[4][728]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 728; m++){
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
void SW_block10_sepconv3_act(DATA_T I[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<728; m++) {
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

void SW_block10_sepconv3(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W1[728][3][3], DATA_T W2[728][728]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<728; m++) {
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
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<728;m++){
        for(i=0;i<14;i++){
            for(j=0;j<14;j++){
                ofm1=0;
                for(k=0;k<728;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block10_sepconv3_bn(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W[4][728]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 728; m++){
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
void SW_add_9(DATA_T I1[728][14][14], DATA_T I2[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y;
	for (m = 0; m< 728; m++) {
		for (x = 0; x < 14; x++) {
			for(y = 0; y < 14; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_block11_sepconv1_act(DATA_T I[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<728; m++) {
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

void SW_block11_sepconv1(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W1[728][3][3], DATA_T W2[728][728]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<728; m++) {
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
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<728;m++){
        for(i=0;i<14;i++){
            for(j=0;j<14;j++){
                ofm1=0;
                for(k=0;k<728;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block11_sepconv1_bn(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W[4][728]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 728; m++){
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
void SW_block11_sepconv2_act(DATA_T I[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<728; m++) {
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

void SW_block11_sepconv2(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W1[728][3][3], DATA_T W2[728][728]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<728; m++) {
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
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<728;m++){
        for(i=0;i<14;i++){
            for(j=0;j<14;j++){
                ofm1=0;
                for(k=0;k<728;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block11_sepconv2_bn(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W[4][728]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 728; m++){
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
void SW_block11_sepconv3_act(DATA_T I[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<728; m++) {
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

void SW_block11_sepconv3(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W1[728][3][3], DATA_T W2[728][728]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<728; m++) {
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
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<728;m++){
        for(i=0;i<14;i++){
            for(j=0;j<14;j++){
                ofm1=0;
                for(k=0;k<728;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block11_sepconv3_bn(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W[4][728]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 728; m++){
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
void SW_add_10(DATA_T I1[728][14][14], DATA_T I2[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y;
	for (m = 0; m< 728; m++) {
		for (x = 0; x < 14; x++) {
			for(y = 0; y < 14; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_block12_sepconv1_act(DATA_T I[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<728; m++) {
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

void SW_block12_sepconv1(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W1[728][3][3], DATA_T W2[728][728]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<728; m++) {
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
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<728;m++){
        for(i=0;i<14;i++){
            for(j=0;j<14;j++){
                ofm1=0;
                for(k=0;k<728;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block12_sepconv1_bn(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W[4][728]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 728; m++){
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
void SW_block12_sepconv2_act(DATA_T I[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<728; m++) {
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

void SW_block12_sepconv2(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W1[728][3][3], DATA_T W2[728][728]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<728; m++) {
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
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<728;m++){
        for(i=0;i<14;i++){
            for(j=0;j<14;j++){
                ofm1=0;
                for(k=0;k<728;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block12_sepconv2_bn(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W[4][728]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 728; m++){
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
void SW_block12_sepconv3_act(DATA_T I[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<728; m++) {
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

void SW_block12_sepconv3(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W1[728][3][3], DATA_T W2[728][728]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<728; m++) {
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
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<728;m++){
        for(i=0;i<14;i++){
            for(j=0;j<14;j++){
                ofm1=0;
                for(k=0;k<728;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block12_sepconv3_bn(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W[4][728]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 728; m++){
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
void SW_add_11(DATA_T I1[728][14][14], DATA_T I2[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y;
	for (m = 0; m< 728; m++) {
		for (x = 0; x < 14; x++) {
			for(y = 0; y < 14; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_block13_sepconv1_act(DATA_T I[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<728; m++) {
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

void SW_block13_sepconv1(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W1[728][3][3], DATA_T W2[728][728]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<728; m++) {
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
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<728;m++){
        for(i=0;i<14;i++){
            for(j=0;j<14;j++){
                ofm1=0;
                for(k=0;k<728;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block13_sepconv1_bn(DATA_T I[728][14][14], DATA_T O[728][14][14], DATA_T W[4][728]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 728; m++){
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
void SW_block13_sepconv2_act(DATA_T I[728][14][14], DATA_T O[728][14][14]) {
	int m, x, y, i, j, k;
	for (m = 0; m<728; m++) {
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

void SW_block13_sepconv2(DATA_T I[728][14][14], DATA_T O[1024][14][14], DATA_T W1[728][3][3], DATA_T W2[1024][728]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
    int p = (1 *(14 - 1) - 14 + 3)/2;
	for (m = 0; m<728; m++) {
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
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<1024;m++){
        for(i=0;i<14;i++){
            for(j=0;j<14;j++){
                ofm1=0;
                for(k=0;k<728;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block13_sepconv2_bn(DATA_T I[1024][14][14], DATA_T O[1024][14][14], DATA_T W[4][1024]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_conv2d_207(DATA_T I[728][14][14], DATA_T O[1024][7][7], DATA_T W[1024][728][1][1]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (2 *(7 - 1) - 14 + 1)/2;
	for (m = 0; m<1024; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = 0;
				for (k = 0; k<728; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x*2 + i < 14 + p && y*2 + j < 14 + p && x*2 + i -p >= 0 && y*2 + j -p >= 0) {
                                    ifm = I[k][x*2 + i - p][y*2 + j -p];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}

				O[m][x][y] = ofm;
			}
		}
	}
}

void SW_block13_pool(DATA_T I[1024][14][14], DATA_T O[1024][7][7])
{
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<1024; m++) {
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
void SW_batch_normalization_207(DATA_T I[1024][7][7], DATA_T O[1024][7][7], DATA_T W[4][1024]) {
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
void SW_add_12(DATA_T I1[1024][7][7], DATA_T I2[1024][7][7], DATA_T O[1024][7][7]) {
	int m, x, y;
	for (m = 0; m< 1024; m++) {
		for (x = 0; x < 7; x++) {
			for(y = 0; y < 7; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_block14_sepconv1(DATA_T I[1024][7][7], DATA_T O[1536][7][7], DATA_T W1[1024][3][3], DATA_T W2[1536][1024]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
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
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<1536;m++){
        for(i=0;i<7;i++){
            for(j=0;j<7;j++){
                ofm1=0;
                for(k=0;k<1024;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block14_sepconv1_bn(DATA_T I[1536][7][7], DATA_T O[1536][7][7], DATA_T W[4][1536]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

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
void SW_block14_sepconv1_act(DATA_T I[1536][7][7], DATA_T O[1536][7][7]) {
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

void SW_block14_sepconv2(DATA_T I[1536][7][7], DATA_T O[2048][7][7], DATA_T W1[1536][3][3], DATA_T W2[2048][1536]) {
	int m, x, y, i, j, k;
	DATA_T ifm=0, ofm=0;
    int p = (1 *(7 - 1) - 7 + 3)/2;
	for (m = 0; m<1536; m++) {
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
						ofm = ofm + ifm * W1[m][i][j];
					}
				}
				I[m][x][y] = ofm;
			}
		}
	}
	DATA_T ofm1=0;

	for(m=0;m<2048;m++){
        for(i=0;i<7;i++){
            for(j=0;j<7;j++){
                ofm1=0;
                for(k=0;k<1536;k++){
	                ofm1+=I[k][i][j]*W2[m][k];
	            }
                O[m][i][j]=ofm1;

	        }
	    }
	}


}

void SW_block14_sepconv2_bn(DATA_T I[2048][7][7], DATA_T O[2048][7][7], DATA_T W[4][2048]) {
	int m, x, y;
	DATA_T epsilon = 0.001;

    for (m = 0; m < 2048; m++){
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
void SW_block14_sepconv2_act(DATA_T I[2048][7][7], DATA_T O[2048][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<2048; m++) {
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

void SW_avg_pool(DATA_T I[2048][7][7], DATA_T O[2048]) {
	int m, x, y;
	double avg;
	int div = 7 * 7;
	for (m = 0; m < 2048; m++){
		avg = 0;
		for (x = 0; x < 7; x++) {
			for (y = 0; y < 7; y++) {
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
	static DATA_T W2[4][32];
	static DATA_T W4[64][32][3][3];
	static DATA_T W5[4][64];
	static DATA_T W7_1[64][3][3];
	static DATA_T W7_2[128][64];
	static DATA_T W8[4][128];
	static DATA_T W10_1[128][3][3];
	static DATA_T W10_2[128][128];
	static DATA_T W11[4][128];
	static DATA_T W12[128][64][1][1];
	static DATA_T W14[4][128];
	static DATA_T W17_1[128][3][3];
	static DATA_T W17_2[256][128];
	static DATA_T W18[4][256];
	static DATA_T W20_1[256][3][3];
	static DATA_T W20_2[256][256];
	static DATA_T W21[4][256];
	static DATA_T W22[256][128][1][1];
	static DATA_T W24[4][256];
	static DATA_T W27_1[256][3][3];
	static DATA_T W27_2[728][256];
	static DATA_T W28[4][728];
	static DATA_T W30_1[728][3][3];
	static DATA_T W30_2[728][728];
	static DATA_T W31[4][728];
	static DATA_T W32[728][256][1][1];
	static DATA_T W34[4][728];
	static DATA_T W37_1[728][3][3];
	static DATA_T W37_2[728][728];
	static DATA_T W38[4][728];
	static DATA_T W40_1[728][3][3];
	static DATA_T W40_2[728][728];
	static DATA_T W41[4][728];
	static DATA_T W43_1[728][3][3];
	static DATA_T W43_2[728][728];
	static DATA_T W44[4][728];
	static DATA_T W47_1[728][3][3];
	static DATA_T W47_2[728][728];
	static DATA_T W48[4][728];
	static DATA_T W50_1[728][3][3];
	static DATA_T W50_2[728][728];
	static DATA_T W51[4][728];
	static DATA_T W53_1[728][3][3];
	static DATA_T W53_2[728][728];
	static DATA_T W54[4][728];
	static DATA_T W57_1[728][3][3];
	static DATA_T W57_2[728][728];
	static DATA_T W58[4][728];
	static DATA_T W60_1[728][3][3];
	static DATA_T W60_2[728][728];
	static DATA_T W61[4][728];
	static DATA_T W63_1[728][3][3];
	static DATA_T W63_2[728][728];
	static DATA_T W64[4][728];
	static DATA_T W67_1[728][3][3];
	static DATA_T W67_2[728][728];
	static DATA_T W68[4][728];
	static DATA_T W70_1[728][3][3];
	static DATA_T W70_2[728][728];
	static DATA_T W71[4][728];
	static DATA_T W73_1[728][3][3];
	static DATA_T W73_2[728][728];
	static DATA_T W74[4][728];
	static DATA_T W77_1[728][3][3];
	static DATA_T W77_2[728][728];
	static DATA_T W78[4][728];
	static DATA_T W80_1[728][3][3];
	static DATA_T W80_2[728][728];
	static DATA_T W81[4][728];
	static DATA_T W83_1[728][3][3];
	static DATA_T W83_2[728][728];
	static DATA_T W84[4][728];
	static DATA_T W87_1[728][3][3];
	static DATA_T W87_2[728][728];
	static DATA_T W88[4][728];
	static DATA_T W90_1[728][3][3];
	static DATA_T W90_2[728][728];
	static DATA_T W91[4][728];
	static DATA_T W93_1[728][3][3];
	static DATA_T W93_2[728][728];
	static DATA_T W94[4][728];
	static DATA_T W97_1[728][3][3];
	static DATA_T W97_2[728][728];
	static DATA_T W98[4][728];
	static DATA_T W100_1[728][3][3];
	static DATA_T W100_2[728][728];
	static DATA_T W101[4][728];
	static DATA_T W103_1[728][3][3];
	static DATA_T W103_2[728][728];
	static DATA_T W104[4][728];
	static DATA_T W107_1[728][3][3];
	static DATA_T W107_2[728][728];
	static DATA_T W108[4][728];
	static DATA_T W110_1[728][3][3];
	static DATA_T W110_2[728][728];
	static DATA_T W111[4][728];
	static DATA_T W113_1[728][3][3];
	static DATA_T W113_2[728][728];
	static DATA_T W114[4][728];
	static DATA_T W117_1[728][3][3];
	static DATA_T W117_2[728][728];
	static DATA_T W118[4][728];
	static DATA_T W120_1[728][3][3];
	static DATA_T W120_2[1024][728];
	static DATA_T W121[4][1024];
	static DATA_T W122[1024][728][1][1];
	static DATA_T W124[4][1024];
	static DATA_T W126_1[1024][3][3];
	static DATA_T W126_2[1536][1024];
	static DATA_T W127[4][1536];
	static DATA_T W129_1[1536][3][3];
	static DATA_T W129_2[2048][1536];
	static DATA_T W130[4][2048];
	static DATA_T B133[1000];
	static DATA_T W133[1000][2048];
	

    static DATA_T O0_SW[3][224][224];
	static DATA_T O1_SW[32][111][111];
	static DATA_T O2_SW[32][111][111];
	static DATA_T O3_SW[32][111][111];
	static DATA_T O4_SW[64][109][109];
	static DATA_T O5_SW[64][109][109];
	static DATA_T O6_SW[64][109][109];
	static DATA_T O7_SW[128][109][109];
	static DATA_T O8_SW[128][109][109];
	static DATA_T O9_SW[128][109][109];
	static DATA_T O10_SW[128][109][109];
	static DATA_T O11_SW[128][109][109];
	static DATA_T O12_SW[128][55][55];
	static DATA_T O13_SW[128][55][55];
	static DATA_T O14_SW[128][55][55];
	static DATA_T O15_SW[128][55][55];
	static DATA_T O16_SW[128][55][55];
	static DATA_T O17_SW[256][55][55];
	static DATA_T O18_SW[256][55][55];
	static DATA_T O19_SW[256][55][55];
	static DATA_T O20_SW[256][55][55];
	static DATA_T O21_SW[256][55][55];
	static DATA_T O22_SW[256][28][28];
	static DATA_T O23_SW[256][28][28];
	static DATA_T O24_SW[256][28][28];
	static DATA_T O25_SW[256][28][28];
	static DATA_T O26_SW[256][28][28];
	static DATA_T O27_SW[728][28][28];
	static DATA_T O28_SW[728][28][28];
	static DATA_T O29_SW[728][28][28];
	static DATA_T O30_SW[728][28][28];
	static DATA_T O31_SW[728][28][28];
	static DATA_T O32_SW[728][14][14];
	static DATA_T O33_SW[728][14][14];
	static DATA_T O34_SW[728][14][14];
	static DATA_T O35_SW[728][14][14];
	static DATA_T O36_SW[728][14][14];
	static DATA_T O37_SW[728][14][14];
	static DATA_T O38_SW[728][14][14];
	static DATA_T O39_SW[728][14][14];
	static DATA_T O40_SW[728][14][14];
	static DATA_T O41_SW[728][14][14];
	static DATA_T O42_SW[728][14][14];
	static DATA_T O43_SW[728][14][14];
	static DATA_T O44_SW[728][14][14];
	static DATA_T O45_SW[728][14][14];
	static DATA_T O46_SW[728][14][14];
	static DATA_T O47_SW[728][14][14];
	static DATA_T O48_SW[728][14][14];
	static DATA_T O49_SW[728][14][14];
	static DATA_T O50_SW[728][14][14];
	static DATA_T O51_SW[728][14][14];
	static DATA_T O52_SW[728][14][14];
	static DATA_T O53_SW[728][14][14];
	static DATA_T O54_SW[728][14][14];
	static DATA_T O55_SW[728][14][14];
	static DATA_T O56_SW[728][14][14];
	static DATA_T O57_SW[728][14][14];
	static DATA_T O58_SW[728][14][14];
	static DATA_T O59_SW[728][14][14];
	static DATA_T O60_SW[728][14][14];
	static DATA_T O61_SW[728][14][14];
	static DATA_T O62_SW[728][14][14];
	static DATA_T O63_SW[728][14][14];
	static DATA_T O64_SW[728][14][14];
	static DATA_T O65_SW[728][14][14];
	static DATA_T O66_SW[728][14][14];
	static DATA_T O67_SW[728][14][14];
	static DATA_T O68_SW[728][14][14];
	static DATA_T O69_SW[728][14][14];
	static DATA_T O70_SW[728][14][14];
	static DATA_T O71_SW[728][14][14];
	static DATA_T O72_SW[728][14][14];
	static DATA_T O73_SW[728][14][14];
	static DATA_T O74_SW[728][14][14];
	static DATA_T O75_SW[728][14][14];
	static DATA_T O76_SW[728][14][14];
	static DATA_T O77_SW[728][14][14];
	static DATA_T O78_SW[728][14][14];
	static DATA_T O79_SW[728][14][14];
	static DATA_T O80_SW[728][14][14];
	static DATA_T O81_SW[728][14][14];
	static DATA_T O82_SW[728][14][14];
	static DATA_T O83_SW[728][14][14];
	static DATA_T O84_SW[728][14][14];
	static DATA_T O85_SW[728][14][14];
	static DATA_T O86_SW[728][14][14];
	static DATA_T O87_SW[728][14][14];
	static DATA_T O88_SW[728][14][14];
	static DATA_T O89_SW[728][14][14];
	static DATA_T O90_SW[728][14][14];
	static DATA_T O91_SW[728][14][14];
	static DATA_T O92_SW[728][14][14];
	static DATA_T O93_SW[728][14][14];
	static DATA_T O94_SW[728][14][14];
	static DATA_T O95_SW[728][14][14];
	static DATA_T O96_SW[728][14][14];
	static DATA_T O97_SW[728][14][14];
	static DATA_T O98_SW[728][14][14];
	static DATA_T O99_SW[728][14][14];
	static DATA_T O100_SW[728][14][14];
	static DATA_T O101_SW[728][14][14];
	static DATA_T O102_SW[728][14][14];
	static DATA_T O103_SW[728][14][14];
	static DATA_T O104_SW[728][14][14];
	static DATA_T O105_SW[728][14][14];
	static DATA_T O106_SW[728][14][14];
	static DATA_T O107_SW[728][14][14];
	static DATA_T O108_SW[728][14][14];
	static DATA_T O109_SW[728][14][14];
	static DATA_T O110_SW[728][14][14];
	static DATA_T O111_SW[728][14][14];
	static DATA_T O112_SW[728][14][14];
	static DATA_T O113_SW[728][14][14];
	static DATA_T O114_SW[728][14][14];
	static DATA_T O115_SW[728][14][14];
	static DATA_T O116_SW[728][14][14];
	static DATA_T O117_SW[728][14][14];
	static DATA_T O118_SW[728][14][14];
	static DATA_T O119_SW[728][14][14];
	static DATA_T O120_SW[1024][14][14];
	static DATA_T O121_SW[1024][14][14];
	static DATA_T O122_SW[1024][7][7];
	static DATA_T O123_SW[1024][7][7];
	static DATA_T O124_SW[1024][7][7];
	static DATA_T O125_SW[1024][7][7];
	static DATA_T O126_SW[1536][7][7];
	static DATA_T O127_SW[1536][7][7];
	static DATA_T O128_SW[1536][7][7];
	static DATA_T O129_SW[2048][7][7];
	static DATA_T O130_SW[2048][7][7];
	static DATA_T O131_SW[2048][7][7];
	static DATA_T O132_SW[2048];
	static DATA_T O133_SW[1000];
	

    FILE *w_stream = fopen(argv[1], "rb");
    if (w_stream == NULL) printf("weight file was not opened");
    FILE *i_stream = fopen(argv[2], "rb");
    if (i_stream == NULL) printf("input file was not opened");
    FILE *o_stream = fopen("../../cpp_generator/xception/Output/C_output.txt", "w");
    if (o_stream == NULL) printf("Output file was not opened");
    FILE *c_num = fopen("../../cpp_generator/xception/Output/c_output_num.txt", "w");
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
		for (i = 0; i < 32 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W4[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B4[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 64 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W5[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 64 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W7_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 64 ; m++) {
    for (k = 0; k < 128 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W7_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W8[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W10_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 128 ; m++) {
    for (k = 0; k < 128 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W10_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W11[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W12[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B12[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 128 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W14[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W17_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 128 ; m++) {
    for (k = 0; k < 256 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W17_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W18[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W20_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 256 ; m++) {
    for (k = 0; k < 256 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W20_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W21[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W22[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B22[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 256 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W24[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W27_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 256 ; m++) {
    for (k = 0; k < 728 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W27_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 728 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W28[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 728 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W30_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 728 ; m++) {
    for (k = 0; k < 728 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W30_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 728 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W31[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 728 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W32[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 728 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B32[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 728 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W34[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 728 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W37_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 728 ; m++) {
    for (k = 0; k < 728 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W37_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 728 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W38[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 728 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W40_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 728 ; m++) {
    for (k = 0; k < 728 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W40_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 728 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W41[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 728 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W43_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 728 ; m++) {
    for (k = 0; k < 728 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W43_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 728 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W44[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 728 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W47_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 728 ; m++) {
    for (k = 0; k < 728 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W47_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 728 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W48[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 728 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W50_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 728 ; m++) {
    for (k = 0; k < 728 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W50_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 728 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W51[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 728 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W53_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 728 ; m++) {
    for (k = 0; k < 728 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W53_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 728 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W54[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 728 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W57_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 728 ; m++) {
    for (k = 0; k < 728 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W57_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 728 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W58[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 728 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W60_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 728 ; m++) {
    for (k = 0; k < 728 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W60_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 728 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W61[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 728 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W63_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 728 ; m++) {
    for (k = 0; k < 728 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W63_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 728 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W64[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 728 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W67_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 728 ; m++) {
    for (k = 0; k < 728 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W67_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 728 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W68[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 728 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W70_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 728 ; m++) {
    for (k = 0; k < 728 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W70_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 728 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W71[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 728 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W73_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 728 ; m++) {
    for (k = 0; k < 728 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W73_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 728 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W74[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 728 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W77_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 728 ; m++) {
    for (k = 0; k < 728 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W77_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 728 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W78[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 728 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W80_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 728 ; m++) {
    for (k = 0; k < 728 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W80_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 728 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W81[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 728 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W83_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 728 ; m++) {
    for (k = 0; k < 728 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W83_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 728 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W84[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 728 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W87_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 728 ; m++) {
    for (k = 0; k < 728 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W87_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 728 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W88[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 728 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W90_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 728 ; m++) {
    for (k = 0; k < 728 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W90_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 728 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W91[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 728 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W93_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 728 ; m++) {
    for (k = 0; k < 728 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W93_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 728 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W94[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 728 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W97_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 728 ; m++) {
    for (k = 0; k < 728 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W97_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 728 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W98[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 728 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W100_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 728 ; m++) {
    for (k = 0; k < 728 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W100_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 728 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W101[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 728 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W103_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 728 ; m++) {
    for (k = 0; k < 728 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W103_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 728 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W104[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 728 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W107_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 728 ; m++) {
    for (k = 0; k < 728 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W107_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 728 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W108[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 728 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W110_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 728 ; m++) {
    for (k = 0; k < 728 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W110_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 728 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W111[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 728 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W113_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 728 ; m++) {
    for (k = 0; k < 728 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W113_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 728 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W114[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 728 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W117_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 728 ; m++) {
    for (k = 0; k < 728 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W117_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 728 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W118[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 728 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W120_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 728 ; m++) {
    for (k = 0; k < 1024 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W120_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1024 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W121[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 728 ; i++) {
			for (j = 0; j < 1024 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W122[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

/*
for (m = 0; m < 1024 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B122[m] = (DATA_T) trash;
}
*/

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1024 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W124[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 1024 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W126_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 1024 ; m++) {
    for (k = 0; k < 1536 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W126_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 1536 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W127[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 1536 ; i++) {


				fread(&trash, sizeof(int), 1, w_stream);
               	W129_1[i][m][k] = (DATA_T) trash;
		}
	}
}

for (m = 0; m < 1536 ; m++) {
    for (k = 0; k < 2048 ; k++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W129_2[k][m] = (DATA_T) trash;
    }
}

	for (x = 0; x < 4; x++) {
    for (y = 0; y < 2048 ; y++) {
        fread(&trash, sizeof(int), 1, w_stream);
        W130[x][y] = (DATA_T) trash;
    }
}
	for (m = 0; m <  2048 ; m++) {
	for (k = 0; k < 1000 ; k++) {
		fread(&trash, sizeof(int), 1, w_stream);
        W133[k][m] = (DATA_T) trash;
	}
}


for (m = 0; m < 1000 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B133[m] = (DATA_T) trash;
}

	
    printf("[C_verifier.cpp]Finish Initialization");

    printf("[C_verifier.cpp]InputLayer\n\n");
	printf("[C_verifier.cpp]Calculate Conv2D1\n\n");
	SW_block1_conv1(O0_SW,O1_SW,W1);
	printf("[C_verifier.cpp]Calculate BatchNormalization2\n\n");
	SW_block1_conv1_bn(O1_SW,O2_SW, W2);
	printf("[C_verifier.cpp]Calculate Activation(Relu)3\n\n");
	SW_block1_conv1_act(O2_SW,O3_SW);
	printf("[C_verifier.cpp]Calculate Conv2D4\n\n");
	SW_block1_conv2(O3_SW,O4_SW,W4);
	printf("[C_verifier.cpp]Calculate BatchNormalization5\n\n");
	SW_block1_conv2_bn(O4_SW,O5_SW, W5);
	printf("[C_verifier.cpp]Calculate Activation(Relu)6\n\n");
	SW_block1_conv2_act(O5_SW,O6_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D7\n\n");
	SW_block2_sepconv1(O6_SW,O7_SW,W7_1,W7_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization8\n\n");
	SW_block2_sepconv1_bn(O7_SW,O8_SW, W8);
	printf("[C_verifier.cpp]Calculate Activation(Relu)9\n\n");
	SW_block2_sepconv2_act(O8_SW,O9_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D10\n\n");
	SW_block2_sepconv2(O9_SW,O10_SW,W10_1,W10_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization11\n\n");
	SW_block2_sepconv2_bn(O10_SW,O11_SW, W11);
	printf("[C_verifier.cpp]Calculate Conv2D12\n\n");
	SW_conv2d_204(O6_SW,O12_SW,W12);
	printf("[C_verifier.cpp]Calculate MaxPooling2D13\n\n");
	SW_block2_pool(O11_SW,O13_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization14\n\n");
	SW_batch_normalization_204(O12_SW,O14_SW, W14);
	printf("[C_verifier.cpp]Calculate Add15\n\n");
	SW_add_1(O13_SW,O14_SW,O15_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)16\n\n");
	SW_block3_sepconv1_act(O15_SW,O16_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D17\n\n");
	SW_block3_sepconv1(O16_SW,O17_SW,W17_1,W17_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization18\n\n");
	SW_block3_sepconv1_bn(O17_SW,O18_SW, W18);
	printf("[C_verifier.cpp]Calculate Activation(Relu)19\n\n");
	SW_block3_sepconv2_act(O18_SW,O19_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D20\n\n");
	SW_block3_sepconv2(O19_SW,O20_SW,W20_1,W20_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization21\n\n");
	SW_block3_sepconv2_bn(O20_SW,O21_SW, W21);
	printf("[C_verifier.cpp]Calculate Conv2D22\n\n");
	SW_conv2d_205(O15_SW,O22_SW,W22);
	printf("[C_verifier.cpp]Calculate MaxPooling2D23\n\n");
	SW_block3_pool(O21_SW,O23_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization24\n\n");
	SW_batch_normalization_205(O22_SW,O24_SW, W24);
	printf("[C_verifier.cpp]Calculate Add25\n\n");
	SW_add_2(O23_SW,O24_SW,O25_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)26\n\n");
	SW_block4_sepconv1_act(O25_SW,O26_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D27\n\n");
	SW_block4_sepconv1(O26_SW,O27_SW,W27_1,W27_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization28\n\n");
	SW_block4_sepconv1_bn(O27_SW,O28_SW, W28);
	printf("[C_verifier.cpp]Calculate Activation(Relu)29\n\n");
	SW_block4_sepconv2_act(O28_SW,O29_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D30\n\n");
	SW_block4_sepconv2(O29_SW,O30_SW,W30_1,W30_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization31\n\n");
	SW_block4_sepconv2_bn(O30_SW,O31_SW, W31);
	printf("[C_verifier.cpp]Calculate Conv2D32\n\n");
	SW_conv2d_206(O25_SW,O32_SW,W32);
	printf("[C_verifier.cpp]Calculate MaxPooling2D33\n\n");
	SW_block4_pool(O31_SW,O33_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization34\n\n");
	SW_batch_normalization_206(O32_SW,O34_SW, W34);
	printf("[C_verifier.cpp]Calculate Add35\n\n");
	SW_add_3(O33_SW,O34_SW,O35_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)36\n\n");
	SW_block5_sepconv1_act(O35_SW,O36_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D37\n\n");
	SW_block5_sepconv1(O36_SW,O37_SW,W37_1,W37_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization38\n\n");
	SW_block5_sepconv1_bn(O37_SW,O38_SW, W38);
	printf("[C_verifier.cpp]Calculate Activation(Relu)39\n\n");
	SW_block5_sepconv2_act(O38_SW,O39_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D40\n\n");
	SW_block5_sepconv2(O39_SW,O40_SW,W40_1,W40_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization41\n\n");
	SW_block5_sepconv2_bn(O40_SW,O41_SW, W41);
	printf("[C_verifier.cpp]Calculate Activation(Relu)42\n\n");
	SW_block5_sepconv3_act(O41_SW,O42_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D43\n\n");
	SW_block5_sepconv3(O42_SW,O43_SW,W43_1,W43_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization44\n\n");
	SW_block5_sepconv3_bn(O43_SW,O44_SW, W44);
	printf("[C_verifier.cpp]Calculate Add45\n\n");
	SW_add_4(O44_SW,O35_SW,O45_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)46\n\n");
	SW_block6_sepconv1_act(O45_SW,O46_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D47\n\n");
	SW_block6_sepconv1(O46_SW,O47_SW,W47_1,W47_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization48\n\n");
	SW_block6_sepconv1_bn(O47_SW,O48_SW, W48);
	printf("[C_verifier.cpp]Calculate Activation(Relu)49\n\n");
	SW_block6_sepconv2_act(O48_SW,O49_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D50\n\n");
	SW_block6_sepconv2(O49_SW,O50_SW,W50_1,W50_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization51\n\n");
	SW_block6_sepconv2_bn(O50_SW,O51_SW, W51);
	printf("[C_verifier.cpp]Calculate Activation(Relu)52\n\n");
	SW_block6_sepconv3_act(O51_SW,O52_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D53\n\n");
	SW_block6_sepconv3(O52_SW,O53_SW,W53_1,W53_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization54\n\n");
	SW_block6_sepconv3_bn(O53_SW,O54_SW, W54);
	printf("[C_verifier.cpp]Calculate Add55\n\n");
	SW_add_5(O54_SW,O45_SW,O55_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)56\n\n");
	SW_block7_sepconv1_act(O55_SW,O56_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D57\n\n");
	SW_block7_sepconv1(O56_SW,O57_SW,W57_1,W57_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization58\n\n");
	SW_block7_sepconv1_bn(O57_SW,O58_SW, W58);
	printf("[C_verifier.cpp]Calculate Activation(Relu)59\n\n");
	SW_block7_sepconv2_act(O58_SW,O59_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D60\n\n");
	SW_block7_sepconv2(O59_SW,O60_SW,W60_1,W60_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization61\n\n");
	SW_block7_sepconv2_bn(O60_SW,O61_SW, W61);
	printf("[C_verifier.cpp]Calculate Activation(Relu)62\n\n");
	SW_block7_sepconv3_act(O61_SW,O62_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D63\n\n");
	SW_block7_sepconv3(O62_SW,O63_SW,W63_1,W63_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization64\n\n");
	SW_block7_sepconv3_bn(O63_SW,O64_SW, W64);
	printf("[C_verifier.cpp]Calculate Add65\n\n");
	SW_add_6(O64_SW,O55_SW,O65_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)66\n\n");
	SW_block8_sepconv1_act(O65_SW,O66_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D67\n\n");
	SW_block8_sepconv1(O66_SW,O67_SW,W67_1,W67_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization68\n\n");
	SW_block8_sepconv1_bn(O67_SW,O68_SW, W68);
	printf("[C_verifier.cpp]Calculate Activation(Relu)69\n\n");
	SW_block8_sepconv2_act(O68_SW,O69_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D70\n\n");
	SW_block8_sepconv2(O69_SW,O70_SW,W70_1,W70_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization71\n\n");
	SW_block8_sepconv2_bn(O70_SW,O71_SW, W71);
	printf("[C_verifier.cpp]Calculate Activation(Relu)72\n\n");
	SW_block8_sepconv3_act(O71_SW,O72_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D73\n\n");
	SW_block8_sepconv3(O72_SW,O73_SW,W73_1,W73_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization74\n\n");
	SW_block8_sepconv3_bn(O73_SW,O74_SW, W74);
	printf("[C_verifier.cpp]Calculate Add75\n\n");
	SW_add_7(O74_SW,O65_SW,O75_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)76\n\n");
	SW_block9_sepconv1_act(O75_SW,O76_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D77\n\n");
	SW_block9_sepconv1(O76_SW,O77_SW,W77_1,W77_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization78\n\n");
	SW_block9_sepconv1_bn(O77_SW,O78_SW, W78);
	printf("[C_verifier.cpp]Calculate Activation(Relu)79\n\n");
	SW_block9_sepconv2_act(O78_SW,O79_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D80\n\n");
	SW_block9_sepconv2(O79_SW,O80_SW,W80_1,W80_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization81\n\n");
	SW_block9_sepconv2_bn(O80_SW,O81_SW, W81);
	printf("[C_verifier.cpp]Calculate Activation(Relu)82\n\n");
	SW_block9_sepconv3_act(O81_SW,O82_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D83\n\n");
	SW_block9_sepconv3(O82_SW,O83_SW,W83_1,W83_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization84\n\n");
	SW_block9_sepconv3_bn(O83_SW,O84_SW, W84);
	printf("[C_verifier.cpp]Calculate Add85\n\n");
	SW_add_8(O84_SW,O75_SW,O85_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)86\n\n");
	SW_block10_sepconv1_act(O85_SW,O86_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D87\n\n");
	SW_block10_sepconv1(O86_SW,O87_SW,W87_1,W87_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization88\n\n");
	SW_block10_sepconv1_bn(O87_SW,O88_SW, W88);
	printf("[C_verifier.cpp]Calculate Activation(Relu)89\n\n");
	SW_block10_sepconv2_act(O88_SW,O89_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D90\n\n");
	SW_block10_sepconv2(O89_SW,O90_SW,W90_1,W90_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization91\n\n");
	SW_block10_sepconv2_bn(O90_SW,O91_SW, W91);
	printf("[C_verifier.cpp]Calculate Activation(Relu)92\n\n");
	SW_block10_sepconv3_act(O91_SW,O92_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D93\n\n");
	SW_block10_sepconv3(O92_SW,O93_SW,W93_1,W93_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization94\n\n");
	SW_block10_sepconv3_bn(O93_SW,O94_SW, W94);
	printf("[C_verifier.cpp]Calculate Add95\n\n");
	SW_add_9(O94_SW,O85_SW,O95_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)96\n\n");
	SW_block11_sepconv1_act(O95_SW,O96_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D97\n\n");
	SW_block11_sepconv1(O96_SW,O97_SW,W97_1,W97_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization98\n\n");
	SW_block11_sepconv1_bn(O97_SW,O98_SW, W98);
	printf("[C_verifier.cpp]Calculate Activation(Relu)99\n\n");
	SW_block11_sepconv2_act(O98_SW,O99_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D100\n\n");
	SW_block11_sepconv2(O99_SW,O100_SW,W100_1,W100_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization101\n\n");
	SW_block11_sepconv2_bn(O100_SW,O101_SW, W101);
	printf("[C_verifier.cpp]Calculate Activation(Relu)102\n\n");
	SW_block11_sepconv3_act(O101_SW,O102_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D103\n\n");
	SW_block11_sepconv3(O102_SW,O103_SW,W103_1,W103_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization104\n\n");
	SW_block11_sepconv3_bn(O103_SW,O104_SW, W104);
	printf("[C_verifier.cpp]Calculate Add105\n\n");
	SW_add_10(O104_SW,O95_SW,O105_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)106\n\n");
	SW_block12_sepconv1_act(O105_SW,O106_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D107\n\n");
	SW_block12_sepconv1(O106_SW,O107_SW,W107_1,W107_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization108\n\n");
	SW_block12_sepconv1_bn(O107_SW,O108_SW, W108);
	printf("[C_verifier.cpp]Calculate Activation(Relu)109\n\n");
	SW_block12_sepconv2_act(O108_SW,O109_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D110\n\n");
	SW_block12_sepconv2(O109_SW,O110_SW,W110_1,W110_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization111\n\n");
	SW_block12_sepconv2_bn(O110_SW,O111_SW, W111);
	printf("[C_verifier.cpp]Calculate Activation(Relu)112\n\n");
	SW_block12_sepconv3_act(O111_SW,O112_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D113\n\n");
	SW_block12_sepconv3(O112_SW,O113_SW,W113_1,W113_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization114\n\n");
	SW_block12_sepconv3_bn(O113_SW,O114_SW, W114);
	printf("[C_verifier.cpp]Calculate Add115\n\n");
	SW_add_11(O114_SW,O105_SW,O115_SW);
	printf("[C_verifier.cpp]Calculate Activation(Relu)116\n\n");
	SW_block13_sepconv1_act(O115_SW,O116_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D117\n\n");
	SW_block13_sepconv1(O116_SW,O117_SW,W117_1,W117_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization118\n\n");
	SW_block13_sepconv1_bn(O117_SW,O118_SW, W118);
	printf("[C_verifier.cpp]Calculate Activation(Relu)119\n\n");
	SW_block13_sepconv2_act(O118_SW,O119_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D120\n\n");
	SW_block13_sepconv2(O119_SW,O120_SW,W120_1,W120_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization121\n\n");
	SW_block13_sepconv2_bn(O120_SW,O121_SW, W121);
	printf("[C_verifier.cpp]Calculate Conv2D122\n\n");
	SW_conv2d_207(O115_SW,O122_SW,W122);
	printf("[C_verifier.cpp]Calculate MaxPooling2D123\n\n");
	SW_block13_pool(O121_SW,O123_SW);
	printf("[C_verifier.cpp]Calculate BatchNormalization124\n\n");
	SW_batch_normalization_207(O122_SW,O124_SW, W124);
	printf("[C_verifier.cpp]Calculate Add125\n\n");
	SW_add_12(O123_SW,O124_SW,O125_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D126\n\n");
	SW_block14_sepconv1(O125_SW,O126_SW,W126_1,W126_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization127\n\n");
	SW_block14_sepconv1_bn(O126_SW,O127_SW, W127);
	printf("[C_verifier.cpp]Calculate Activation(Relu)128\n\n");
	SW_block14_sepconv1_act(O127_SW,O128_SW);
	printf("[C_verifier.cpp]Calculate SeparableConv2D129\n\n");
	SW_block14_sepconv2(O128_SW,O129_SW,W129_1,W129_2);
	printf("[C_verifier.cpp]Calculate BatchNormalization130\n\n");
	SW_block14_sepconv2_bn(O129_SW,O130_SW, W130);
	printf("[C_verifier.cpp]Calculate Activation(Relu)131\n\n");
	SW_block14_sepconv2_act(O130_SW,O131_SW);
	printf("[C_verifier.cpp]Calculate GlobalAveragePooling2D132\n\n");
	SW_avg_pool(O131_SW,O132_SW);
	printf("[C_verifier.cpp]Calculate Dense133\n\n");
	SW_predictions(O132_SW,O133_SW,W133,B133);
	

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
		for(y = 0; y < 64 ; y++) {
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
		for(y = 0; y < 64 ; y++) {
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
		for(y = 0; y < 64 ; y++) {
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


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 109 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 109 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
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
		for(y = 0; y < 128 ; y++) {
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
		for(y = 0; y < 128 ; y++) {
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


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 109 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 109 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O10_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O10_SW[y][k][x]);
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
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O11_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O11_SW[y][k][x]);
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
for (k = 0; k < 55 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 55 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O12_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O12_SW[y][k][x]);
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
for (k = 0; k < 55 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 55 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O13_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O13_SW[y][k][x]);
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 55 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 55 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O14_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O14_SW[y][k][x]);
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


fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 55 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 55 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O15_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O15_SW[y][k][x]);
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 55 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 55 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 128 ; y++) {
			fprintf(o_stream,"%.6f ",O16_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O16_SW[y][k][x]);
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


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 55 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 55 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O17_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O17_SW[y][k][x]);
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 55 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 55 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O18_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O18_SW[y][k][x]);
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 55 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 55 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O19_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O19_SW[y][k][x]);
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


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 55 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 55 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O20_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O20_SW[y][k][x]);
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 55 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 55 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O21_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O21_SW[y][k][x]);
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
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O22_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O22_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O23_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O23_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O24_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O24_SW[y][k][x]);
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
		for(y = 0; y < 256 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
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


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 28 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 28 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","MaxPooling2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
			fprintf(o_stream,"%.6f ",O116_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O116_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 728 ; y++) {
			fprintf(o_stream,"%.6f ",O117_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O117_SW[y][k][x]);
		}
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
		for(y = 0; y < 728 ; y++) {
			fprintf(o_stream,"%.6f ",O118_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O118_SW[y][k][x]);
		}
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
		for(y = 0; y < 728 ; y++) {
			fprintf(o_stream,"%.6f ",O119_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O119_SW[y][k][x]);
		}
		if(x != 14 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 14 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 14 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 14 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
			fprintf(o_stream,"%.6f ",O120_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O120_SW[y][k][x]);
		}
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
			fprintf(o_stream,"%.6f ",O121_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O121_SW[y][k][x]);
		}
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
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
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


fprintf(o_stream,"%s","MaxPooling2D : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
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


fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1024 ; y++) {
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


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1536 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1536 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 1536 ; y++) {
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


fprintf(o_stream,"%s","SeparableConv2D : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2048 ; y++) {
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


fprintf(o_stream,"%s","BatchNormalization : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2048 ; y++) {
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


fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 2048 ; y++) {
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


fprintf(o_stream,"%s","GlobalAveragePooling2D : [[");
for (k = 0; k <  2048 ; k++) {
	fprintf(o_stream,"%.6f ",O132_SW[k]);
	fprintf(c_num,"%.6f ",O132_SW[k]);
}
fprintf(o_stream,"%s","]]\n\n");


fprintf(o_stream,"%s","Dense : [[");
for (k = 0; k <  1000 ; k++) {
	fprintf(o_stream,"%.6f ",O133_SW[k]);
	fprintf(c_num,"%.6f ",O133_SW[k]);
}
fprintf(o_stream,"%s","]]\n\n");





    fclose(w_stream);
    fclose(i_stream);
    fclose(o_stream);
    fclose(c_num);

    return 0;
}
