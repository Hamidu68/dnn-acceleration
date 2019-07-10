#include <iostream>
#include <ap_int.h>
#include <hls_stream.h>
typedef ap_int<80> DATA_T;

void Stream_input(DATA_T I[3][32][32], hls::stream<DATA_T> &I_strm) {
	int k, x, y;

#pragma HLS ARRAY_PARTITION variable=I complete dim=1
	Stream_input_x_loop: for (x = 0; x<32; x++) {
	Stream_input_y_loop: for (y = 0; y<32; y++) {
#pragma HLS PIPELINE
		Stream_input_m_loop : for (k = 0; k<3; k++) {
			I_strm.write(I[k][x][y]);
		}
	}
	}
}

void Stream_output(hls::stream<DATA_T> &O_strm, DATA_T O[10]) {
	int x;
#pragma HLS ARRAY_PARTITION variable=O complete dim=1

	Stream_input_x_loop: for (x = 0; x<10; x++) {

#pragma HLS PIPELINE
		O[x] = O_strm.read();
	}
}


static DATA_T I_i[3][32][32];
static DATA_T W1_i[64][3][3][3];
static DATA_T B1_i[64];
static DATA_T W3_i[64][64][3][3];
static DATA_T B3_i[64];
static DATA_T W4_i[64][64][3][3];
static DATA_T B4_i[64];
static DATA_T W5_i[64][64];
static DATA_T B5_i[64];
static DATA_T W8_i[64][64][3][3];
static DATA_T B8_i[64];
static DATA_T W10_i[64][64][3][3];
static DATA_T B10_i[64];
static DATA_T W13_i[128][64][3][3];
static DATA_T B13_i[128];
static DATA_T W15_i[128][128][3][3];
static DATA_T B15_i[128];
static DATA_T W16_i[128][64];
static DATA_T B16_i[128];
static DATA_T W19_i[128][128][3][3];
static DATA_T B19_i[128];
static DATA_T W21_i[128][128][3][3];
static DATA_T B21_i[128];
static DATA_T W24_i[256][128][3][3];
static DATA_T B24_i[256];
static DATA_T W26_i[256][256][3][3];
static DATA_T B26_i[256];
static DATA_T W27_i[256][128];
static DATA_T B27_i[256];
static DATA_T W30_i[256][256][3][3];
static DATA_T B30_i[256];
static DATA_T W32_i[256][256][3][3];
static DATA_T B32_i[256];
static DATA_T W35_i[512][256][3][3];
static DATA_T B35_i[512];
static DATA_T W37_i[512][512][3][3];
static DATA_T B37_i[512];
static DATA_T W38_i[512][256];
static DATA_T B38_i[512];
static DATA_T W41_i[512][512][3][3];
static DATA_T B41_i[512];
static DATA_T W43_i[512][512][3][3];
static DATA_T B43_i[512];
static DATA_T W48_i[10][512];
static DATA_T B48_i[10];


void HW_conv2d_1(hls::stream<DATA_T> &I_strm, DATA_T W[64][3][3][3], DATA_T B[64], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ofm[64];
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[3][3][32];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_1_x_loop: for (x = 0; x<32 + 1; x++) {
	HW_conv2d_1_y_loop: for (y = 0; y<32 + 1; y++) {
	HW_conv2d_1_k_loop: for (k = 0; k<3; k++) {
		if (x < 32 && y < 32) {
			I[k][x % 3][y] = I_strm.read();
		}
		if (x && y) {
#pragma HLS PIPELINE
			if (x - 2 + 0 < 32 && y - 2 + 0 < 32 && x - 2 + 0 >= 0 && y - 2 + 0 >= 0) {
				I_0 = I[k][(x - 2) % 3][(y - 2)];
			}
			else {
				I_0 = 0; // zero padding
			}
			if (x - 2 + 0 < 32 && y - 2 + 1 < 32 && x - 2 + 0 >= 0 && y - 2 + 1 >= 0) {
				I_1 = I[k][(x - 2) % 3][(y - 1)];
			}
			else {
				I_1 = 0; // zero padding
			}
			if (x - 2 + 0 < 32 && y - 2 + 2 < 32 && x - 2 + 0 >= 0 && y - 2 + 2 >= 0) {
				I_2 = I[k][(x - 2) % 3][(y)];
			}
			else {
				I_2 = 0; // zero padding
			}

			if (x - 2 + 1 < 32 && y - 2 + 0 < 32 && x - 2 + 1 >= 0 && y - 2 + 0 >= 0) {
				I_3 = I[k][(x - 1) % 3][(y - 2)];
			}
			else {
				I_3 = 0; // zero padding
			}
			if (x - 2 + 1 < 32 && y - 2 + 1 < 32 && x - 2 + 1 >= 0 && y - 2 + 1 >= 0) {
				I_4 = I[k][(x - 1) % 3][(y - 1)];
			}
			else {
				I_4 = 0; // zero padding
			}
			if (x - 2 + 1 < 32 && y - 2 + 2 < 32 && x - 2 + 1 >= 0 && y - 2 + 2 >= 0) {
				I_5 = I[k][(x - 1) % 3][(y)];
			}
			else {
				I_5 = 0; // zero padding
			}
			if (x - 2 + 2 < 32 && y - 2 + 0 < 32 && x - 2 + 2 >= 0 && y - 2 + 0 >= 0) {
				I_6 = I[k][(x) % 3][(y - 2)];
			}
			else {
				I_6 = 0; // zero padding
			}
			if (x - 2 + 2 < 32 && y - 2 + 1 < 32 && x - 2 + 2 >= 0 && y - 2 + 1 >= 0) {
				I_7 = I[k][(x) % 3][(y - 1)];
			}
			else {
				I_7 = 0; // zero padding
			}
			if (x - 2 + 2 < 32 && y - 2 + 2 < 32 && x - 2 + 2 >= 0 && y - 2 + 2 >= 0) {
				I_8 = I[k][(x) % 3][(y)];
			}
			else {
				I_8 = 0; // zero padding
			}

		HW_conv2d_1_m_loop: for (m = 0; m<64; m++) {
#pragma HLS PIPELINE
			if (k == 0) {
				ofm[m] = B[m];
			}
			ofm[m] = ofm[m] + I_0 * W[m][k][0][0];
			ofm[m] = ofm[m] + I_1 * W[m][k][0][1];
			ofm[m] = ofm[m] + I_2 * W[m][k][0][2];
			ofm[m] = ofm[m] + I_3 * W[m][k][1][0];
			ofm[m] = ofm[m] + I_4 * W[m][k][1][1];
			ofm[m] = ofm[m] + I_5 * W[m][k][1][2];
			ofm[m] = ofm[m] + I_6 * W[m][k][2][0];
			ofm[m] = ofm[m] + I_7 * W[m][k][2][1];
			ofm[m] = ofm[m] + I_8 * W[m][k][2][2];
		}
							if (k == 3 - 1) {
								for (m = 0; m<64; m++) {
#pragma HLS UNROLL
									O_strm.write(ofm[m]);
								}
							}
		}
	}
	}
	}
}

void HW_activation_1(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O1_strm, hls::stream<DATA_T> &O2_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm;
HW_activation_1_x_loop: for (x = 0; x < 32; x++) {
HW_activation_1_y_loop: for (y = 0; y < 32; y++) {
HW_activation_1_m_loop: for (m = 0; m < 64; m++) {
	ifm = I_strm.read();
	if (ifm < 0) {
		O1_strm.write(0); O2_strm.write(0);
	}
	else {
		O1_strm.write(ifm); O2_strm.write(ifm);
	}
}
}
}
}

void HW_res0a_branch2a(hls::stream<DATA_T> &I_strm, DATA_T W[64][64][3][3], DATA_T B[64], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;
	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ofm[64];
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[64][3][32];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_res0a_branch2a_x_loop: for (x = 0; x<32 + 1; x++) {
	HW_res0a_branch2a_y_loop: for (y = 0; y<32 + 1; y++) {
	HW_res0a_branch2a_k_loop: for (k = 0; k<64; k++) {
		if (x < 32 && y < 32) {
			I[k][x % 3][y] = I_strm.read();
		}
		if ((x >> 1) && (y >> 1)) {
			if (x % 2 == 0 && y % 2 == 0) {
#pragma HLS PIPELINE
				if (x - 2 + 0 < 32 && y - 2 + 0 < 32) {
					I_0 = I[k][(x - 2) % 3][(y - 2)];
				}
				else {
					I_0 = 0; // zero padding
				}
				if (x - 2 + 0 < 32 && y - 2 + 1 < 32) {
					I_1 = I[k][(x - 2) % 3][(y - 1)];
				}
				else {
					I_1 = 0; // zero padding
				}
				if (x - 2 + 0 < 32 && y - 2 + 2 < 32) {
					I_2 = I[k][(x - 2) % 3][(y)];
				}
				else {
					I_2 = 0; // zero padding
				}

				if (x - 2 + 1 < 32 && y - 2 + 0 < 32) {
					I_3 = I[k][(x - 1) % 3][(y - 2)];
				}
				else {
					I_3 = 0; // zero padding
				}
				if (x - 2 + 1 < 32 && y - 2 + 1 < 32) {
					I_4 = I[k][(x - 1) % 3][(y - 1)];
				}
				else {
					I_4 = 0; // zero padding
				}
				if (x - 2 + 1 < 32 && y - 2 + 2 < 32) {
					I_5 = I[k][(x - 1) % 3][(y)];
				}
				else {
					I_5 = 0; // zero padding
				}
				if (x - 2 + 2 < 32 && y - 2 + 0 < 32) {
					I_6 = I[k][(x) % 3][(y - 2)];
				}
				else {
					I_6 = 0; // zero padding
				}
				if (x - 2 + 2 < 32 && y - 2 + 1 < 32) {
					I_7 = I[k][(x) % 3][(y - 1)];
				}
				else {
					I_7 = 0; // zero padding
				}
				if (x - 2 + 2 < 32 && y - 2 + 2 < 32) {
					I_8 = I[k][(x) % 3][(y)];
				}
				else {
					I_8 = 0; // zero padding
				}

			HW_res0a_branch2a_m_loop: for (m = 0; m<64; m++) {
#pragma HLS PIPELINE
				if (k == 0) {
					ofm[m] = B[m];
				}
				ofm[m] = ofm[m] + I_0 * W[m][k][0][0];
				ofm[m] = ofm[m] + I_1 * W[m][k][0][1];
				ofm[m] = ofm[m] + I_2 * W[m][k][0][2];
				ofm[m] = ofm[m] + I_3 * W[m][k][1][0];
				ofm[m] = ofm[m] + I_4 * W[m][k][1][1];
				ofm[m] = ofm[m] + I_5 * W[m][k][1][2];
				ofm[m] = ofm[m] + I_6 * W[m][k][2][0];
				ofm[m] = ofm[m] + I_7 * W[m][k][2][1];
				ofm[m] = ofm[m] + I_8 * W[m][k][2][2];
			}
									  if (k == 64 - 1) {
										  for (m = 0; m<64; m++) {
#pragma HLS UNROLL
											  O_strm.write(ofm[m]);
										  }
									  }
			}
		}
	}
	}
	}
}

void HW_conv2d_2(hls::stream<DATA_T> &I_strm, DATA_T W[64][64][3][3], DATA_T B[64], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ofm[64];
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[64][3][16];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_2_x_loop: for (x = 0; x<16 + 1; x++) {
	HW_conv2d_2_y_loop: for (y = 0; y<16 + 1; y++) {
	HW_conv2d_2_k_loop: for (k = 0; k<64; k++) {
		if (x < 16 && y < 16) {
			I[k][x % 3][y] = I_strm.read();
		}
		if (x && y) {
#pragma HLS PIPELINE
			if (x - 2 + 0 < 16 && y - 2 + 0 < 16 && x - 2 + 0 >= 0 && y - 2 + 0 >= 0) {
				I_0 = I[k][(x - 2) % 3][(y - 2)];
			}
			else {
				I_0 = 0; // zero padding
			}
			if (x - 2 + 0 < 16 && y - 2 + 1 < 16 && x - 2 + 0 >= 0 && y - 2 + 1 >= 0) {
				I_1 = I[k][(x - 2) % 3][(y - 1)];
			}
			else {
				I_1 = 0; // zero padding
			}
			if (x - 2 + 0 < 16 && y - 2 + 2 < 16 && x - 2 + 0 >= 0 && y - 2 + 2 >= 0) {
				I_2 = I[k][(x - 2) % 3][(y)];
			}
			else {
				I_2 = 0; // zero padding
			}

			if (x - 2 + 1 < 16 && y - 2 + 0 < 16 && x - 2 + 1 >= 0 && y - 2 + 0 >= 0) {
				I_3 = I[k][(x - 1) % 3][(y - 2)];
			}
			else {
				I_3 = 0; // zero padding
			}
			if (x - 2 + 1 < 16 && y - 2 + 1 < 16 && x - 2 + 1 >= 0 && y - 2 + 1 >= 0) {
				I_4 = I[k][(x - 1) % 3][(y - 1)];
			}
			else {
				I_4 = 0; // zero padding
			}
			if (x - 2 + 1 < 16 && y - 2 + 2 < 16 && x - 2 + 1 >= 0 && y - 2 + 2 >= 0) {
				I_5 = I[k][(x - 1) % 3][(y)];
			}
			else {
				I_5 = 0; // zero padding
			}
			if (x - 2 + 2 < 16 && y - 2 + 0 < 16 && x - 2 + 2 >= 0 && y - 2 + 0 >= 0) {
				I_6 = I[k][(x) % 3][(y - 2)];
			}
			else {
				I_6 = 0; // zero padding
			}
			if (x - 2 + 2 < 16 && y - 2 + 1 < 16 && x - 2 + 2 >= 0 && y - 2 + 1 >= 0) {
				I_7 = I[k][(x) % 3][(y - 1)];
			}
			else {
				I_7 = 0; // zero padding
			}
			if (x - 2 + 2 < 16 && y - 2 + 2 < 16 && x - 2 + 2 >= 0 && y - 2 + 2 >= 0) {
				I_8 = I[k][(x) % 3][(y)];
			}
			else {
				I_8 = 0; // zero padding
			}

		HW_conv2d_2_m_loop: for (m = 0; m<64; m++) {
#pragma HLS PIPELINE
			if (k == 0) {
				ofm[m] = B[m];
			}
			ofm[m] = ofm[m] + I_0 * W[m][k][0][0];
			ofm[m] = ofm[m] + I_1 * W[m][k][0][1];
			ofm[m] = ofm[m] + I_2 * W[m][k][0][2];
			ofm[m] = ofm[m] + I_3 * W[m][k][1][0];
			ofm[m] = ofm[m] + I_4 * W[m][k][1][1];
			ofm[m] = ofm[m] + I_5 * W[m][k][1][2];
			ofm[m] = ofm[m] + I_6 * W[m][k][2][0];
			ofm[m] = ofm[m] + I_7 * W[m][k][2][1];
			ofm[m] = ofm[m] + I_8 * W[m][k][2][2];
		}
							if (k == 64 - 1) {
								for (m = 0; m<64; m++) {
#pragma HLS UNROLL
									O_strm.write(ofm[m]);
								}
							}
		}
	}
	}
	}
}

void HW_conv2d_3(hls::stream<DATA_T> &I_strm, DATA_T W[64][64], DATA_T B[64], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;
	DATA_T ofm[64];
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0
	DATA_T ifm;

HW_conv2d_3_x_loop: for (x = 0; x < 32; x++) {
HW_conv2d_3_y_loop: for (y = 0; y < 32; y++) {
HW_conv2d_3_k_loop: for (k = 0; k < 64; k++) {
	ifm = I_strm.read();
	if (x % 2 == 0 && y % 2 == 0) {
	HW_conv2d_3_m_loop: for (m = 0; m < 64; m++) {
#pragma HLS PIPELINE
		if (k == 0) {
			ofm[m] = B[m];
		}
		ofm[m] = ofm[m] + ifm * W[m][k];
	}
						if (k == 64 - 1) {
							for (m = 0; m < 64; m++) {
#pragma HLS UNROLL
								O_strm.write(ofm[m]);
							}
						}
	}
}
}
}
}

void HW_activation_2(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm;
HW_activation_2_x_loop: for (x = 0; x < 16; x++) {
HW_activation_2_y_loop: for (y = 0; y < 16; y++) {
HW_activation_2_m_loop: for (m = 0; m < 64; m++) {
	ifm = I_strm.read();
	if (ifm < 0) {
		O_strm.write(0);
	}
	else {
		O_strm.write(ifm);
	}
}
}
}
}

void HW_add_1(hls::stream<DATA_T> &I1_strm, hls::stream<DATA_T> &I2_strm, hls::stream<DATA_T> &O1_strm, hls::stream<DATA_T> &O2_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm;
HW_add_1_x_loop_1: for (x = 0; x < 16; x++) {
HW_add_1_y_loop_1: for (y = 0; y < 16; y++) {
HW_add_1_m_loop_1: for (m = 0; m < 64; m++) {
	ifm = I1_strm.read();
	ifm = ifm + I2_strm.read();
	O1_strm.write(ifm);
	O2_strm.write(ifm);
}
}
}
}

void HW_conv2d_4(hls::stream<DATA_T> &I_strm, DATA_T W[64][64][3][3], DATA_T B[64], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ofm[64];
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[64][3][16];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_4_x_loop: for (x = 0; x<16 + 1; x++) {
	HW_conv2d_4_y_loop: for (y = 0; y<16 + 1; y++) {
	HW_conv2d_4_k_loop: for (k = 0; k<64; k++) {
		if (x < 16 && y < 16) {
			I[k][x % 3][y] = I_strm.read();
		}
		if (x && y) {
#pragma HLS PIPELINE
			if (x - 2 + 0 < 16 && y - 2 + 0 < 16 && x - 2 + 0 >= 0 && y - 2 + 0 >= 0) {
				I_0 = I[k][(x - 2) % 3][(y - 2)];
			}
			else {
				I_0 = 0; // zero padding
			}
			if (x - 2 + 0 < 16 && y - 2 + 1 < 16 && x - 2 + 0 >= 0 && y - 2 + 1 >= 0) {
				I_1 = I[k][(x - 2) % 3][(y - 1)];
			}
			else {
				I_1 = 0; // zero padding
			}
			if (x - 2 + 0 < 16 && y - 2 + 2 < 16 && x - 2 + 0 >= 0 && y - 2 + 2 >= 0) {
				I_2 = I[k][(x - 2) % 3][(y)];
			}
			else {
				I_2 = 0; // zero padding
			}

			if (x - 2 + 1 < 16 && y - 2 + 0 < 16 && x - 2 + 1 >= 0 && y - 2 + 0 >= 0) {
				I_3 = I[k][(x - 1) % 3][(y - 2)];
			}
			else {
				I_3 = 0; // zero padding
			}
			if (x - 2 + 1 < 16 && y - 2 + 1 < 16 && x - 2 + 1 >= 0 && y - 2 + 1 >= 0) {
				I_4 = I[k][(x - 1) % 3][(y - 1)];
			}
			else {
				I_4 = 0; // zero padding
			}
			if (x - 2 + 1 < 16 && y - 2 + 2 < 16 && x - 2 + 1 >= 0 && y - 2 + 2 >= 0) {
				I_5 = I[k][(x - 1) % 3][(y)];
			}
			else {
				I_5 = 0; // zero padding
			}
			if (x - 2 + 2 < 16 && y - 2 + 0 < 16 && x - 2 + 2 >= 0 && y - 2 + 0 >= 0) {
				I_6 = I[k][(x) % 3][(y - 2)];
			}
			else {
				I_6 = 0; // zero padding
			}
			if (x - 2 + 2 < 16 && y - 2 + 1 < 16 && x - 2 + 2 >= 0 && y - 2 + 1 >= 0) {
				I_7 = I[k][(x) % 3][(y - 1)];
			}
			else {
				I_7 = 0; // zero padding
			}
			if (x - 2 + 2 < 16 && y - 2 + 2 < 16 && x - 2 + 2 >= 0 && y - 2 + 2 >= 0) {
				I_8 = I[k][(x) % 3][(y)];
			}
			else {
				I_8 = 0; // zero padding
			}

		HW_conv2d_4_m_loop: for (m = 0; m<64; m++) {
#pragma HLS PIPELINE
			if (k == 0) {
				ofm[m] = B[m];
			}
			ofm[m] = ofm[m] + I_0 * W[m][k][0][0];
			ofm[m] = ofm[m] + I_1 * W[m][k][0][1];
			ofm[m] = ofm[m] + I_2 * W[m][k][0][2];
			ofm[m] = ofm[m] + I_3 * W[m][k][1][0];
			ofm[m] = ofm[m] + I_4 * W[m][k][1][1];
			ofm[m] = ofm[m] + I_5 * W[m][k][1][2];
			ofm[m] = ofm[m] + I_6 * W[m][k][2][0];
			ofm[m] = ofm[m] + I_7 * W[m][k][2][1];
			ofm[m] = ofm[m] + I_8 * W[m][k][2][2];
		}
							if (k == 64 - 1) {
								for (m = 0; m<64; m++) {
#pragma HLS UNROLL
									O_strm.write(ofm[m]);
								}
							}
		}
	}
	}
	}
}

void HW_activation_3(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm;
HW_activation_3_x_loop: for (x = 0; x < 16; x++) {
HW_activation_3_y_loop: for (y = 0; y < 16; y++) {
HW_activation_3_m_loop: for (m = 0; m < 64; m++) {
	ifm = I_strm.read();
	if (ifm < 0) {
		O_strm.write(0);
	}
	else {
		O_strm.write(ifm);
	}
}
}
}
}

void HW_conv2d_5(hls::stream<DATA_T> &I_strm, DATA_T W[64][64][3][3], DATA_T B[64], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ofm[64];
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[64][3][16];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_5_x_loop: for (x = 0; x<16 + 1; x++) {
	HW_conv2d_5_y_loop: for (y = 0; y<16 + 1; y++) {
	HW_conv2d_5_k_loop: for (k = 0; k<64; k++) {
		if (x < 16 && y < 16) {
			I[k][x % 3][y] = I_strm.read();
		}
		if (x && y) {
#pragma HLS PIPELINE
			if (x - 2 + 0 < 16 && y - 2 + 0 < 16 && x - 2 + 0 >= 0 && y - 2 + 0 >= 0) {
				I_0 = I[k][(x - 2) % 3][(y - 2)];
			}
			else {
				I_0 = 0; // zero padding
			}
			if (x - 2 + 0 < 16 && y - 2 + 1 < 16 && x - 2 + 0 >= 0 && y - 2 + 1 >= 0) {
				I_1 = I[k][(x - 2) % 3][(y - 1)];
			}
			else {
				I_1 = 0; // zero padding
			}
			if (x - 2 + 0 < 16 && y - 2 + 2 < 16 && x - 2 + 0 >= 0 && y - 2 + 2 >= 0) {
				I_2 = I[k][(x - 2) % 3][(y)];
			}
			else {
				I_2 = 0; // zero padding
			}

			if (x - 2 + 1 < 16 && y - 2 + 0 < 16 && x - 2 + 1 >= 0 && y - 2 + 0 >= 0) {
				I_3 = I[k][(x - 1) % 3][(y - 2)];
			}
			else {
				I_3 = 0; // zero padding
			}
			if (x - 2 + 1 < 16 && y - 2 + 1 < 16 && x - 2 + 1 >= 0 && y - 2 + 1 >= 0) {
				I_4 = I[k][(x - 1) % 3][(y - 1)];
			}
			else {
				I_4 = 0; // zero padding
			}
			if (x - 2 + 1 < 16 && y - 2 + 2 < 16 && x - 2 + 1 >= 0 && y - 2 + 2 >= 0) {
				I_5 = I[k][(x - 1) % 3][(y)];
			}
			else {
				I_5 = 0; // zero padding
			}
			if (x - 2 + 2 < 16 && y - 2 + 0 < 16 && x - 2 + 2 >= 0 && y - 2 + 0 >= 0) {
				I_6 = I[k][(x) % 3][(y - 2)];
			}
			else {
				I_6 = 0; // zero padding
			}
			if (x - 2 + 2 < 16 && y - 2 + 1 < 16 && x - 2 + 2 >= 0 && y - 2 + 1 >= 0) {
				I_7 = I[k][(x) % 3][(y - 1)];
			}
			else {
				I_7 = 0; // zero padding
			}
			if (x - 2 + 2 < 16 && y - 2 + 2 < 16 && x - 2 + 2 >= 0 && y - 2 + 2 >= 0) {
				I_8 = I[k][(x) % 3][(y)];
			}
			else {
				I_8 = 0; // zero padding
			}

		HW_conv2d_5_m_loop: for (m = 0; m<64; m++) {
#pragma HLS PIPELINE
			if (k == 0) {
				ofm[m] = B[m];
			}
			ofm[m] = ofm[m] + I_0 * W[m][k][0][0];
			ofm[m] = ofm[m] + I_1 * W[m][k][0][1];
			ofm[m] = ofm[m] + I_2 * W[m][k][0][2];
			ofm[m] = ofm[m] + I_3 * W[m][k][1][0];
			ofm[m] = ofm[m] + I_4 * W[m][k][1][1];
			ofm[m] = ofm[m] + I_5 * W[m][k][1][2];
			ofm[m] = ofm[m] + I_6 * W[m][k][2][0];
			ofm[m] = ofm[m] + I_7 * W[m][k][2][1];
			ofm[m] = ofm[m] + I_8 * W[m][k][2][2];
		}
							if (k == 64 - 1) {
								for (m = 0; m<64; m++) {
#pragma HLS UNROLL
									O_strm.write(ofm[m]);
								}
							}
		}
	}
	}
	}
}

void HW_activation_4(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm;
HW_activation_4_x_loop: for (x = 0; x < 16; x++) {
HW_activation_4_y_loop: for (y = 0; y < 16; y++) {
HW_activation_4_m_loop: for (m = 0; m < 64; m++) {
	ifm = I_strm.read();
	if (ifm < 0) {
		O_strm.write(0);
	}
	else {
		O_strm.write(ifm);
	}
}
}
}
}

void HW_add_2(hls::stream<DATA_T> &I1_strm, hls::stream<DATA_T> &I2_strm, hls::stream<DATA_T> &O1_strm, hls::stream<DATA_T> &O2_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm;
HW_add_2_x_loop_1: for (x = 0; x < 16; x++) {
HW_add_2_y_loop_1: for (y = 0; y < 16; y++) {
HW_add_2_m_loop_1: for (m = 0; m < 64; m++) {
	ifm = I1_strm.read();
	ifm = ifm + I2_strm.read();
	O1_strm.write(ifm);
	O2_strm.write(ifm);
}
}
}
}

void HW_conv2d_6(hls::stream<DATA_T> &I_strm, DATA_T W[128][64][3][3], DATA_T B[128], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;
	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ofm[128];
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[64][3][16];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_6_x_loop: for (x = 0; x<16 + 1; x++) {
	HW_conv2d_6_y_loop: for (y = 0; y<16 + 1; y++) {
	HW_conv2d_6_k_loop: for (k = 0; k<64; k++) {
		if (x < 16 && y < 16) {
			I[k][x % 3][y] = I_strm.read();
		}
		if ((x >> 1) && (y >> 1)) {
			if (x % 2 == 0 && y % 2 == 0) {
#pragma HLS PIPELINE
				if (x - 2 + 0 < 16 && y - 2 + 0 < 16) {
					I_0 = I[k][(x - 2) % 3][(y - 2)];
				}
				else {
					I_0 = 0; // zero padding
				}
				if (x - 2 + 0 < 16 && y - 2 + 1 < 16) {
					I_1 = I[k][(x - 2) % 3][(y - 1)];
				}
				else {
					I_1 = 0; // zero padding
				}
				if (x - 2 + 0 < 16 && y - 2 + 2 < 16) {
					I_2 = I[k][(x - 2) % 3][(y)];
				}
				else {
					I_2 = 0; // zero padding
				}

				if (x - 2 + 1 < 16 && y - 2 + 0 < 16) {
					I_3 = I[k][(x - 1) % 3][(y - 2)];
				}
				else {
					I_3 = 0; // zero padding
				}
				if (x - 2 + 1 < 16 && y - 2 + 1 < 16) {
					I_4 = I[k][(x - 1) % 3][(y - 1)];
				}
				else {
					I_4 = 0; // zero padding
				}
				if (x - 2 + 1 < 16 && y - 2 + 2 < 16) {
					I_5 = I[k][(x - 1) % 3][(y)];
				}
				else {
					I_5 = 0; // zero padding
				}
				if (x - 2 + 2 < 16 && y - 2 + 0 < 16) {
					I_6 = I[k][(x) % 3][(y - 2)];
				}
				else {
					I_6 = 0; // zero padding
				}
				if (x - 2 + 2 < 16 && y - 2 + 1 < 16) {
					I_7 = I[k][(x) % 3][(y - 1)];
				}
				else {
					I_7 = 0; // zero padding
				}
				if (x - 2 + 2 < 16 && y - 2 + 2 < 16) {
					I_8 = I[k][(x) % 3][(y)];
				}
				else {
					I_8 = 0; // zero padding
				}

			HW_conv2d_6_m_loop: for (m = 0; m<128; m++) {
#pragma HLS PIPELINE
				if (k == 0) {
					ofm[m] = B[m];
				}
				ofm[m] = ofm[m] + I_0 * W[m][k][0][0];
				ofm[m] = ofm[m] + I_1 * W[m][k][0][1];
				ofm[m] = ofm[m] + I_2 * W[m][k][0][2];
				ofm[m] = ofm[m] + I_3 * W[m][k][1][0];
				ofm[m] = ofm[m] + I_4 * W[m][k][1][1];
				ofm[m] = ofm[m] + I_5 * W[m][k][1][2];
				ofm[m] = ofm[m] + I_6 * W[m][k][2][0];
				ofm[m] = ofm[m] + I_7 * W[m][k][2][1];
				ofm[m] = ofm[m] + I_8 * W[m][k][2][2];
			}
								if (k == 64 - 1) {
									for (m = 0; m<128; m++) {
#pragma HLS UNROLL
										O_strm.write(ofm[m]);
									}
								}
			}
		}
	}
	}
	}
}

void HW_activation_5(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm;
HW_activation_5_x_loop: for (x = 0; x < 8; x++) {
HW_activation_5_y_loop: for (y = 0; y < 8; y++) {
HW_activation_5_m_loop: for (m = 0; m < 128; m++) {
	ifm = I_strm.read();
	if (ifm < 0) {
		O_strm.write(0);
	}
	else {
		O_strm.write(ifm);
	}
}
}
}
}

void HW_conv2d_7(hls::stream<DATA_T> &I_strm, DATA_T W[128][128][3][3], DATA_T B[128], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ofm[128];
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[128][3][8];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_7_x_loop: for (x = 0; x<8 + 1; x++) {
	HW_conv2d_7_y_loop: for (y = 0; y<8 + 1; y++) {
	HW_conv2d_7_k_loop: for (k = 0; k<128; k++) {
		if (x < 8 && y < 8) {
			I[k][x % 3][y] = I_strm.read();
		}
		if (x && y) {
#pragma HLS PIPELINE
			if (x - 2 + 0 < 8 && y - 2 + 0 < 8 && x - 2 + 0 >= 0 && y - 2 + 0 >= 0) {
				I_0 = I[k][(x - 2) % 3][(y - 2)];
			}
			else {
				I_0 = 0; // zero padding
			}
			if (x - 2 + 0 < 8 && y - 2 + 1 < 8 && x - 2 + 0 >= 0 && y - 2 + 1 >= 0) {
				I_1 = I[k][(x - 2) % 3][(y - 1)];
			}
			else {
				I_1 = 0; // zero padding
			}
			if (x - 2 + 0 < 8 && y - 2 + 2 < 8 && x - 2 + 0 >= 0 && y - 2 + 2 >= 0) {
				I_2 = I[k][(x - 2) % 3][(y)];
			}
			else {
				I_2 = 0; // zero padding
			}

			if (x - 2 + 1 < 8 && y - 2 + 0 < 8 && x - 2 + 1 >= 0 && y - 2 + 0 >= 0) {
				I_3 = I[k][(x - 1) % 3][(y - 2)];
			}
			else {
				I_3 = 0; // zero padding
			}
			if (x - 2 + 1 < 8 && y - 2 + 1 < 8 && x - 2 + 1 >= 0 && y - 2 + 1 >= 0) {
				I_4 = I[k][(x - 1) % 3][(y - 1)];
			}
			else {
				I_4 = 0; // zero padding
			}
			if (x - 2 + 1 < 8 && y - 2 + 2 < 8 && x - 2 + 1 >= 0 && y - 2 + 2 >= 0) {
				I_5 = I[k][(x - 1) % 3][(y)];
			}
			else {
				I_5 = 0; // zero padding
			}
			if (x - 2 + 2 < 8 && y - 2 + 0 < 8 && x - 2 + 2 >= 0 && y - 2 + 0 >= 0) {
				I_6 = I[k][(x) % 3][(y - 2)];
			}
			else {
				I_6 = 0; // zero padding
			}
			if (x - 2 + 2 < 8 && y - 2 + 1 < 8 && x - 2 + 2 >= 0 && y - 2 + 1 >= 0) {
				I_7 = I[k][(x) % 3][(y - 1)];
			}
			else {
				I_7 = 0; // zero padding
			}
			if (x - 2 + 2 < 8 && y - 2 + 2 < 8 && x - 2 + 2 >= 0 && y - 2 + 2 >= 0) {
				I_8 = I[k][(x) % 3][(y)];
			}
			else {
				I_8 = 0; // zero padding
			}

		HW_conv2d_7_m_loop: for (m = 0; m<128; m++) {
#pragma HLS PIPELINE
			if (k == 0) {
				ofm[m] = B[m];
			}
			ofm[m] = ofm[m] + I_0 * W[m][k][0][0];
			ofm[m] = ofm[m] + I_1 * W[m][k][0][1];
			ofm[m] = ofm[m] + I_2 * W[m][k][0][2];
			ofm[m] = ofm[m] + I_3 * W[m][k][1][0];
			ofm[m] = ofm[m] + I_4 * W[m][k][1][1];
			ofm[m] = ofm[m] + I_5 * W[m][k][1][2];
			ofm[m] = ofm[m] + I_6 * W[m][k][2][0];
			ofm[m] = ofm[m] + I_7 * W[m][k][2][1];
			ofm[m] = ofm[m] + I_8 * W[m][k][2][2];
		}
							if (k == 128 - 1) {
								for (m = 0; m<128; m++) {
#pragma HLS UNROLL
									O_strm.write(ofm[m]);
								}
							}
		}
	}
	}
	}
}

void HW_conv2d_8(hls::stream<DATA_T> &I_strm, DATA_T W[128][64], DATA_T B[128], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;
	DATA_T ofm[128];
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0
	DATA_T ifm;

HW_conv2d_8_x_loop: for (x = 0; x < 16; x++) {
HW_conv2d_8_y_loop: for (y = 0; y < 16; y++) {
HW_conv2d_8_k_loop: for (k = 0; k < 64; k++) {
	ifm = I_strm.read();
	if (x % 2 == 0 && y % 2 == 0) {
	HW_conv2d_8_m_loop: for (m = 0; m < 128; m++) {
#pragma HLS PIPELINE
		if (k == 0) {
			ofm[m] = B[m];
		}
		ofm[m] = ofm[m] + ifm * W[m][k];
	}
						if (k == 64 - 1) {
							for (m = 0; m < 128; m++) {
#pragma HLS UNROLL
								O_strm.write(ofm[m]);
							}
						}
	}
}
}
}
}

void HW_activation_6(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm;
HW_activation_6_x_loop: for (x = 0; x < 8; x++) {
HW_activation_6_y_loop: for (y = 0; y < 8; y++) {
HW_activation_6_m_loop: for (m = 0; m < 128; m++) {
	ifm = I_strm.read();
	if (ifm < 0) {
		O_strm.write(0);
	}
	else {
		O_strm.write(ifm);
	}
}
}
}
}

void HW_add_3(hls::stream<DATA_T> &I1_strm, hls::stream<DATA_T> &I2_strm, hls::stream<DATA_T> &O1_strm, hls::stream<DATA_T> &O2_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm;
HW_add_3_x_loop_1: for (x = 0; x < 8; x++) {
HW_add_3_y_loop_1: for (y = 0; y < 8; y++) {
HW_add_3_m_loop_1: for (m = 0; m < 128; m++) {
	ifm = I1_strm.read();
	ifm = ifm + I2_strm.read();
	O1_strm.write(ifm);
	O2_strm.write(ifm);
}
}
}
}

void HW_conv2d_9(hls::stream<DATA_T> &I_strm, DATA_T W[128][128][3][3], DATA_T B[128], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ofm[128];
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[128][3][8];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_9_x_loop: for (x = 0; x<8 + 1; x++) {
	HW_conv2d_9_y_loop: for (y = 0; y<8 + 1; y++) {
	HW_conv2d_9_k_loop: for (k = 0; k<128; k++) {
		if (x < 8 && y < 8) {
			I[k][x % 3][y] = I_strm.read();
		}
		if (x && y) {
#pragma HLS PIPELINE
			if (x - 2 + 0 < 8 && y - 2 + 0 < 8 && x - 2 + 0 >= 0 && y - 2 + 0 >= 0) {
				I_0 = I[k][(x - 2) % 3][(y - 2)];
			}
			else {
				I_0 = 0; // zero padding
			}
			if (x - 2 + 0 < 8 && y - 2 + 1 < 8 && x - 2 + 0 >= 0 && y - 2 + 1 >= 0) {
				I_1 = I[k][(x - 2) % 3][(y - 1)];
			}
			else {
				I_1 = 0; // zero padding
			}
			if (x - 2 + 0 < 8 && y - 2 + 2 < 8 && x - 2 + 0 >= 0 && y - 2 + 2 >= 0) {
				I_2 = I[k][(x - 2) % 3][(y)];
			}
			else {
				I_2 = 0; // zero padding
			}

			if (x - 2 + 1 < 8 && y - 2 + 0 < 8 && x - 2 + 1 >= 0 && y - 2 + 0 >= 0) {
				I_3 = I[k][(x - 1) % 3][(y - 2)];
			}
			else {
				I_3 = 0; // zero padding
			}
			if (x - 2 + 1 < 8 && y - 2 + 1 < 8 && x - 2 + 1 >= 0 && y - 2 + 1 >= 0) {
				I_4 = I[k][(x - 1) % 3][(y - 1)];
			}
			else {
				I_4 = 0; // zero padding
			}
			if (x - 2 + 1 < 8 && y - 2 + 2 < 8 && x - 2 + 1 >= 0 && y - 2 + 2 >= 0) {
				I_5 = I[k][(x - 1) % 3][(y)];
			}
			else {
				I_5 = 0; // zero padding
			}
			if (x - 2 + 2 < 8 && y - 2 + 0 < 8 && x - 2 + 2 >= 0 && y - 2 + 0 >= 0) {
				I_6 = I[k][(x) % 3][(y - 2)];
			}
			else {
				I_6 = 0; // zero padding
			}
			if (x - 2 + 2 < 8 && y - 2 + 1 < 8 && x - 2 + 2 >= 0 && y - 2 + 1 >= 0) {
				I_7 = I[k][(x) % 3][(y - 1)];
			}
			else {
				I_7 = 0; // zero padding
			}
			if (x - 2 + 2 < 8 && y - 2 + 2 < 8 && x - 2 + 2 >= 0 && y - 2 + 2 >= 0) {
				I_8 = I[k][(x) % 3][(y)];
			}
			else {
				I_8 = 0; // zero padding
			}

		HW_conv2d_9_m_loop: for (m = 0; m<128; m++) {
#pragma HLS PIPELINE
			if (k == 0) {
				ofm[m] = B[m];
			}
			ofm[m] = ofm[m] + I_0 * W[m][k][0][0];
			ofm[m] = ofm[m] + I_1 * W[m][k][0][1];
			ofm[m] = ofm[m] + I_2 * W[m][k][0][2];
			ofm[m] = ofm[m] + I_3 * W[m][k][1][0];
			ofm[m] = ofm[m] + I_4 * W[m][k][1][1];
			ofm[m] = ofm[m] + I_5 * W[m][k][1][2];
			ofm[m] = ofm[m] + I_6 * W[m][k][2][0];
			ofm[m] = ofm[m] + I_7 * W[m][k][2][1];
			ofm[m] = ofm[m] + I_8 * W[m][k][2][2];
		}
							if (k == 128 - 1) {
								for (m = 0; m<128; m++) {
#pragma HLS UNROLL
									O_strm.write(ofm[m]);
								}
							}
		}
	}
	}
	}
}

void HW_activation_7(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm;
HW_activation_7_x_loop: for (x = 0; x < 8; x++) {
HW_activation_7_y_loop: for (y = 0; y < 8; y++) {
HW_activation_7_m_loop: for (m = 0; m < 128; m++) {
	ifm = I_strm.read();
	if (ifm < 0) {
		O_strm.write(0);
	}
	else {
		O_strm.write(ifm);
	}
}
}
}
}

void HW_conv2d_10(hls::stream<DATA_T> &I_strm, DATA_T W[128][128][3][3], DATA_T B[128], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ofm[128];
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[128][3][8];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_10_x_loop: for (x = 0; x<8 + 1; x++) {
	HW_conv2d_10_y_loop: for (y = 0; y<8 + 1; y++) {
	HW_conv2d_10_k_loop: for (k = 0; k<128; k++) {
		if (x < 8 && y < 8) {
			I[k][x % 3][y] = I_strm.read();
		}
		if (x && y) {
#pragma HLS PIPELINE
			if (x - 2 + 0 < 8 && y - 2 + 0 < 8 && x - 2 + 0 >= 0 && y - 2 + 0 >= 0) {
				I_0 = I[k][(x - 2) % 3][(y - 2)];
			}
			else {
				I_0 = 0; // zero padding
			}
			if (x - 2 + 0 < 8 && y - 2 + 1 < 8 && x - 2 + 0 >= 0 && y - 2 + 1 >= 0) {
				I_1 = I[k][(x - 2) % 3][(y - 1)];
			}
			else {
				I_1 = 0; // zero padding
			}
			if (x - 2 + 0 < 8 && y - 2 + 2 < 8 && x - 2 + 0 >= 0 && y - 2 + 2 >= 0) {
				I_2 = I[k][(x - 2) % 3][(y)];
			}
			else {
				I_2 = 0; // zero padding
			}

			if (x - 2 + 1 < 8 && y - 2 + 0 < 8 && x - 2 + 1 >= 0 && y - 2 + 0 >= 0) {
				I_3 = I[k][(x - 1) % 3][(y - 2)];
			}
			else {
				I_3 = 0; // zero padding
			}
			if (x - 2 + 1 < 8 && y - 2 + 1 < 8 && x - 2 + 1 >= 0 && y - 2 + 1 >= 0) {
				I_4 = I[k][(x - 1) % 3][(y - 1)];
			}
			else {
				I_4 = 0; // zero padding
			}
			if (x - 2 + 1 < 8 && y - 2 + 2 < 8 && x - 2 + 1 >= 0 && y - 2 + 2 >= 0) {
				I_5 = I[k][(x - 1) % 3][(y)];
			}
			else {
				I_5 = 0; // zero padding
			}
			if (x - 2 + 2 < 8 && y - 2 + 0 < 8 && x - 2 + 2 >= 0 && y - 2 + 0 >= 0) {
				I_6 = I[k][(x) % 3][(y - 2)];
			}
			else {
				I_6 = 0; // zero padding
			}
			if (x - 2 + 2 < 8 && y - 2 + 1 < 8 && x - 2 + 2 >= 0 && y - 2 + 1 >= 0) {
				I_7 = I[k][(x) % 3][(y - 1)];
			}
			else {
				I_7 = 0; // zero padding
			}
			if (x - 2 + 2 < 8 && y - 2 + 2 < 8 && x - 2 + 2 >= 0 && y - 2 + 2 >= 0) {
				I_8 = I[k][(x) % 3][(y)];
			}
			else {
				I_8 = 0; // zero padding
			}

		HW_conv2d_10_m_loop: for (m = 0; m<128; m++) {
#pragma HLS PIPELINE
			if (k == 0) {
				ofm[m] = B[m];
			}
			ofm[m] = ofm[m] + I_0 * W[m][k][0][0];
			ofm[m] = ofm[m] + I_1 * W[m][k][0][1];
			ofm[m] = ofm[m] + I_2 * W[m][k][0][2];
			ofm[m] = ofm[m] + I_3 * W[m][k][1][0];
			ofm[m] = ofm[m] + I_4 * W[m][k][1][1];
			ofm[m] = ofm[m] + I_5 * W[m][k][1][2];
			ofm[m] = ofm[m] + I_6 * W[m][k][2][0];
			ofm[m] = ofm[m] + I_7 * W[m][k][2][1];
			ofm[m] = ofm[m] + I_8 * W[m][k][2][2];
		}
							 if (k == 128 - 1) {
								 for (m = 0; m<128; m++) {
#pragma HLS UNROLL
									 O_strm.write(ofm[m]);
								 }
							 }
		}
	}
	}
	}
}

void HW_activation_8(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm;
HW_activation_8_x_loop: for (x = 0; x < 8; x++) {
HW_activation_8_y_loop: for (y = 0; y < 8; y++) {
HW_activation_8_m_loop: for (m = 0; m < 128; m++) {
	ifm = I_strm.read();
	if (ifm < 0) {
		O_strm.write(0);
	}
	else {
		O_strm.write(ifm);
	}
}
}
}
}

void HW_add_4(hls::stream<DATA_T> &I1_strm, hls::stream<DATA_T> &I2_strm, hls::stream<DATA_T> &O1_strm, hls::stream<DATA_T> &O2_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm;
HW_add_4_x_loop_1: for (x = 0; x < 8; x++) {
HW_add_4_y_loop_1: for (y = 0; y < 8; y++) {
HW_add_4_m_loop_1: for (m = 0; m < 128; m++) {
	ifm = I1_strm.read();
	ifm = ifm + I2_strm.read();
	O1_strm.write(ifm);
	O2_strm.write(ifm);
}
}
}
}

void HW_conv2d_11(hls::stream<DATA_T> &I_strm, DATA_T W[256][128][3][3], DATA_T B[256], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;
	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ofm[256];
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[128][3][8];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_11_x_loop: for (x = 0; x<8 + 1; x++) {
	HW_conv2d_11_y_loop: for (y = 0; y<8 + 1; y++) {
	HW_conv2d_11_k_loop: for (k = 0; k<128; k++) {
		if (x < 8 && y < 8) {
			I[k][x % 3][y] = I_strm.read();
		}
		if ((x >> 1) && (y >> 1)) {
			if (x % 2 == 0 && y % 2 == 0) {
#pragma HLS PIPELINE
				if (x - 2 + 0 < 8 && y - 2 + 0 < 8) {
					I_0 = I[k][(x - 2) % 3][(y - 2)];
				}
				else {
					I_0 = 0; // zero padding
				}
				if (x - 2 + 0 < 8 && y - 2 + 1 < 8) {
					I_1 = I[k][(x - 2) % 3][(y - 1)];
				}
				else {
					I_1 = 0; // zero padding
				}
				if (x - 2 + 0 < 8 && y - 2 + 2 < 8) {
					I_2 = I[k][(x - 2) % 3][(y)];
				}
				else {
					I_2 = 0; // zero padding
				}

				if (x - 2 + 1 < 8 && y - 2 + 0 < 8) {
					I_3 = I[k][(x - 1) % 3][(y - 2)];
				}
				else {
					I_3 = 0; // zero padding
				}
				if (x - 2 + 1 < 8 && y - 2 + 1 < 8) {
					I_4 = I[k][(x - 1) % 3][(y - 1)];
				}
				else {
					I_4 = 0; // zero padding
				}
				if (x - 2 + 1 < 8 && y - 2 + 2 < 8) {
					I_5 = I[k][(x - 1) % 3][(y)];
				}
				else {
					I_5 = 0; // zero padding
				}
				if (x - 2 + 2 < 8 && y - 2 + 0 < 8) {
					I_6 = I[k][(x) % 3][(y - 2)];
				}
				else {
					I_6 = 0; // zero padding
				}
				if (x - 2 + 2 < 8 && y - 2 + 1 < 8) {
					I_7 = I[k][(x) % 3][(y - 1)];
				}
				else {
					I_7 = 0; // zero padding
				}
				if (x - 2 + 2 < 8 && y - 2 + 2 < 8) {
					I_8 = I[k][(x) % 3][(y)];
				}
				else {
					I_8 = 0; // zero padding
				}

			HW_conv2d_11_m_loop: for (m = 0; m<256; m++) {
#pragma HLS PIPELINE
				if (k == 0) {
					ofm[m] = B[m];
				}
				ofm[m] = ofm[m] + I_0 * W[m][k][0][0];
				ofm[m] = ofm[m] + I_1 * W[m][k][0][1];
				ofm[m] = ofm[m] + I_2 * W[m][k][0][2];
				ofm[m] = ofm[m] + I_3 * W[m][k][1][0];
				ofm[m] = ofm[m] + I_4 * W[m][k][1][1];
				ofm[m] = ofm[m] + I_5 * W[m][k][1][2];
				ofm[m] = ofm[m] + I_6 * W[m][k][2][0];
				ofm[m] = ofm[m] + I_7 * W[m][k][2][1];
				ofm[m] = ofm[m] + I_8 * W[m][k][2][2];
			}
								 if (k == 128 - 1) {
									 for (m = 0; m<256; m++) {
#pragma HLS UNROLL
										 O_strm.write(ofm[m]);
									 }
								 }
			}
		}
	}
	}
	}
}

void HW_activation_9(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm;
HW_activation_9_x_loop: for (x = 0; x < 4; x++) {
HW_activation_9_y_loop: for (y = 0; y < 4; y++) {
HW_activation_9_m_loop: for (m = 0; m < 256; m++) {
	ifm = I_strm.read();
	if (ifm < 0) {
		O_strm.write(0);
	}
	else {
		O_strm.write(ifm);
	}
}
}
}
}

void HW_conv2d_12(hls::stream<DATA_T> &I_strm, DATA_T W[256][256][3][3], DATA_T B[256], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ofm[256];
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[256][3][4];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_12_x_loop: for (x = 0; x<4 + 1; x++) {
	HW_conv2d_12_y_loop: for (y = 0; y<4 + 1; y++) {
	HW_conv2d_12_k_loop: for (k = 0; k<256; k++) {
		if (x < 4 && y < 4) {
			I[k][x % 3][y] = I_strm.read();
		}
		if (x && y) {
#pragma HLS PIPELINE
			if (x - 2 + 0 < 4 && y - 2 + 0 < 4 && x - 2 + 0 >= 0 && y - 2 + 0 >= 0) {
				I_0 = I[k][(x - 2) % 3][(y - 2)];
			}
			else {
				I_0 = 0; // zero padding
			}
			if (x - 2 + 0 < 4 && y - 2 + 1 < 4 && x - 2 + 0 >= 0 && y - 2 + 1 >= 0) {
				I_1 = I[k][(x - 2) % 3][(y - 1)];
			}
			else {
				I_1 = 0; // zero padding
			}
			if (x - 2 + 0 < 4 && y - 2 + 2 < 4 && x - 2 + 0 >= 0 && y - 2 + 2 >= 0) {
				I_2 = I[k][(x - 2) % 3][(y)];
			}
			else {
				I_2 = 0; // zero padding
			}

			if (x - 2 + 1 < 4 && y - 2 + 0 < 4 && x - 2 + 1 >= 0 && y - 2 + 0 >= 0) {
				I_3 = I[k][(x - 1) % 3][(y - 2)];
			}
			else {
				I_3 = 0; // zero padding
			}
			if (x - 2 + 1 < 4 && y - 2 + 1 < 4 && x - 2 + 1 >= 0 && y - 2 + 1 >= 0) {
				I_4 = I[k][(x - 1) % 3][(y - 1)];
			}
			else {
				I_4 = 0; // zero padding
			}
			if (x - 2 + 1 < 4 && y - 2 + 2 < 4 && x - 2 + 1 >= 0 && y - 2 + 2 >= 0) {
				I_5 = I[k][(x - 1) % 3][(y)];
			}
			else {
				I_5 = 0; // zero padding
			}
			if (x - 2 + 2 < 4 && y - 2 + 0 < 4 && x - 2 + 2 >= 0 && y - 2 + 0 >= 0) {
				I_6 = I[k][(x) % 3][(y - 2)];
			}
			else {
				I_6 = 0; // zero padding
			}
			if (x - 2 + 2 < 4 && y - 2 + 1 < 4 && x - 2 + 2 >= 0 && y - 2 + 1 >= 0) {
				I_7 = I[k][(x) % 3][(y - 1)];
			}
			else {
				I_7 = 0; // zero padding
			}
			if (x - 2 + 2 < 4 && y - 2 + 2 < 4 && x - 2 + 2 >= 0 && y - 2 + 2 >= 0) {
				I_8 = I[k][(x) % 3][(y)];
			}
			else {
				I_8 = 0; // zero padding
			}

		HW_conv2d_12_m_loop: for (m = 0; m<256; m++) {
#pragma HLS PIPELINE
			if (k == 0) {
				ofm[m] = B[m];
			}
			ofm[m] = ofm[m] + I_0 * W[m][k][0][0];
			ofm[m] = ofm[m] + I_1 * W[m][k][0][1];
			ofm[m] = ofm[m] + I_2 * W[m][k][0][2];
			ofm[m] = ofm[m] + I_3 * W[m][k][1][0];
			ofm[m] = ofm[m] + I_4 * W[m][k][1][1];
			ofm[m] = ofm[m] + I_5 * W[m][k][1][2];
			ofm[m] = ofm[m] + I_6 * W[m][k][2][0];
			ofm[m] = ofm[m] + I_7 * W[m][k][2][1];
			ofm[m] = ofm[m] + I_8 * W[m][k][2][2];
		}
							 if (k == 256 - 1) {
								 for (m = 0; m<256; m++) {
#pragma HLS UNROLL
									 O_strm.write(ofm[m]);
								 }
							 }
		}
	}
	}
	}
}

void HW_conv2d_13(hls::stream<DATA_T> &I_strm, DATA_T W[256][128], DATA_T B[256], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;
	DATA_T ofm[256];
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0
	DATA_T ifm;

HW_conv2d_13_x_loop: for (x = 0; x < 8; x++) {
HW_conv2d_13_y_loop: for (y = 0; y < 8; y++) {
HW_conv2d_13_k_loop: for (k = 0; k < 128; k++) {
	ifm = I_strm.read();
	if (x % 2 == 0 && y % 2 == 0) {
	HW_conv2d_13_m_loop: for (m = 0; m < 256; m++) {
#pragma HLS PIPELINE
		if (k == 0) {
			ofm[m] = B[m];
		}
		ofm[m] = ofm[m] + ifm * W[m][k];
	}
						 if (k == 128 - 1) {
							 for (m = 0; m < 256; m++) {
#pragma HLS UNROLL
								 O_strm.write(ofm[m]);
							 }
						 }
	}
}
}
}
}

void HW_activation_10(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm;
HW_activation_10_x_loop: for (x = 0; x < 4; x++) {
HW_activation_10_y_loop: for (y = 0; y < 4; y++) {
HW_activation_10_m_loop: for (m = 0; m < 256; m++) {
	ifm = I_strm.read();
	if (ifm < 0) {
		O_strm.write(0);
	}
	else {
		O_strm.write(ifm);
	}
}
}
}
}

void HW_add_5(hls::stream<DATA_T> &I1_strm, hls::stream<DATA_T> &I2_strm, hls::stream<DATA_T> &O1_strm, hls::stream<DATA_T> &O2_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm;
HW_add_5_x_loop_1: for (x = 0; x < 4; x++) {
HW_add_5_y_loop_1: for (y = 0; y < 4; y++) {
HW_add_5_m_loop_1: for (m = 0; m < 256; m++) {
	ifm = I1_strm.read();
	ifm = ifm + I2_strm.read();
	O1_strm.write(ifm);
	O2_strm.write(ifm);
}
}
}
}

void HW_conv2d_14(hls::stream<DATA_T> &I_strm, DATA_T W[256][256][3][3], DATA_T B[256], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ofm[256];
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[256][3][4];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_14_x_loop: for (x = 0; x<4 + 1; x++) {
	HW_conv2d_14_y_loop: for (y = 0; y<4 + 1; y++) {
	HW_conv2d_14_k_loop: for (k = 0; k<256; k++) {
		if (x < 4 && y < 4) {
			I[k][x % 3][y] = I_strm.read();
		}
		if (x && y) {
#pragma HLS PIPELINE
			if (x - 2 + 0 < 4 && y - 2 + 0 < 4 && x - 2 + 0 >= 0 && y - 2 + 0 >= 0) {
				I_0 = I[k][(x - 2) % 3][(y - 2)];
			}
			else {
				I_0 = 0; // zero padding
			}
			if (x - 2 + 0 < 4 && y - 2 + 1 < 4 && x - 2 + 0 >= 0 && y - 2 + 1 >= 0) {
				I_1 = I[k][(x - 2) % 3][(y - 1)];
			}
			else {
				I_1 = 0; // zero padding
			}
			if (x - 2 + 0 < 4 && y - 2 + 2 < 4 && x - 2 + 0 >= 0 && y - 2 + 2 >= 0) {
				I_2 = I[k][(x - 2) % 3][(y)];
			}
			else {
				I_2 = 0; // zero padding
			}

			if (x - 2 + 1 < 4 && y - 2 + 0 < 4 && x - 2 + 1 >= 0 && y - 2 + 0 >= 0) {
				I_3 = I[k][(x - 1) % 3][(y - 2)];
			}
			else {
				I_3 = 0; // zero padding
			}
			if (x - 2 + 1 < 4 && y - 2 + 1 < 4 && x - 2 + 1 >= 0 && y - 2 + 1 >= 0) {
				I_4 = I[k][(x - 1) % 3][(y - 1)];
			}
			else {
				I_4 = 0; // zero padding
			}
			if (x - 2 + 1 < 4 && y - 2 + 2 < 4 && x - 2 + 1 >= 0 && y - 2 + 2 >= 0) {
				I_5 = I[k][(x - 1) % 3][(y)];
			}
			else {
				I_5 = 0; // zero padding
			}
			if (x - 2 + 2 < 4 && y - 2 + 0 < 4 && x - 2 + 2 >= 0 && y - 2 + 0 >= 0) {
				I_6 = I[k][(x) % 3][(y - 2)];
			}
			else {
				I_6 = 0; // zero padding
			}
			if (x - 2 + 2 < 4 && y - 2 + 1 < 4 && x - 2 + 2 >= 0 && y - 2 + 1 >= 0) {
				I_7 = I[k][(x) % 3][(y - 1)];
			}
			else {
				I_7 = 0; // zero padding
			}
			if (x - 2 + 2 < 4 && y - 2 + 2 < 4 && x - 2 + 2 >= 0 && y - 2 + 2 >= 0) {
				I_8 = I[k][(x) % 3][(y)];
			}
			else {
				I_8 = 0; // zero padding
			}

		HW_conv2d_14_m_loop: for (m = 0; m<256; m++) {
#pragma HLS PIPELINE
			if (k == 0) {
				ofm[m] = B[m];
			}
			ofm[m] = ofm[m] + I_0 * W[m][k][0][0];
			ofm[m] = ofm[m] + I_1 * W[m][k][0][1];
			ofm[m] = ofm[m] + I_2 * W[m][k][0][2];
			ofm[m] = ofm[m] + I_3 * W[m][k][1][0];
			ofm[m] = ofm[m] + I_4 * W[m][k][1][1];
			ofm[m] = ofm[m] + I_5 * W[m][k][1][2];
			ofm[m] = ofm[m] + I_6 * W[m][k][2][0];
			ofm[m] = ofm[m] + I_7 * W[m][k][2][1];
			ofm[m] = ofm[m] + I_8 * W[m][k][2][2];
		}
							 if (k == 256 - 1) {
								 for (m = 0; m<256; m++) {
#pragma HLS UNROLL
									 O_strm.write(ofm[m]);
								 }
							 }
		}
	}
	}
	}
}

void HW_activation_11(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm;
HW_activation_11_x_loop: for (x = 0; x < 4; x++) {
HW_activation_11_y_loop: for (y = 0; y < 4; y++) {
HW_activation_11_m_loop: for (m = 0; m < 256; m++) {
	ifm = I_strm.read();
	if (ifm < 0) {
		O_strm.write(0);
	}
	else {
		O_strm.write(ifm);
	}
}
}
}
}

void HW_conv2d_15(hls::stream<DATA_T> &I_strm, DATA_T W[256][256][3][3], DATA_T B[256], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ofm[256];
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[256][3][4];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_15_x_loop: for (x = 0; x<4 + 1; x++) {
	HW_conv2d_15_y_loop: for (y = 0; y<4 + 1; y++) {
	HW_conv2d_15_k_loop: for (k = 0; k<256; k++) {
		if (x < 4 && y < 4) {
			I[k][x % 3][y] = I_strm.read();
		}
		if (x && y) {
#pragma HLS PIPELINE
			if (x - 2 + 0 < 4 && y - 2 + 0 < 4 && x - 2 + 0 >= 0 && y - 2 + 0 >= 0) {
				I_0 = I[k][(x - 2) % 3][(y - 2)];
			}
			else {
				I_0 = 0; // zero padding
			}
			if (x - 2 + 0 < 4 && y - 2 + 1 < 4 && x - 2 + 0 >= 0 && y - 2 + 1 >= 0) {
				I_1 = I[k][(x - 2) % 3][(y - 1)];
			}
			else {
				I_1 = 0; // zero padding
			}
			if (x - 2 + 0 < 4 && y - 2 + 2 < 4 && x - 2 + 0 >= 0 && y - 2 + 2 >= 0) {
				I_2 = I[k][(x - 2) % 3][(y)];
			}
			else {
				I_2 = 0; // zero padding
			}

			if (x - 2 + 1 < 4 && y - 2 + 0 < 4 && x - 2 + 1 >= 0 && y - 2 + 0 >= 0) {
				I_3 = I[k][(x - 1) % 3][(y - 2)];
			}
			else {
				I_3 = 0; // zero padding
			}
			if (x - 2 + 1 < 4 && y - 2 + 1 < 4 && x - 2 + 1 >= 0 && y - 2 + 1 >= 0) {
				I_4 = I[k][(x - 1) % 3][(y - 1)];
			}
			else {
				I_4 = 0; // zero padding
			}
			if (x - 2 + 1 < 4 && y - 2 + 2 < 4 && x - 2 + 1 >= 0 && y - 2 + 2 >= 0) {
				I_5 = I[k][(x - 1) % 3][(y)];
			}
			else {
				I_5 = 0; // zero padding
			}
			if (x - 2 + 2 < 4 && y - 2 + 0 < 4 && x - 2 + 2 >= 0 && y - 2 + 0 >= 0) {
				I_6 = I[k][(x) % 3][(y - 2)];
			}
			else {
				I_6 = 0; // zero padding
			}
			if (x - 2 + 2 < 4 && y - 2 + 1 < 4 && x - 2 + 2 >= 0 && y - 2 + 1 >= 0) {
				I_7 = I[k][(x) % 3][(y - 1)];
			}
			else {
				I_7 = 0; // zero padding
			}
			if (x - 2 + 2 < 4 && y - 2 + 2 < 4 && x - 2 + 2 >= 0 && y - 2 + 2 >= 0) {
				I_8 = I[k][(x) % 3][(y)];
			}
			else {
				I_8 = 0; // zero padding
			}

		HW_conv2d_15_m_loop: for (m = 0; m<256; m++) {
#pragma HLS PIPELINE
			if (k == 0) {
				ofm[m] = B[m];
			}
			ofm[m] = ofm[m] + I_0 * W[m][k][0][0];
			ofm[m] = ofm[m] + I_1 * W[m][k][0][1];
			ofm[m] = ofm[m] + I_2 * W[m][k][0][2];
			ofm[m] = ofm[m] + I_3 * W[m][k][1][0];
			ofm[m] = ofm[m] + I_4 * W[m][k][1][1];
			ofm[m] = ofm[m] + I_5 * W[m][k][1][2];
			ofm[m] = ofm[m] + I_6 * W[m][k][2][0];
			ofm[m] = ofm[m] + I_7 * W[m][k][2][1];
			ofm[m] = ofm[m] + I_8 * W[m][k][2][2];
		}
							 if (k == 256 - 1) {
								 for (m = 0; m<256; m++) {
#pragma HLS UNROLL
									 O_strm.write(ofm[m]);
								 }
							 }
		}
	}
	}
	}
}

void HW_activation_12(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm;
HW_activation_12_x_loop: for (x = 0; x < 4; x++) {
HW_activation_12_y_loop: for (y = 0; y < 4; y++) {
HW_activation_12_m_loop: for (m = 0; m < 256; m++) {
	ifm = I_strm.read();
	if (ifm < 0) {
		O_strm.write(0);
	}
	else {
		O_strm.write(ifm);
	}
}
}
}
}

void HW_add_6(hls::stream<DATA_T> &I1_strm, hls::stream<DATA_T> &I2_strm, hls::stream<DATA_T> &O1_strm, hls::stream<DATA_T> &O2_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm;
HW_add_6_x_loop_1: for (x = 0; x < 4; x++) {
HW_add_6_y_loop_1: for (y = 0; y < 4; y++) {
HW_add_6_m_loop_1: for (m = 0; m < 256; m++) {
	ifm = I1_strm.read();
	ifm = ifm + I2_strm.read();
	O1_strm.write(ifm);
	O2_strm.write(ifm);
}
}
}
}

void HW_conv2d_16(hls::stream<DATA_T> &I_strm, DATA_T W[512][256][3][3], DATA_T B[512], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;
	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ofm[512];
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[256][3][4];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_16_x_loop: for (x = 0; x<4 + 1; x++) {
	HW_conv2d_16_y_loop: for (y = 0; y<4 + 1; y++) {
	HW_conv2d_16_k_loop: for (k = 0; k<256; k++) {
		if (x < 4 && y < 4) {
			I[k][x % 3][y] = I_strm.read();
		}
		if ((x >> 1) && (y >> 1)) {
			if (x % 2 == 0 && y % 2 == 0) {
#pragma HLS PIPELINE
				if (x - 2 + 0 < 4 && y - 2 + 0 < 4) {
					I_0 = I[k][(x - 2) % 3][(y - 2)];
				}
				else {
					I_0 = 0; // zero padding
				}
				if (x - 2 + 0 < 4 && y - 2 + 1 < 4) {
					I_1 = I[k][(x - 2) % 3][(y - 1)];
				}
				else {
					I_1 = 0; // zero padding
				}
				if (x - 2 + 0 < 4 && y - 2 + 2 < 4) {
					I_2 = I[k][(x - 2) % 3][(y)];
				}
				else {
					I_2 = 0; // zero padding
				}

				if (x - 2 + 1 < 4 && y - 2 + 0 < 4) {
					I_3 = I[k][(x - 1) % 3][(y - 2)];
				}
				else {
					I_3 = 0; // zero padding
				}
				if (x - 2 + 1 < 4 && y - 2 + 1 < 4) {
					I_4 = I[k][(x - 1) % 3][(y - 1)];
				}
				else {
					I_4 = 0; // zero padding
				}
				if (x - 2 + 1 < 4 && y - 2 + 2 < 4) {
					I_5 = I[k][(x - 1) % 3][(y)];
				}
				else {
					I_5 = 0; // zero padding
				}
				if (x - 2 + 2 < 4 && y - 2 + 0 < 4) {
					I_6 = I[k][(x) % 3][(y - 2)];
				}
				else {
					I_6 = 0; // zero padding
				}
				if (x - 2 + 2 < 4 && y - 2 + 1 < 4) {
					I_7 = I[k][(x) % 3][(y - 1)];
				}
				else {
					I_7 = 0; // zero padding
				}
				if (x - 2 + 2 < 4 && y - 2 + 2 < 4) {
					I_8 = I[k][(x) % 3][(y)];
				}
				else {
					I_8 = 0; // zero padding
				}

			HW_conv2d_16_m_loop: for (m = 0; m<512; m++) {
#pragma HLS PIPELINE
				if (k == 0) {
					ofm[m] = B[m];
				}
				ofm[m] = ofm[m] + I_0 * W[m][k][0][0];
				ofm[m] = ofm[m] + I_1 * W[m][k][0][1];
				ofm[m] = ofm[m] + I_2 * W[m][k][0][2];
				ofm[m] = ofm[m] + I_3 * W[m][k][1][0];
				ofm[m] = ofm[m] + I_4 * W[m][k][1][1];
				ofm[m] = ofm[m] + I_5 * W[m][k][1][2];
				ofm[m] = ofm[m] + I_6 * W[m][k][2][0];
				ofm[m] = ofm[m] + I_7 * W[m][k][2][1];
				ofm[m] = ofm[m] + I_8 * W[m][k][2][2];
			}
								 if (k == 256 - 1) {
									 for (m = 0; m<512; m++) {
#pragma HLS UNROLL
										 O_strm.write(ofm[m]);
									 }
								 }
			}
		}
	}
	}
	}
}

void HW_activation_13(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm;
HW_activation_13_x_loop: for (x = 0; x < 2; x++) {
HW_activation_13_y_loop: for (y = 0; y < 2; y++) {
HW_activation_13_m_loop: for (m = 0; m < 512; m++) {
	ifm = I_strm.read();
	if (ifm < 0) {
		O_strm.write(0);
	}
	else {
		O_strm.write(ifm);
	}
}
}
}
}

void HW_conv2d_17(hls::stream<DATA_T> &I_strm, DATA_T W[512][512][3][3], DATA_T B[512], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ofm[512];
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[512][3][2];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_17_x_loop: for (x = 0; x<2 + 1; x++) {
	HW_conv2d_17_y_loop: for (y = 0; y<2 + 1; y++) {
	HW_conv2d_17_k_loop: for (k = 0; k<512; k++) {
		if (x < 2 && y < 2) {
			I[k][x % 3][y] = I_strm.read();
		}
		if (x && y) {
#pragma HLS PIPELINE
			if (x - 2 + 0 < 2 && y - 2 + 0 < 2 && x - 2 + 0 >= 0 && y - 2 + 0 >= 0) {
				I_0 = I[k][(x - 2) % 3][(y - 2)];
			}
			else {
				I_0 = 0; // zero padding
			}
			if (x - 2 + 0 < 2 && y - 2 + 1 < 2 && x - 2 + 0 >= 0 && y - 2 + 1 >= 0) {
				I_1 = I[k][(x - 2) % 3][(y - 1)];
			}
			else {
				I_1 = 0; // zero padding
			}
			if (x - 2 + 0 < 2 && y - 2 + 2 < 2 && x - 2 + 0 >= 0 && y - 2 + 2 >= 0) {
				I_2 = I[k][(x - 2) % 3][(y)];
			}
			else {
				I_2 = 0; // zero padding
			}

			if (x - 2 + 1 < 2 && y - 2 + 0 < 2 && x - 2 + 1 >= 0 && y - 2 + 0 >= 0) {
				I_3 = I[k][(x - 1) % 3][(y - 2)];
			}
			else {
				I_3 = 0; // zero padding
			}
			if (x - 2 + 1 < 2 && y - 2 + 1 < 2 && x - 2 + 1 >= 0 && y - 2 + 1 >= 0) {
				I_4 = I[k][(x - 1) % 3][(y - 1)];
			}
			else {
				I_4 = 0; // zero padding
			}
			if (x - 2 + 1 < 2 && y - 2 + 2 < 2 && x - 2 + 1 >= 0 && y - 2 + 2 >= 0) {
				I_5 = I[k][(x - 1) % 3][(y)];
			}
			else {
				I_5 = 0; // zero padding
			}
			if (x - 2 + 2 < 2 && y - 2 + 0 < 2 && x - 2 + 2 >= 0 && y - 2 + 0 >= 0) {
				I_6 = I[k][(x) % 3][(y - 2)];
			}
			else {
				I_6 = 0; // zero padding
			}
			if (x - 2 + 2 < 2 && y - 2 + 1 < 2 && x - 2 + 2 >= 0 && y - 2 + 1 >= 0) {
				I_7 = I[k][(x) % 3][(y - 1)];
			}
			else {
				I_7 = 0; // zero padding
			}
			if (x - 2 + 2 < 2 && y - 2 + 2 < 2 && x - 2 + 2 >= 0 && y - 2 + 2 >= 0) {
				I_8 = I[k][(x) % 3][(y)];
			}
			else {
				I_8 = 0; // zero padding
			}

		HW_conv2d_17_m_loop: for (m = 0; m<512; m++) {
#pragma HLS PIPELINE
			if (k == 0) {
				ofm[m] = B[m];
			}
			ofm[m] = ofm[m] + I_0 * W[m][k][0][0];
			ofm[m] = ofm[m] + I_1 * W[m][k][0][1];
			ofm[m] = ofm[m] + I_2 * W[m][k][0][2];
			ofm[m] = ofm[m] + I_3 * W[m][k][1][0];
			ofm[m] = ofm[m] + I_4 * W[m][k][1][1];
			ofm[m] = ofm[m] + I_5 * W[m][k][1][2];
			ofm[m] = ofm[m] + I_6 * W[m][k][2][0];
			ofm[m] = ofm[m] + I_7 * W[m][k][2][1];
			ofm[m] = ofm[m] + I_8 * W[m][k][2][2];
		}
							 if (k == 512 - 1) {
								 for (m = 0; m<512; m++) {
#pragma HLS UNROLL
									 O_strm.write(ofm[m]);
								 }
							 }
		}
	}
	}
	}
}

void HW_conv2d_18(hls::stream<DATA_T> &I_strm, DATA_T W[512][256], DATA_T B[512], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;
	DATA_T ofm[512];
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0
	DATA_T ifm;

HW_conv2d_18_x_loop: for (x = 0; x < 4; x++) {
HW_conv2d_18_y_loop: for (y = 0; y < 4; y++) {
HW_conv2d_18_k_loop: for (k = 0; k < 256; k++) {
	ifm = I_strm.read();
	if (x % 2 == 0 && y % 2 == 0) {
	HW_conv2d_18_m_loop: for (m = 0; m < 512; m++) {
#pragma HLS PIPELINE
		if (k == 0) {
			ofm[m] = B[m];
		}
		ofm[m] = ofm[m] + ifm * W[m][k];
	}
						 if (k == 256 - 1) {
							 for (m = 0; m < 512; m++) {
#pragma HLS UNROLL
								 O_strm.write(ofm[m]);
							 }
						 }
	}
}
}
}
}

void HW_activation_14(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm;
HW_activation_14_x_loop: for (x = 0; x < 2; x++) {
HW_activation_14_y_loop: for (y = 0; y < 2; y++) {
HW_activation_14_m_loop: for (m = 0; m < 512; m++) {
	ifm = I_strm.read();
	if (ifm < 0) {
		O_strm.write(0);
	}
	else {
		O_strm.write(ifm);
	}
}
}
}
}

void HW_add_7(hls::stream<DATA_T> &I1_strm, hls::stream<DATA_T> &I2_strm, hls::stream<DATA_T> &O1_strm, hls::stream<DATA_T> &O2_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm;
HW_add_7_x_loop_1: for (x = 0; x < 2; x++) {
HW_add_7_y_loop_1: for (y = 0; y < 2; y++) {
HW_add_7_m_loop_1: for (m = 0; m < 512; m++) {
	ifm = I1_strm.read();
	ifm = ifm + I2_strm.read();
	O1_strm.write(ifm);
	O2_strm.write(ifm);
}
}
}
}

void HW_conv2d_19(hls::stream<DATA_T> &I_strm, DATA_T W[512][512][3][3], DATA_T B[512], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ofm[512];
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[512][3][2];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_19_x_loop: for (x = 0; x<2 + 1; x++) {
	HW_conv2d_19_y_loop: for (y = 0; y<2 + 1; y++) {
	HW_conv2d_19_k_loop: for (k = 0; k<512; k++) {
		if (x < 2 && y < 2) {
			I[k][x % 3][y] = I_strm.read();
		}
		if (x && y) {
#pragma HLS PIPELINE
			if (x - 2 + 0 < 2 && y - 2 + 0 < 2 && x - 2 + 0 >= 0 && y - 2 + 0 >= 0) {
				I_0 = I[k][(x - 2) % 3][(y - 2)];
			}
			else {
				I_0 = 0; // zero padding
			}
			if (x - 2 + 0 < 2 && y - 2 + 1 < 2 && x - 2 + 0 >= 0 && y - 2 + 1 >= 0) {
				I_1 = I[k][(x - 2) % 3][(y - 1)];
			}
			else {
				I_1 = 0; // zero padding
			}
			if (x - 2 + 0 < 2 && y - 2 + 2 < 2 && x - 2 + 0 >= 0 && y - 2 + 2 >= 0) {
				I_2 = I[k][(x - 2) % 3][(y)];
			}
			else {
				I_2 = 0; // zero padding
			}

			if (x - 2 + 1 < 2 && y - 2 + 0 < 2 && x - 2 + 1 >= 0 && y - 2 + 0 >= 0) {
				I_3 = I[k][(x - 1) % 3][(y - 2)];
			}
			else {
				I_3 = 0; // zero padding
			}
			if (x - 2 + 1 < 2 && y - 2 + 1 < 2 && x - 2 + 1 >= 0 && y - 2 + 1 >= 0) {
				I_4 = I[k][(x - 1) % 3][(y - 1)];
			}
			else {
				I_4 = 0; // zero padding
			}
			if (x - 2 + 1 < 2 && y - 2 + 2 < 2 && x - 2 + 1 >= 0 && y - 2 + 2 >= 0) {
				I_5 = I[k][(x - 1) % 3][(y)];
			}
			else {
				I_5 = 0; // zero padding
			}
			if (x - 2 + 2 < 2 && y - 2 + 0 < 2 && x - 2 + 2 >= 0 && y - 2 + 0 >= 0) {
				I_6 = I[k][(x) % 3][(y - 2)];
			}
			else {
				I_6 = 0; // zero padding
			}
			if (x - 2 + 2 < 2 && y - 2 + 1 < 2 && x - 2 + 2 >= 0 && y - 2 + 1 >= 0) {
				I_7 = I[k][(x) % 3][(y - 1)];
			}
			else {
				I_7 = 0; // zero padding
			}
			if (x - 2 + 2 < 2 && y - 2 + 2 < 2 && x - 2 + 2 >= 0 && y - 2 + 2 >= 0) {
				I_8 = I[k][(x) % 3][(y)];
			}
			else {
				I_8 = 0; // zero padding
			}

		HW_conv2d_19_m_loop: for (m = 0; m<512; m++) {
#pragma HLS PIPELINE
			if (k == 0) {
				ofm[m] = B[m];
			}
			ofm[m] = ofm[m] + I_0 * W[m][k][0][0];
			ofm[m] = ofm[m] + I_1 * W[m][k][0][1];
			ofm[m] = ofm[m] + I_2 * W[m][k][0][2];
			ofm[m] = ofm[m] + I_3 * W[m][k][1][0];
			ofm[m] = ofm[m] + I_4 * W[m][k][1][1];
			ofm[m] = ofm[m] + I_5 * W[m][k][1][2];
			ofm[m] = ofm[m] + I_6 * W[m][k][2][0];
			ofm[m] = ofm[m] + I_7 * W[m][k][2][1];
			ofm[m] = ofm[m] + I_8 * W[m][k][2][2];
		}
							 if (k == 512 - 1) {
								 for (m = 0; m<512; m++) {
#pragma HLS UNROLL
									 O_strm.write(ofm[m]);
								 }
							 }
		}
	}
	}
	}
}

void HW_activation_15(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm;
HW_activation_15_x_loop: for (x = 0; x < 2; x++) {
HW_activation_15_y_loop: for (y = 0; y < 2; y++) {
HW_activation_15_m_loop: for (m = 0; m < 512; m++) {
	ifm = I_strm.read();
	if (ifm < 0) {
		O_strm.write(0);
	}
	else {
		O_strm.write(ifm);
	}
}
}
}
}

void HW_conv2d_20(hls::stream<DATA_T> &I_strm, DATA_T W[512][512][3][3], DATA_T B[512], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ofm[512];
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[512][3][2];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_20_x_loop: for (x = 0; x<2 + 1; x++) {
	HW_conv2d_20_y_loop: for (y = 0; y<2 + 1; y++) {
	HW_conv2d_20_k_loop: for (k = 0; k<512; k++) {
		if (x < 2 && y < 2) {
			I[k][x % 3][y] = I_strm.read();
		}
		if (x && y) {
#pragma HLS PIPELINE
			if (x - 2 + 0 < 2 && y - 2 + 0 < 2 && x - 2 + 0 >= 0 && y - 2 + 0 >= 0) {
				I_0 = I[k][(x - 2) % 3][(y - 2)];
			}
			else {
				I_0 = 0; // zero padding
			}
			if (x - 2 + 0 < 2 && y - 2 + 1 < 2 && x - 2 + 0 >= 0 && y - 2 + 1 >= 0) {
				I_1 = I[k][(x - 2) % 3][(y - 1)];
			}
			else {
				I_1 = 0; // zero padding
			}
			if (x - 2 + 0 < 2 && y - 2 + 2 < 2 && x - 2 + 0 >= 0 && y - 2 + 2 >= 0) {
				I_2 = I[k][(x - 2) % 3][(y)];
			}
			else {
				I_2 = 0; // zero padding
			}

			if (x - 2 + 1 < 2 && y - 2 + 0 < 2 && x - 2 + 1 >= 0 && y - 2 + 0 >= 0) {
				I_3 = I[k][(x - 1) % 3][(y - 2)];
			}
			else {
				I_3 = 0; // zero padding
			}
			if (x - 2 + 1 < 2 && y - 2 + 1 < 2 && x - 2 + 1 >= 0 && y - 2 + 1 >= 0) {
				I_4 = I[k][(x - 1) % 3][(y - 1)];
			}
			else {
				I_4 = 0; // zero padding
			}
			if (x - 2 + 1 < 2 && y - 2 + 2 < 2 && x - 2 + 1 >= 0 && y - 2 + 2 >= 0) {
				I_5 = I[k][(x - 1) % 3][(y)];
			}
			else {
				I_5 = 0; // zero padding
			}
			if (x - 2 + 2 < 2 && y - 2 + 0 < 2 && x - 2 + 2 >= 0 && y - 2 + 0 >= 0) {
				I_6 = I[k][(x) % 3][(y - 2)];
			}
			else {
				I_6 = 0; // zero padding
			}
			if (x - 2 + 2 < 2 && y - 2 + 1 < 2 && x - 2 + 2 >= 0 && y - 2 + 1 >= 0) {
				I_7 = I[k][(x) % 3][(y - 1)];
			}
			else {
				I_7 = 0; // zero padding
			}
			if (x - 2 + 2 < 2 && y - 2 + 2 < 2 && x - 2 + 2 >= 0 && y - 2 + 2 >= 0) {
				I_8 = I[k][(x) % 3][(y)];
			}
			else {
				I_8 = 0; // zero padding
			}

		HW_conv2d_20_m_loop: for (m = 0; m<512; m++) {
#pragma HLS PIPELINE
			if (k == 0) {
				ofm[m] = B[m];
			}
			ofm[m] = ofm[m] + I_0 * W[m][k][0][0];
			ofm[m] = ofm[m] + I_1 * W[m][k][0][1];
			ofm[m] = ofm[m] + I_2 * W[m][k][0][2];
			ofm[m] = ofm[m] + I_3 * W[m][k][1][0];
			ofm[m] = ofm[m] + I_4 * W[m][k][1][1];
			ofm[m] = ofm[m] + I_5 * W[m][k][1][2];
			ofm[m] = ofm[m] + I_6 * W[m][k][2][0];
			ofm[m] = ofm[m] + I_7 * W[m][k][2][1];
			ofm[m] = ofm[m] + I_8 * W[m][k][2][2];
		}
							 if (k == 512 - 1) {
								 for (m = 0; m<512; m++) {
#pragma HLS UNROLL
									 O_strm.write(ofm[m]);
								 }
							 }
		}
	}
	}
	}
}

void HW_activation_16(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm;
HW_activation_16_x_loop: for (x = 0; x < 2; x++) {
HW_activation_16_y_loop: for (y = 0; y < 2; y++) {
HW_activation_16_m_loop: for (m = 0; m < 512; m++) {
	ifm = I_strm.read();
	if (ifm < 0) {
		O_strm.write(0);
	}
	else {
		O_strm.write(ifm);
	}
}
}
}
}

void HW_add_8(hls::stream<DATA_T> &I1_strm, hls::stream<DATA_T> &I2_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm;
HW_add_8_x_loop_1: for (x = 0; x < 2; x++) {
HW_add_8_y_loop_1: for (y = 0; y < 2; y++) {
HW_add_8_m_loop_1: for (m = 0; m < 512; m++) {
	ifm = I1_strm.read();
	ifm = ifm + I2_strm.read();
	O_strm.write(ifm);
}
}
}
}

void HW_activation_17(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm;
HW_activation_17_x_loop: for (x = 0; x < 2; x++) {
HW_activation_17_y_loop: for (y = 0; y < 2; y++) {
HW_activation_17_m_loop: for (m = 0; m < 512; m++) {
	ifm = I_strm.read();
	if (ifm < 0) {
		O_strm.write(0);
	}
	else {
		O_strm.write(ifm);
	}
}
}
}
}

void HW_global_average_pooling2d_1(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	int div = 2 * 2;
	DATA_T ofm[512];
	DATA_T ifm;

HW_global_average_pooling2d_1_x_loop: for (x = 0; x<2; x++) {
HW_global_average_pooling2d_1_y_loop: for (y = 0; y<2; y++) {
HW_global_average_pooling2d_1_m_1_loop: for (m = 0; m<512; m++) {
#pragma HLS PIPELINE
	ifm = I_strm.read();
	if (x || y)
		ofm[m] += ifm;
	else
		ofm[m] = ifm;
}
}
									  if (x == 2 - 1) {
									  HW_global_average_pooling2d_1_m_2_loop: for (m = 0; m<512; m++) {
#pragma HLS UNROLL
										  ofm[m] /= div;
										  O_strm.write(ofm[m]);
									  }
									  }
}
}

void HW_dense_1(hls::stream<DATA_T> &I_strm, DATA_T W[10][512], DATA_T B[10], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, c;
	DATA_T maximum;
	DATA_T ofm[10];
	DATA_T ifm;
	//Dense
HW_dense_1_c_loop: for (c = 0; c < 512; c++) {
	ifm = I_strm.read();
HW_dense_1_m_1_loop: for (m = 0; m<10; m++) {
#pragma HLS PIPELINE
	if (c == 0)
		ofm[m] = B[m];
	ofm[m] += W[m][c] * ifm;
}
}
				   //find Max
				   maximum = ofm[0];
			   HW_dense_1_m_2_loop: for (m = 1; m < 10; m++) {
				   if (maximum<ofm[m])
					   maximum = ofm[m];
			   }
									//one hot label
								HW_dense_1_m_3_loop: for (m = 0; m < 10; m++) {
									if (maximum != ofm[m])
										O_strm.write(0);
									else
										O_strm.write(1);
								}
}



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
							if (x * 1 + i < 32 + 1 && y * 1 + j < 32 + 1 && x * 1 + i - 1 >= 0 && y * 1 + j - 1 >= 0) {
								ifm = I[k][x * 1 + i - 1][y * 1 + j - 1];
							}
							else {
								ifm = 0; // zero padding
							}
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
							if (x * 2 + i < 32 + 0 && y * 2 + j < 32 + 0 && x * 2 + i - 0 >= 0 && y * 2 + j - 0 >= 0) {
								ifm = I[k][x * 2 + i - 0][y * 2 + j - 0];
							}
							else {
								ifm = 0; // zero padding
							}
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
							if (x * 1 + i < 16 + 1 && y * 1 + j < 16 + 1 && x * 1 + i - 1 >= 0 && y * 1 + j - 1 >= 0) {
								ifm = I[k][x * 1 + i - 1][y * 1 + j - 1];
							}
							else {
								ifm = 0; // zero padding
							}
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
								ifm = I[k][x * 2 + i][y * 2 + j];
							}

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
			for (y = 0; y < 16; y++) {
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
							if (x * 1 + i < 16 + 1 && y * 1 + j < 16 + 1 && x * 1 + i - 1 >= 0 && y * 1 + j - 1 >= 0) {
								ifm = I[k][x * 1 + i - 1][y * 1 + j - 1];
							}
							else {
								ifm = 0; // zero padding
							}
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
							if (x * 1 + i < 16 + 1 && y * 1 + j < 16 + 1 && x * 1 + i - 1 >= 0 && y * 1 + j - 1 >= 0) {
								ifm = I[k][x * 1 + i - 1][y * 1 + j - 1];
							}
							else {
								ifm = 0; // zero padding
							}
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
			for (y = 0; y < 16; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_conv2d_6(DATA_T I[64][16][16], DATA_T O[128][8][8], DATA_T W[128][64][3][3], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<8; x++) {
			for (y = 0; y<8; y++) {
				ofm = B[m];
				for (k = 0; k<64; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x * 2 + i < 16 + 0 && y * 2 + j < 16 + 0 && x * 2 + i - 0 >= 0 && y * 2 + j - 0 >= 0) {
								ifm = I[k][x * 2 + i - 0][y * 2 + j - 0];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
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
	for (m = 0; m<128; m++) {
		for (x = 0; x<8; x++) {
			for (y = 0; y<8; y++) {
				ofm = B[m];
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x * 1 + i < 8 + 1 && y * 1 + j < 8 + 1 && x * 1 + i - 1 >= 0 && y * 1 + j - 1 >= 0) {
								ifm = I[k][x * 1 + i - 1][y * 1 + j - 1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
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
								ifm = I[k][x * 2 + i][y * 2 + j];
							}

							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
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
			for (y = 0; y < 8; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
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
							if (x * 1 + i < 8 + 1 && y * 1 + j < 8 + 1 && x * 1 + i - 1 >= 0 && y * 1 + j - 1 >= 0) {
								ifm = I[k][x * 1 + i - 1][y * 1 + j - 1];
							}
							else {
								ifm = 0; // zero padding
							}
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

void SW_conv2d_10(DATA_T I[128][8][8], DATA_T O[128][8][8], DATA_T W[128][128][3][3], DATA_T B[128]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<128; m++) {
		for (x = 0; x<8; x++) {
			for (y = 0; y<8; y++) {
				ofm = B[m];
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x * 1 + i < 8 + 1 && y * 1 + j < 8 + 1 && x * 1 + i - 1 >= 0 && y * 1 + j - 1 >= 0) {
								ifm = I[k][x * 1 + i - 1][y * 1 + j - 1];
							}
							else {
								ifm = 0; // zero padding
							}
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
			for (y = 0; y < 8; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_conv2d_11(DATA_T I[128][8][8], DATA_T O[256][4][4], DATA_T W[256][128][3][3], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<4; x++) {
			for (y = 0; y<4; y++) {
				ofm = B[m];
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x * 2 + i < 8 + 0 && y * 2 + j < 8 + 0 && x * 2 + i - 0 >= 0 && y * 2 + j - 0 >= 0) {
								ifm = I[k][x * 2 + i - 0][y * 2 + j - 0];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
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
	for (m = 0; m<256; m++) {
		for (x = 0; x<4; x++) {
			for (y = 0; y<4; y++) {
				ofm = B[m];
				for (k = 0; k<256; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x * 1 + i < 4 + 1 && y * 1 + j < 4 + 1 && x * 1 + i - 1 >= 0 && y * 1 + j - 1 >= 0) {
								ifm = I[k][x * 1 + i - 1][y * 1 + j - 1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
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
								ifm = I[k][x * 2 + i][y * 2 + j];
							}

							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
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
			for (y = 0; y < 4; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_conv2d_14(DATA_T I[256][4][4], DATA_T O[256][4][4], DATA_T W[256][256][3][3], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<256; m++) {
		for (x = 0; x<4; x++) {
			for (y = 0; y<4; y++) {
				ofm = B[m];
				for (k = 0; k<256; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x * 1 + i < 4 + 1 && y * 1 + j < 4 + 1 && x * 1 + i - 1 >= 0 && y * 1 + j - 1 >= 0) {
								ifm = I[k][x * 1 + i - 1][y * 1 + j - 1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
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
	for (m = 0; m<256; m++) {
		for (x = 0; x<4; x++) {
			for (y = 0; y<4; y++) {
				ofm = B[m];
				for (k = 0; k<256; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x * 1 + i < 4 + 1 && y * 1 + j < 4 + 1 && x * 1 + i - 1 >= 0 && y * 1 + j - 1 >= 0) {
								ifm = I[k][x * 1 + i - 1][y * 1 + j - 1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
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
			for (y = 0; y < 4; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_conv2d_16(DATA_T I[256][4][4], DATA_T O[512][2][2], DATA_T W[512][256][3][3], DATA_T B[512]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<512; m++) {
		for (x = 0; x<2; x++) {
			for (y = 0; y<2; y++) {
				ofm = B[m];
				for (k = 0; k<256; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x * 2 + i < 4 + 0 && y * 2 + j < 4 + 0 && x * 2 + i - 0 >= 0 && y * 2 + j - 0 >= 0) {
								ifm = I[k][x * 2 + i - 0][y * 2 + j - 0];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
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
	for (m = 0; m<512; m++) {
		for (x = 0; x<2; x++) {
			for (y = 0; y<2; y++) {
				ofm = B[m];
				for (k = 0; k<512; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x * 1 + i < 2 + 1 && y * 1 + j < 2 + 1 && x * 1 + i - 1 >= 0 && y * 1 + j - 1 >= 0) {
								ifm = I[k][x * 1 + i - 1][y * 1 + j - 1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
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
								ifm = I[k][x * 2 + i][y * 2 + j];
							}

							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
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
			for (y = 0; y < 2; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}
void SW_conv2d_19(DATA_T I[512][2][2], DATA_T O[512][2][2], DATA_T W[512][512][3][3], DATA_T B[512]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<512; m++) {
		for (x = 0; x<2; x++) {
			for (y = 0; y<2; y++) {
				ofm = B[m];
				for (k = 0; k<512; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x * 1 + i < 2 + 1 && y * 1 + j < 2 + 1 && x * 1 + i - 1 >= 0 && y * 1 + j - 1 >= 0) {
								ifm = I[k][x * 1 + i - 1][y * 1 + j - 1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
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
	for (m = 0; m<512; m++) {
		for (x = 0; x<2; x++) {
			for (y = 0; y<2; y++) {
				ofm = B[m];
				for (k = 0; k<512; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x * 1 + i < 2 + 1 && y * 1 + j < 2 + 1 && x * 1 + i - 1 >= 0 && y * 1 + j - 1 >= 0) {
								ifm = I[k][x * 1 + i - 1][y * 1 + j - 1];
							}
							else {
								ifm = 0; // zero padding
							}
							ofm = ofm + ifm * W[m][k][i][j];
						}
					}
				}
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
			for (y = 0; y < 2; y++) {
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
	DATA_T avg;
	int div = 2 * 2;
	for (m = 0; m < 512; m++) {
		avg = 0;
		for (x = 0; x < 2; x++) {
			for (y = 0; y < 2; y++) {
				avg += I[m][x][y];
			}
		}
		O[m] = avg / div;
	}
}
void SW_dense_1(DATA_T I[512], DATA_T O[10], DATA_T W[10][512], DATA_T B[10])
{
	//Dense
	int m, c;
	DATA_T maximum = 0;

	for (m = 0; m<10; m++) {
		O[m] = B[m];
		for (c = 0; c < 512; c++) {
			O[m] += W[m][c] * I[c];
		}
	}
	//Find max
	for (m = 0; m < 10; m++) {
		if (maximum<O[m])
			maximum = O[m];
	}
	//One hot key
	for (m = 0; m < 10; m++) {
		if (maximum != O[m])
			O[m] = 0;
		else
			O[m] = 1;
	}
}


void resnet18(DATA_T I_i[3][32][32], DATA_T W1_i[64][3][3][3], DATA_T B1_i[64], DATA_T W3_i[64][64][3][3], DATA_T B3_i[64], DATA_T W4_i[64][64][3][3], DATA_T B4_i[64], DATA_T W5_i[64][64], DATA_T B5_i[64], DATA_T W8_i[64][64][3][3], DATA_T B8_i[64], DATA_T W10_i[64][64][3][3], DATA_T B10_i[64], DATA_T W13_i[128][64][3][3], DATA_T B13_i[128], DATA_T W15_i[128][128][3][3], DATA_T B15_i[128], DATA_T W16_i[128][64], DATA_T B16_i[128], DATA_T W19_i[128][128][3][3], DATA_T B19_i[128], DATA_T W21_i[128][128][3][3], DATA_T B21_i[128], DATA_T W24_i[256][128][3][3], DATA_T B24_i[256], DATA_T W26_i[256][256][3][3], DATA_T B26_i[256], DATA_T W27_i[256][128], DATA_T B27_i[256], DATA_T W30_i[256][256][3][3], DATA_T B30_i[256], DATA_T W32_i[256][256][3][3], DATA_T B32_i[256], DATA_T W35_i[512][256][3][3], DATA_T B35_i[512], DATA_T W37_i[512][512][3][3], DATA_T B37_i[512], DATA_T W38_i[512][256], DATA_T B38_i[512], DATA_T W41_i[512][512][3][3], DATA_T B41_i[512], DATA_T W43_i[512][512][3][3], DATA_T B43_i[512], DATA_T W48_i[10][512], DATA_T B48_i[10], DATA_T O[10]) {
#pragma HLS ARRAY_PARTITION variable=W1_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W1_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W1_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B1_i complete
#pragma HLS ARRAY_PARTITION variable=W3_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W3_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W3_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B3_i complete
#pragma HLS ARRAY_PARTITION variable=W4_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W4_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W4_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B4_i complete
#pragma HLS ARRAY_PARTITION variable=W5_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=B5_i complete
#pragma HLS ARRAY_PARTITION variable=W8_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W8_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W8_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B8_i complete
#pragma HLS ARRAY_PARTITION variable=W10_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W10_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W10_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B10_i complete
#pragma HLS ARRAY_PARTITION variable=W13_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W13_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W13_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B13_i complete
#pragma HLS ARRAY_PARTITION variable=W15_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W15_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W15_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B15_i complete
#pragma HLS ARRAY_PARTITION variable=W16_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=B16_i complete
#pragma HLS ARRAY_PARTITION variable=W19_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W19_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W19_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B19_i complete
#pragma HLS ARRAY_PARTITION variable=W21_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W21_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W21_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B21_i complete
#pragma HLS ARRAY_PARTITION variable=W24_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W24_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W24_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B24_i complete
#pragma HLS ARRAY_PARTITION variable=W26_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W26_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W26_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B26_i complete
#pragma HLS ARRAY_PARTITION variable=W27_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=B27_i complete
#pragma HLS ARRAY_PARTITION variable=W30_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W30_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W30_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B30_i complete
#pragma HLS ARRAY_PARTITION variable=W32_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W32_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W32_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B32_i complete
#pragma HLS ARRAY_PARTITION variable=W35_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W35_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W35_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B35_i complete
#pragma HLS ARRAY_PARTITION variable=W37_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W37_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W37_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B37_i complete
#pragma HLS ARRAY_PARTITION variable=W38_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=B38_i complete
#pragma HLS ARRAY_PARTITION variable=W41_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W41_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W41_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B41_i complete
#pragma HLS ARRAY_PARTITION variable=W43_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W43_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W43_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B43_i complete
#pragma HLS ARRAY_PARTITION variable=W48_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=B48_i complete


#pragma HLS DATAFLOW

	static hls::stream<DATA_T> O0_strm("O0_strm");
	static hls::stream<DATA_T> O1_strm("O1_strm");
	static hls::stream<DATA_T> O2_strm("O2_strm");
	static hls::stream<DATA_T> O3_strm("O3_strm");
	static hls::stream<DATA_T> O4_strm("O4_strm");
	static hls::stream<DATA_T> O5_strm("O5_strm");
	static hls::stream<DATA_T> O6_strm("O6_strm");
	static hls::stream<DATA_T> O7_strm("O7_strm");
	static hls::stream<DATA_T> O8_strm("O8_strm");
	static hls::stream<DATA_T> O9_strm("O9_strm");
	static hls::stream<DATA_T> O10_strm("O10_strm");
	static hls::stream<DATA_T> O11_strm("O11_strm");
	static hls::stream<DATA_T> O12_strm("O12_strm");
	static hls::stream<DATA_T> O13_strm("O13_strm");
	static hls::stream<DATA_T> O14_strm("O14_strm");
	static hls::stream<DATA_T> O15_strm("O15_strm");
	static hls::stream<DATA_T> O16_strm("O16_strm");
	static hls::stream<DATA_T> O17_strm("O17_strm");
	static hls::stream<DATA_T> O18_strm("O18_strm");
	static hls::stream<DATA_T> O19_strm("O19_strm");
	static hls::stream<DATA_T> O20_strm("O20_strm");
	static hls::stream<DATA_T> O21_strm("O21_strm");
	static hls::stream<DATA_T> O22_strm("O22_strm");
	static hls::stream<DATA_T> O23_strm("O23_strm");
	static hls::stream<DATA_T> O24_strm("O24_strm");
	static hls::stream<DATA_T> O25_strm("O25_strm");
	static hls::stream<DATA_T> O26_strm("O26_strm");
	static hls::stream<DATA_T> O27_strm("O27_strm");
	static hls::stream<DATA_T> O28_strm("O28_strm");
	static hls::stream<DATA_T> O29_strm("O29_strm");
	static hls::stream<DATA_T> O30_strm("O30_strm");
	static hls::stream<DATA_T> O31_strm("O31_strm");
	static hls::stream<DATA_T> O32_strm("O32_strm");
	static hls::stream<DATA_T> O33_strm("O33_strm");
	static hls::stream<DATA_T> O34_strm("O34_strm");
	static hls::stream<DATA_T> O35_strm("O35_strm");
	static hls::stream<DATA_T> O36_strm("O36_strm");
	static hls::stream<DATA_T> O37_strm("O37_strm");
	static hls::stream<DATA_T> O38_strm("O38_strm");
	static hls::stream<DATA_T> O39_strm("O39_strm");
	static hls::stream<DATA_T> O40_strm("O40_strm");
	static hls::stream<DATA_T> O41_strm("O41_strm");
	static hls::stream<DATA_T> O42_strm("O42_strm");
	static hls::stream<DATA_T> O43_strm("O43_strm");
	static hls::stream<DATA_T> O44_strm("O44_strm");
	static hls::stream<DATA_T> O45_strm("O45_strm");
	static hls::stream<DATA_T> O46_strm("O46_strm");
	static hls::stream<DATA_T> O47_strm("O47_strm");
	static hls::stream<DATA_T> O48_strm("O48_strm");
	static hls::stream<DATA_T> O49_strm("O49_strm");
	static hls::stream<DATA_T> O50_strm("O50_strm");
	static hls::stream<DATA_T> O51_strm("O51_strm");
	static hls::stream<DATA_T> O52_strm("O52_strm");
	static hls::stream<DATA_T> O53_strm("O53_strm");
	static hls::stream<DATA_T> O54_strm("O54_strm");
	static hls::stream<DATA_T> O55_strm("O55_strm");
	static hls::stream<DATA_T> O56_strm("O56_strm");

	Stream_input(I_i, O0_strm);
	HW_conv2d_1(O0_strm, W1_i, B1_i, O1_strm);
	HW_activation_1(O1_strm, O2_strm, O4_strm); //double
	HW_res0a_branch2a(O2_strm, W3_i, B3_i, O3_strm);
	HW_conv2d_2(O3_strm, W4_i, B4_i, O6_strm);
	HW_conv2d_3(O4_strm, W5_i, B5_i, O5_strm);
	HW_activation_2(O6_strm, O7_strm);
	HW_add_1(O5_strm, O7_strm, O8_strm, O9_strm); //double
	HW_conv2d_4(O8_strm, W8_i, B8_i, O10_strm);
	HW_activation_3(O10_strm, O11_strm);
	HW_conv2d_5(O11_strm, W10_i, B10_i, O12_strm);
	HW_activation_4(O12_strm, O13_strm);
	HW_add_2(O9_strm, O13_strm, O14_strm, O15_strm);//double
	HW_conv2d_6(O14_strm, W13_i, B13_i, O16_strm);
	HW_activation_5(O16_strm, O17_strm);
	HW_conv2d_7(O17_strm, W15_i, B15_i, O18_strm);
	HW_conv2d_8(O15_strm, W16_i, B16_i, O19_strm);
	HW_activation_6(O18_strm, O20_strm);
	HW_add_3(O19_strm, O20_strm, O21_strm, O22_strm);//double
	HW_conv2d_9(O21_strm, W19_i, B19_i, O23_strm);
	HW_activation_7(O23_strm, O24_strm);
	HW_conv2d_10(O24_strm, W21_i, B21_i, O25_strm);
	HW_activation_8(O25_strm, O26_strm);
	HW_add_4(O22_strm, O26_strm, O27_strm, O28_strm); //double
	HW_conv2d_11(O27_strm, W24_i, B24_i, O29_strm);
	HW_activation_9(O29_strm, O30_strm);
	HW_conv2d_12(O30_strm, W26_i, B26_i, O31_strm);
	HW_conv2d_13(O28_strm, W27_i, B27_i, O32_strm);
	HW_activation_10(O31_strm, O33_strm);
	HW_add_5(O32_strm, O33_strm, O34_strm, O35_strm); //double
	HW_conv2d_14(O34_strm, W30_i, B30_i, O36_strm);
	HW_activation_11(O36_strm, O37_strm);
	HW_conv2d_15(O37_strm, W32_i, B32_i, O38_strm);
	HW_activation_12(O38_strm, O39_strm);
	HW_add_6(O35_strm, O39_strm, O40_strm, O41_strm); //double
	HW_conv2d_16(O40_strm, W35_i, B35_i, O42_strm);
	HW_activation_13(O42_strm, O43_strm);
	HW_conv2d_17(O43_strm, W37_i, B37_i, O44_strm);
	HW_conv2d_18(O41_strm, W38_i, B38_i, O45_strm);
	HW_activation_14(O44_strm, O46_strm);
	HW_add_7(O46_strm, O45_strm, O47_strm, O48_strm); //double
	HW_conv2d_19(O47_strm, W41_i, B41_i, O49_strm);
	HW_activation_15(O49_strm, O50_strm);
	HW_conv2d_20(O50_strm, W43_i, B43_i, O51_strm);
	HW_activation_16(O51_strm, O52_strm);
	HW_add_8(O48_strm, O52_strm, O53_strm);
	HW_activation_17(O53_strm, O54_strm);
	HW_global_average_pooling2d_1(O54_strm, O55_strm);
	HW_dense_1(O55_strm, W48_i, B48_i, O56_strm);
	Stream_output(O56_strm, O);

}

void resnet18_top(DATA_T I[3][32][32], DATA_T W1[64][3][3][3], DATA_T B1[64], DATA_T W3[64][64][3][3], DATA_T B3[64], DATA_T W4[64][64][3][3], DATA_T B4[64], DATA_T W5[64][64][1][1], DATA_T B5[64], DATA_T W8[64][64][3][3], DATA_T B8[64], DATA_T W10[64][64][3][3], DATA_T B10[64], DATA_T W13[128][64][3][3], DATA_T B13[128], DATA_T W15[128][128][3][3], DATA_T B15[128], DATA_T W16[128][64][1][1], DATA_T B16[128], DATA_T W19[128][128][3][3], DATA_T B19[128], DATA_T W21[128][128][3][3], DATA_T B21[128], DATA_T W24[256][128][3][3], DATA_T B24[256], DATA_T W26[256][256][3][3], DATA_T B26[256], DATA_T W27[256][128][1][1], DATA_T B27[256], DATA_T W30[256][256][3][3], DATA_T B30[256], DATA_T W32[256][256][3][3], DATA_T B32[256], DATA_T W35[512][256][3][3], DATA_T B35[512], DATA_T W37[512][512][3][3], DATA_T B37[512], DATA_T W38[512][256][1][1], DATA_T B38[512], DATA_T W41[512][512][3][3], DATA_T B41[512], DATA_T W43[512][512][3][3], DATA_T B43[512], DATA_T W48[10][512], DATA_T B48[10], DATA_T O[10]) {

	DATA_T O_i[10];
	int m, x, y, i, j, k;
            I_i_k_loop: for (k=0; k<3; k++) {
  I_i_x_loop: for (x=0; x<32; x++) {
  I_i_y_loop: for (y=0; y<32; y++) {
  I_i[k][x][y] = I[k][x][y];
    }
  }
}
W1_i_m_loop: for (m=0; m<64; m++) {
  W1_i_k_loop: for (k=0; k<3; k++) {
  W1_i_i_loop: for (i=0; i<3; i++) {
  W1_i_j_loop: for (j=0; j<3; j++) {
  W1_i[m][k][i][j] = W1[m][k][i][j];
      }
    }
  }
}
B1_i_m_loop: for (m=0; m<64; m++) {
	B1_i[m] = B1[m];
}
W3_i_m_loop: for (m=0; m<64; m++) {
  W3_i_k_loop: for (k=0; k<64; k++) {
  W3_i_i_loop: for (i=0; i<3; i++) {
  W3_i_j_loop: for (j=0; j<3; j++) {
  W3_i[m][k][i][j] = W3[m][k][i][j];
      }
    }
  }
}
B3_i_m_loop: for (m=0; m<64; m++) {
	B3_i[m] = B3[m];
}
W4_i_m_loop: for (m=0; m<64; m++) {
  W4_i_k_loop: for (k=0; k<64; k++) {
  W4_i_i_loop: for (i=0; i<3; i++) {
  W4_i_j_loop: for (j=0; j<3; j++) {
  W4_i[m][k][i][j] = W4[m][k][i][j];
      }
    }
  }
}
B4_i_m_loop: for (m=0; m<64; m++) {
	B4_i[m] = B4[m];
}
W5_i_m_loop: for (m=0; m<64; m++) {
  W5_i_k_loop: for (k=0; k<64; k++) {
  W5_i[m][k] = W5[m][k][0][0];
  }
}
B5_i_m_loop: for (m=0; m<64; m++) {
	B5_i[m] = B5[m];
}
W8_i_m_loop: for (m=0; m<64; m++) {
  W8_i_k_loop: for (k=0; k<64; k++) {
  W8_i_i_loop: for (i=0; i<3; i++) {
  W8_i_j_loop: for (j=0; j<3; j++) {
  W8_i[m][k][i][j] = W8[m][k][i][j];
      }
    }
  }
}
B8_i_m_loop: for (m=0; m<64; m++) {
	B8_i[m] = B8[m];
}
W10_i_m_loop: for (m=0; m<64; m++) {
  W10_i_k_loop: for (k=0; k<64; k++) {
  W10_i_i_loop: for (i=0; i<3; i++) {
  W10_i_j_loop: for (j=0; j<3; j++) {
  W10_i[m][k][i][j] = W10[m][k][i][j];
      }
    }
  }
}
B10_i_m_loop: for (m=0; m<64; m++) {
	B10_i[m] = B10[m];
}
W13_i_m_loop: for (m=0; m<128; m++) {
  W13_i_k_loop: for (k=0; k<64; k++) {
  W13_i_i_loop: for (i=0; i<3; i++) {
  W13_i_j_loop: for (j=0; j<3; j++) {
  W13_i[m][k][i][j] = W13[m][k][i][j];
      }
    }
  }
}
B13_i_m_loop: for (m=0; m<128; m++) {
	B13_i[m] = B13[m];
}
W15_i_m_loop: for (m=0; m<128; m++) {
  W15_i_k_loop: for (k=0; k<128; k++) {
  W15_i_i_loop: for (i=0; i<3; i++) {
  W15_i_j_loop: for (j=0; j<3; j++) {
  W15_i[m][k][i][j] = W15[m][k][i][j];
      }
    }
  }
}
B15_i_m_loop: for (m=0; m<128; m++) {
	B15_i[m] = B15[m];
}
W16_i_m_loop: for (m=0; m<128; m++) {
  W16_i_k_loop: for (k=0; k<64; k++) {
  W16_i[m][k] = W16[m][k][0][0];
  }
}
B16_i_m_loop: for (m=0; m<128; m++) {
	B16_i[m] = B16[m];
}
W19_i_m_loop: for (m=0; m<128; m++) {
  W19_i_k_loop: for (k=0; k<128; k++) {
  W19_i_i_loop: for (i=0; i<3; i++) {
  W19_i_j_loop: for (j=0; j<3; j++) {
  W19_i[m][k][i][j] = W19[m][k][i][j];
      }
    }
  }
}
B19_i_m_loop: for (m=0; m<128; m++) {
	B19_i[m] = B19[m];
}
W21_i_m_loop: for (m=0; m<128; m++) {
  W21_i_k_loop: for (k=0; k<128; k++) {
  W21_i_i_loop: for (i=0; i<3; i++) {
  W21_i_j_loop: for (j=0; j<3; j++) {
  W21_i[m][k][i][j] = W21[m][k][i][j];
      }
    }
  }
}
B21_i_m_loop: for (m=0; m<128; m++) {
	B21_i[m] = B21[m];
}
W24_i_m_loop: for (m=0; m<256; m++) {
  W24_i_k_loop: for (k=0; k<128; k++) {
  W24_i_i_loop: for (i=0; i<3; i++) {
  W24_i_j_loop: for (j=0; j<3; j++) {
  W24_i[m][k][i][j] = W24[m][k][i][j];
      }
    }
  }
}
B24_i_m_loop: for (m=0; m<256; m++) {
	B24_i[m] = B24[m];
}
W26_i_m_loop: for (m=0; m<256; m++) {
  W26_i_k_loop: for (k=0; k<256; k++) {
  W26_i_i_loop: for (i=0; i<3; i++) {
  W26_i_j_loop: for (j=0; j<3; j++) {
  W26_i[m][k][i][j] = W26[m][k][i][j];
      }
    }
  }
}
B26_i_m_loop: for (m=0; m<256; m++) {
	B26_i[m] = B26[m];
}
W27_i_m_loop: for (m=0; m<256; m++) {
  W27_i_k_loop: for (k=0; k<128; k++) {
  W27_i[m][k] = W27[m][k][0][0];
  }
}
B27_i_m_loop: for (m=0; m<256; m++) {
	B27_i[m] = B27[m];
}
W30_i_m_loop: for (m=0; m<256; m++) {
  W30_i_k_loop: for (k=0; k<256; k++) {
  W30_i_i_loop: for (i=0; i<3; i++) {
  W30_i_j_loop: for (j=0; j<3; j++) {
  W30_i[m][k][i][j] = W30[m][k][i][j];
      }
    }
  }
}
B30_i_m_loop: for (m=0; m<256; m++) {
	B30_i[m] = B30[m];
}
W32_i_m_loop: for (m=0; m<256; m++) {
  W32_i_k_loop: for (k=0; k<256; k++) {
  W32_i_i_loop: for (i=0; i<3; i++) {
  W32_i_j_loop: for (j=0; j<3; j++) {
  W32_i[m][k][i][j] = W32[m][k][i][j];
      }
    }
  }
}
B32_i_m_loop: for (m=0; m<256; m++) {
	B32_i[m] = B32[m];
}
W35_i_m_loop: for (m=0; m<512; m++) {
  W35_i_k_loop: for (k=0; k<256; k++) {
  W35_i_i_loop: for (i=0; i<3; i++) {
  W35_i_j_loop: for (j=0; j<3; j++) {
  W35_i[m][k][i][j] = W35[m][k][i][j];
      }
    }
  }
}
B35_i_m_loop: for (m=0; m<512; m++) {
	B35_i[m] = B35[m];
}
W37_i_m_loop: for (m=0; m<512; m++) {
  W37_i_k_loop: for (k=0; k<512; k++) {
  W37_i_i_loop: for (i=0; i<3; i++) {
  W37_i_j_loop: for (j=0; j<3; j++) {
  W37_i[m][k][i][j] = W37[m][k][i][j];
      }
    }
  }
}
B37_i_m_loop: for (m=0; m<512; m++) {
	B37_i[m] = B37[m];
}
W38_i_m_loop: for (m=0; m<512; m++) {
  W38_i_k_loop: for (k=0; k<256; k++) {
  W38_i[m][k] = W38[m][k][0][0];
  }
}
B38_i_m_loop: for (m=0; m<512; m++) {
	B38_i[m] = B38[m];
}
W41_i_m_loop: for (m=0; m<512; m++) {
  W41_i_k_loop: for (k=0; k<512; k++) {
  W41_i_i_loop: for (i=0; i<3; i++) {
  W41_i_j_loop: for (j=0; j<3; j++) {
  W41_i[m][k][i][j] = W41[m][k][i][j];
      }
    }
  }
}
B41_i_m_loop: for (m=0; m<512; m++) {
	B41_i[m] = B41[m];
}
W43_i_m_loop: for (m=0; m<512; m++) {
  W43_i_k_loop: for (k=0; k<512; k++) {
  W43_i_i_loop: for (i=0; i<3; i++) {
  W43_i_j_loop: for (j=0; j<3; j++) {
  W43_i[m][k][i][j] = W43[m][k][i][j];
      }
    }
  }
}
B43_i_m_loop: for (m=0; m<512; m++) {
	B43_i[m] = B43[m];
}
W48_i_m_loop: for (m=0; m<10; m++) {
  W48_i_k_loop: for (k=0; k<512; k++) {
  W48_i[m][k] = W48[m][k];
      }
}
B48_i_m_loop: for (m=0; m<10; m++) {
	B48_i[m] = B48[m];
}


    resnet18(I_i, W1_i, B1_i, W3_i, B3_i, W4_i, B4_i, W5_i, B5_i, W8_i, B8_i, W10_i, B10_i, W13_i, B13_i, W15_i, B15_i, W16_i, B16_i, W19_i, B19_i, W21_i, B21_i, W24_i, B24_i, W26_i, B26_i, W27_i, B27_i, W30_i, B30_i, W32_i, B32_i, W35_i, B35_i, W37_i, B37_i, W38_i, B38_i, W41_i, B41_i, W43_i, B43_i, W48_i, B48_i,  O_i);

    	for(m=0; m<10; m++) { O[m] = O_i[m]; }																																																																				  for (m = 0; m<10; m++) { O[m] = O_i[m]; }

}

void resnet18_sw(DATA_T I[3][32][32], DATA_T W1[64][3][3][3], DATA_T B1[64], DATA_T W3[64][64][3][3], DATA_T B3[64], DATA_T W4[64][64][3][3], DATA_T B4[64], DATA_T W5[64][64][1][1], DATA_T B5[64], DATA_T W8[64][64][3][3], DATA_T B8[64], DATA_T W10[64][64][3][3], DATA_T B10[64], DATA_T W13[128][64][3][3], DATA_T B13[128], DATA_T W15[128][128][3][3], DATA_T B15[128], DATA_T W16[128][64][1][1], DATA_T B16[128], DATA_T W19[128][128][3][3], DATA_T B19[128], DATA_T W21[128][128][3][3], DATA_T B21[128], DATA_T W24[256][128][3][3], DATA_T B24[256], DATA_T W26[256][256][3][3], DATA_T B26[256], DATA_T W27[256][128][1][1], DATA_T B27[256], DATA_T W30[256][256][3][3], DATA_T B30[256], DATA_T W32[256][256][3][3], DATA_T B32[256], DATA_T W35[512][256][3][3], DATA_T B35[512], DATA_T W37[512][512][3][3], DATA_T B37[512], DATA_T W38[512][256][1][1], DATA_T B38[512], DATA_T W41[512][512][3][3], DATA_T B41[512], DATA_T W43[512][512][3][3], DATA_T B43[512], DATA_T W48[10][512], DATA_T B48[10], DATA_T O48_SW[10]) {
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


	SW_conv2d_1(I, O1_SW, W1, B1);
	SW_activation_1(O1_SW, O2_SW);
	SW_res0a_branch2a(O2_SW, O3_SW, W3, B3);
	SW_conv2d_2(O3_SW, O4_SW, W4, B4);
	SW_conv2d_3(O2_SW, O5_SW, W5, B5);
	SW_activation_2(O4_SW, O6_SW);
	SW_add_1(O5_SW, O6_SW, O7_SW);
	SW_conv2d_4(O7_SW, O8_SW, W8, B8);
	SW_activation_3(O8_SW, O9_SW);
	SW_conv2d_5(O9_SW, O10_SW, W10, B10);
	SW_activation_4(O10_SW, O11_SW);
	SW_add_2(O7_SW, O11_SW, O12_SW);
	SW_conv2d_6(O12_SW, O13_SW, W13, B13);
	SW_activation_5(O13_SW, O14_SW);
	SW_conv2d_7(O14_SW, O15_SW, W15, B15);
	SW_conv2d_8(O12_SW, O16_SW, W16, B16);
	SW_activation_6(O15_SW, O17_SW);
	SW_add_3(O16_SW, O17_SW, O18_SW);
	SW_conv2d_9(O18_SW, O19_SW, W19, B19);
	SW_activation_7(O19_SW, O20_SW);
	SW_conv2d_10(O20_SW, O21_SW, W21, B21);
	SW_activation_8(O21_SW, O22_SW);
	SW_add_4(O18_SW, O22_SW, O23_SW);
	SW_conv2d_11(O23_SW, O24_SW, W24, B24);
	SW_activation_9(O24_SW, O25_SW);
	SW_conv2d_12(O25_SW, O26_SW, W26, B26);
	SW_conv2d_13(O23_SW, O27_SW, W27, B27);
	SW_activation_10(O26_SW, O28_SW);
	SW_add_5(O27_SW, O28_SW, O29_SW);
	SW_conv2d_14(O29_SW, O30_SW, W30, B30);
	SW_activation_11(O30_SW, O31_SW);
	SW_conv2d_15(O31_SW, O32_SW, W32, B32);
	SW_activation_12(O32_SW, O33_SW);
	SW_add_6(O29_SW, O33_SW, O34_SW);
	SW_conv2d_16(O34_SW, O35_SW, W35, B35);
	SW_activation_13(O35_SW, O36_SW);
	SW_conv2d_17(O36_SW, O37_SW, W37, B37);
	SW_conv2d_18(O34_SW, O38_SW, W38, B38);
	SW_activation_14(O37_SW, O39_SW);
	SW_add_7(O38_SW, O39_SW, O40_SW);
	SW_conv2d_19(O40_SW, O41_SW, W41, B41);
	SW_activation_15(O41_SW, O42_SW);
	SW_conv2d_20(O42_SW, O43_SW, W43, B43);
	SW_activation_16(O43_SW, O44_SW);
	SW_add_8(O40_SW, O44_SW, O45_SW);
	SW_activation_17(O45_SW, O46_SW);
	SW_global_average_pooling2d_1(O46_SW, O47_SW);
	SW_dense_1(O47_SW, O48_SW, W48, B48);

}
