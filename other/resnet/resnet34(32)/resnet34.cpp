#include <iostream>
#include <ap_int.h>
#include <hls_stream.h>

typedef int DATA_T;
typedef ap_uint<256> uint256_t;
typedef ap_uint<512> uint512_t;

void Stream_input(DATA_T I[3][32][32], hls::stream<DATA_T> &I_strm) {
  int k, x, y;

#pragma HLS ARRAY_PARTITION variable=I complete dim=1
	Stream_input_x_loop: for (x=0; x<32; x++) {
	 	Stream_input_y_loop: for (y=0; y<32; y++) {
#pragma HLS PIPELINE
	 		Stream_input_m_loop: for (k=0; k<3; k++) {
	 			I_strm.write(I[k][x][y]);
	    }
	  }
	}
}

void Stream_output(hls::stream<DATA_T> &O_strm, DATA_T O[10]) {
  int x;
#pragma HLS ARRAY_PARTITION variable=O complete dim=1

	Stream_input_x_loop: for (x=0; x<10; x++) {

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
	static DATA_T W13_i[64][64][3][3];
	static DATA_T B13_i[64];
	static DATA_T W15_i[64][64][3][3];
	static DATA_T B15_i[64];
	static DATA_T W18_i[128][64][3][3];
	static DATA_T B18_i[128];
	static DATA_T W20_i[128][128][3][3];
	static DATA_T B20_i[128];
	static DATA_T W21_i[128][64];
	static DATA_T B21_i[128];
	static DATA_T W24_i[128][128][3][3];
	static DATA_T B24_i[128];
	static DATA_T W26_i[128][128][3][3];
	static DATA_T B26_i[128];
	static DATA_T W29_i[128][128][3][3];
	static DATA_T B29_i[128];
	static DATA_T W31_i[128][128][3][3];
	static DATA_T B31_i[128];
	static DATA_T W34_i[128][128][3][3];
	static DATA_T B34_i[128];
	static DATA_T W36_i[128][128][3][3];
	static DATA_T B36_i[128];
	static DATA_T W39_i[256][128][3][3];
	static DATA_T B39_i[256];
	static DATA_T W41_i[256][256][3][3];
	static DATA_T B41_i[256];
	static DATA_T W42_i[256][128];
	static DATA_T B42_i[256];
	static DATA_T W45_i[256][256][3][3];
	static DATA_T B45_i[256];
	static DATA_T W47_i[256][256][3][3];
	static DATA_T B47_i[256];
	static DATA_T W50_i[256][256][3][3];
	static DATA_T B50_i[256];
	static DATA_T W52_i[256][256][3][3];
	static DATA_T B52_i[256];
	static DATA_T W55_i[256][256][3][3];
	static DATA_T B55_i[256];
	static DATA_T W57_i[256][256][3][3];
	static DATA_T B57_i[256];
	static DATA_T W60_i[256][256][3][3];
	static DATA_T B60_i[256];
	static DATA_T W62_i[256][256][3][3];
	static DATA_T B62_i[256];
	static DATA_T W65_i[256][256][3][3];
	static DATA_T B65_i[256];
	static DATA_T W67_i[256][256][3][3];
	static DATA_T B67_i[256];
	static DATA_T W70_i[512][256][3][3];
	static DATA_T B70_i[512];
	static DATA_T W72_i[512][512][3][3];
	static DATA_T B72_i[512];
	static DATA_T W73_i[512][256];
	static DATA_T B73_i[512];
	static DATA_T W76_i[512][512][3][3];
	static DATA_T B76_i[512];
	static DATA_T W78_i[512][512][3][3];
	static DATA_T B78_i[512];
	static DATA_T W81_i[512][512][3][3];
	static DATA_T B81_i[512];
	static DATA_T W83_i[512][512][3][3];
	static DATA_T B83_i[512];
	static DATA_T W88_i[10][512];
	static DATA_T B88_i[10];
	

void HW_conv2d_1(hls::stream<DATA_T> &I_strm, DATA_T W[64][3][3][3],DATA_T B[64], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[3], ofm[64];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[3][3][32];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_1_x_loop: for (x = 0; x<32 + 1; x++) {
		HW_conv2d_1_y_loop: for (y = 0; y<32 + 1 ; y++) {
			HW_conv2d_1_k_loop: for (k = 0; k<3; k++) {
			    if(x < 32 && y < 32){
	                I[k][x % 3][y] = I_strm.read();}
				if (x > 0 && y > 0){
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
					if (k == 3-1) {
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
	int num = 0;
	DATA_T ifm;
	HW_activation_1_x_loop: for (x = 0; x < 32; x++) {
		HW_activation_1_y_loop: for (y = 0; y < 32; y++) {
			HW_activation_1_m_loop: for (m = 0; m < 64; m++) {
				ifm = I_strm.read();
				if (ifm < 0) {
					O1_strm.write(0); O2_strm.write(0);
				}
				else{
					O1_strm.write(ifm); O2_strm.write(ifm);
				}
			}
		}
	}
}

void HW_res0a_branch2a(hls::stream<DATA_T> &I_strm, DATA_T W[64][64][3][3],DATA_T B[64], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;
	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[64], ofm[64];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[64][3][32];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_res0a_branch2a_x_loop: for (x = 0; x<32 + 1 ; x++) {
		HW_res0a_branch2a_y_loop: for (y = 0; y<32 + 1 ; y++) {
			HW_res0a_branch2a_k_loop: for (k = 0; k<64; k++) {
				if(x < 32 && y < 32){
	                I[k][x % 3][y] = I_strm.read();
				}
				if ((x >> 1) && (y >> 1)){
				    if(x % 2 == 0 && y % 2 == 0){
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
						if (x - 2 + 1 < 32 && y - 2 + 2 < 32 ) {
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
						if (k == 64 -1) {
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

void HW_conv2d_2(hls::stream<DATA_T> &I_strm, DATA_T W[64][64][3][3],DATA_T B[64], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[64], ofm[64];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[64][3][16];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_2_x_loop: for (x = 0; x<16 + 1; x++) {
		HW_conv2d_2_y_loop: for (y = 0; y<16 + 1 ; y++) {
			HW_conv2d_2_k_loop: for (k = 0; k<64; k++) {
			    if(x < 16 && y < 16){
	                I[k][x % 3][y] = I_strm.read();}
				if (x > 0 && y > 0){
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
					if (k == 64-1) {
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

void HW_conv2d_3(hls::stream<DATA_T> &I_strm, DATA_T W[64][64], DATA_T B[64], hls::stream<DATA_T> &O_strm ) {
#pragma HLS INLINE
	int m, x, y, k;
	DATA_T ofm[64];
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0
	DATA_T ifm;

	HW_conv2d_3_x_loop: for (x = 0; x < 32; x++) {
		HW_conv2d_3_y_loop: for (y = 0; y < 32; y++) {
			HW_conv2d_3_k_loop: for (k = 0; k < 64; k++) {
				ifm = I_strm.read();
				if(x % 2 == 0 && y % 2 == 0){
					HW_conv2d_3_m_loop: for (m = 0; m < 64; m++) {
	#pragma HLS PIPELINE
						if (k == 0) {
							ofm[m] = B[m];}
						ofm[m] = ofm[m] + ifm * W[m][k];
					}
					if (k == 64-1) {
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
	int num = 0;
	DATA_T ifm;
	HW_activation_2_x_loop: for (x = 0; x < 16; x++) {
		HW_activation_2_y_loop: for (y = 0; y < 16; y++) {
			HW_activation_2_m_loop: for (m = 0; m < 64; m++) {
				ifm = I_strm.read();
				if (ifm < 0) {
					O_strm.write(0);}
				else{
					O_strm.write(ifm);
				}
			}
		}
	}
}

void HW_add_1(hls::stream<DATA_T> &I1_strm, hls::stream<DATA_T> &I2_strm, hls::stream<DATA_T> &O1_strm, hls::stream<DATA_T> &O2_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm1;
	DATA_T ifm2;
	HW_add_1_x_loop_1: for (x = 0; x < 16; x++) {
	    HW_add_1_y_loop_1: for (y = 0; y < 16; y++) {
	    	HW_add_1_m_loop_1: for (m = 0; m < 64; m++) {
				ifm1 = I1_strm.read();
				ifm2 = I2_strm.read();
				O1_strm.write(ifm1 + ifm2);
				O2_strm.write(ifm1 + ifm2);
			}
		}
	}
}

void HW_conv2d_4(hls::stream<DATA_T> &I_strm, DATA_T W[64][64][3][3],DATA_T B[64], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[64], ofm[64];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[64][3][16];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_4_x_loop: for (x = 0; x<16 + 1; x++) {
		HW_conv2d_4_y_loop: for (y = 0; y<16 + 1 ; y++) {
			HW_conv2d_4_k_loop: for (k = 0; k<64; k++) {
			    if(x < 16 && y < 16){
	                I[k][x % 3][y] = I_strm.read();}
				if (x > 0 && y > 0){
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
					if (k == 64-1) {
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
	int num = 0;
	DATA_T ifm;
	HW_activation_3_x_loop: for (x = 0; x < 16; x++) {
		HW_activation_3_y_loop: for (y = 0; y < 16; y++) {
			HW_activation_3_m_loop: for (m = 0; m < 64; m++) {
				ifm = I_strm.read();
				if (ifm < 0) {
					O_strm.write(0);}
				else{
					O_strm.write(ifm);
				}
			}
		}
	}
}

void HW_conv2d_5(hls::stream<DATA_T> &I_strm, DATA_T W[64][64][3][3],DATA_T B[64], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[64], ofm[64];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[64][3][16];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_5_x_loop: for (x = 0; x<16 + 1; x++) {
		HW_conv2d_5_y_loop: for (y = 0; y<16 + 1 ; y++) {
			HW_conv2d_5_k_loop: for (k = 0; k<64; k++) {
			    if(x < 16 && y < 16){
	                I[k][x % 3][y] = I_strm.read();}
				if (x > 0 && y > 0){
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
					if (k == 64-1) {
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
	int num = 0;
	DATA_T ifm;
	HW_activation_4_x_loop: for (x = 0; x < 16; x++) {
		HW_activation_4_y_loop: for (y = 0; y < 16; y++) {
			HW_activation_4_m_loop: for (m = 0; m < 64; m++) {
				ifm = I_strm.read();
				if (ifm < 0) {
					O_strm.write(0);}
				else{
					O_strm.write(ifm);
				}
			}
		}
	}
}

void HW_add_2(hls::stream<DATA_T> &I1_strm, hls::stream<DATA_T> &I2_strm, hls::stream<DATA_T> &O1_strm, hls::stream<DATA_T> &O2_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm1;
	DATA_T ifm2;
	HW_add_2_x_loop_1: for (x = 0; x < 16; x++) {
	    HW_add_2_y_loop_1: for (y = 0; y < 16; y++) {
	    	HW_add_2_m_loop_1: for (m = 0; m < 64; m++) {
				ifm1 = I1_strm.read();
				ifm2 = I2_strm.read();
				O1_strm.write(ifm1 + ifm2);
				O2_strm.write(ifm1 + ifm2);
			}
		}
	}
}

void HW_conv2d_6(hls::stream<DATA_T> &I_strm, DATA_T W[64][64][3][3],DATA_T B[64], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[64], ofm[64];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[64][3][16];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_6_x_loop: for (x = 0; x<16 + 1; x++) {
		HW_conv2d_6_y_loop: for (y = 0; y<16 + 1 ; y++) {
			HW_conv2d_6_k_loop: for (k = 0; k<64; k++) {
			    if(x < 16 && y < 16){
	                I[k][x % 3][y] = I_strm.read();}
				if (x > 0 && y > 0){
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

					HW_conv2d_6_m_loop: for (m = 0; m<64; m++) {
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
					if (k == 64-1) {
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

void HW_activation_5(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	int num = 0;
	DATA_T ifm;
	HW_activation_5_x_loop: for (x = 0; x < 16; x++) {
		HW_activation_5_y_loop: for (y = 0; y < 16; y++) {
			HW_activation_5_m_loop: for (m = 0; m < 64; m++) {
				ifm = I_strm.read();
				if (ifm < 0) {
					O_strm.write(0);}
				else{
					O_strm.write(ifm);
				}
			}
		}
	}
}

void HW_conv2d_7(hls::stream<DATA_T> &I_strm, DATA_T W[64][64][3][3],DATA_T B[64], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[64], ofm[64];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[64][3][16];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_7_x_loop: for (x = 0; x<16 + 1; x++) {
		HW_conv2d_7_y_loop: for (y = 0; y<16 + 1 ; y++) {
			HW_conv2d_7_k_loop: for (k = 0; k<64; k++) {
			    if(x < 16 && y < 16){
	                I[k][x % 3][y] = I_strm.read();}
				if (x > 0 && y > 0){
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

					HW_conv2d_7_m_loop: for (m = 0; m<64; m++) {
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
					if (k == 64-1) {
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

void HW_activation_6(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	int num = 0;
	DATA_T ifm;
	HW_activation_6_x_loop: for (x = 0; x < 16; x++) {
		HW_activation_6_y_loop: for (y = 0; y < 16; y++) {
			HW_activation_6_m_loop: for (m = 0; m < 64; m++) {
				ifm = I_strm.read();
				if (ifm < 0) {
					O_strm.write(0);}
				else{
					O_strm.write(ifm);
				}
			}
		}
	}
}

void HW_add_3(hls::stream<DATA_T> &I1_strm, hls::stream<DATA_T> &I2_strm, hls::stream<DATA_T> &O1_strm, hls::stream<DATA_T> &O2_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm1;
	DATA_T ifm2;
	HW_add_3_x_loop_1: for (x = 0; x < 16; x++) {
	    HW_add_3_y_loop_1: for (y = 0; y < 16; y++) {
	    	HW_add_3_m_loop_1: for (m = 0; m < 64; m++) {
				ifm1 = I1_strm.read();
				ifm2 = I2_strm.read();
				O1_strm.write(ifm1 + ifm2);
				O2_strm.write(ifm1 + ifm2);
			}
		}
	}
}

void HW_conv2d_8(hls::stream<DATA_T> &I_strm, DATA_T W[128][64][3][3],DATA_T B[128], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;
	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[64], ofm[128];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[64][3][16];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_8_x_loop: for (x = 0; x<16 + 1 ; x++) {
		HW_conv2d_8_y_loop: for (y = 0; y<16 + 1 ; y++) {
			HW_conv2d_8_k_loop: for (k = 0; k<64; k++) {
				if(x < 16 && y < 16){
	                I[k][x % 3][y] = I_strm.read();
				}
				if ((x >> 1) && (y >> 1)){
				    if(x % 2 == 0 && y % 2 == 0){
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
						if (x - 2 + 1 < 16 && y - 2 + 2 < 16 ) {
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

						HW_conv2d_8_m_loop: for (m = 0; m<128; m++) {
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
						if (k == 64 -1) {
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

void HW_activation_7(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	int num = 0;
	DATA_T ifm;
	HW_activation_7_x_loop: for (x = 0; x < 8; x++) {
		HW_activation_7_y_loop: for (y = 0; y < 8; y++) {
			HW_activation_7_m_loop: for (m = 0; m < 128; m++) {
				ifm = I_strm.read();
				if (ifm < 0) {
					O_strm.write(0);}
				else{
					O_strm.write(ifm);
				}
			}
		}
	}
}

void HW_conv2d_9(hls::stream<DATA_T> &I_strm, DATA_T W[128][128][3][3],DATA_T B[128], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[128], ofm[128];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[128][3][8];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_9_x_loop: for (x = 0; x<8 + 1; x++) {
		HW_conv2d_9_y_loop: for (y = 0; y<8 + 1 ; y++) {
			HW_conv2d_9_k_loop: for (k = 0; k<128; k++) {
			    if(x < 8 && y < 8){
	                I[k][x % 3][y] = I_strm.read();}
				if (x > 0 && y > 0){
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
					if (k == 128-1) {
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

void HW_conv2d_10(hls::stream<DATA_T> &I_strm, DATA_T W[128][64], DATA_T B[128], hls::stream<DATA_T> &O_strm ) {
#pragma HLS INLINE
	int m, x, y, k;
	DATA_T ofm[128];
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0
	DATA_T ifm;

	HW_conv2d_10_x_loop: for (x = 0; x < 16; x++) {
		HW_conv2d_10_y_loop: for (y = 0; y < 16; y++) {
			HW_conv2d_10_k_loop: for (k = 0; k < 64; k++) {
				ifm = I_strm.read();
				if(x % 2 == 0 && y % 2 == 0){
					HW_conv2d_10_m_loop: for (m = 0; m < 128; m++) {
	#pragma HLS PIPELINE
						if (k == 0) {
							ofm[m] = B[m];}
						ofm[m] = ofm[m] + ifm * W[m][k];
					}
					if (k == 64-1) {
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

void HW_activation_8(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	int num = 0;
	DATA_T ifm;
	HW_activation_8_x_loop: for (x = 0; x < 8; x++) {
		HW_activation_8_y_loop: for (y = 0; y < 8; y++) {
			HW_activation_8_m_loop: for (m = 0; m < 128; m++) {
				ifm = I_strm.read();
				if (ifm < 0) {
					O_strm.write(0);}
				else{
					O_strm.write(ifm);
				}
			}
		}
	}
}

void HW_add_4(hls::stream<DATA_T> &I1_strm, hls::stream<DATA_T> &I2_strm, hls::stream<DATA_T> &O1_strm, hls::stream<DATA_T> &O2_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm1;
	DATA_T ifm2;
	HW_add_4_x_loop_1: for (x = 0; x < 8; x++) {
	    HW_add_4_y_loop_1: for (y = 0; y < 8; y++) {
	    	HW_add_4_m_loop_1: for (m = 0; m < 128; m++) {
				ifm1 = I1_strm.read();
				ifm2 = I2_strm.read();
				O1_strm.write(ifm1 + ifm2);
				O2_strm.write(ifm1 + ifm2);
			}
		}
	}
}

void HW_conv2d_11(hls::stream<DATA_T> &I_strm, DATA_T W[128][128][3][3],DATA_T B[128], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[128], ofm[128];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[128][3][8];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_11_x_loop: for (x = 0; x<8 + 1; x++) {
		HW_conv2d_11_y_loop: for (y = 0; y<8 + 1 ; y++) {
			HW_conv2d_11_k_loop: for (k = 0; k<128; k++) {
			    if(x < 8 && y < 8){
	                I[k][x % 3][y] = I_strm.read();}
				if (x > 0 && y > 0){
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

					HW_conv2d_11_m_loop: for (m = 0; m<128; m++) {
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
					if (k == 128-1) {
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

void HW_activation_9(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	int num = 0;
	DATA_T ifm;
	HW_activation_9_x_loop: for (x = 0; x < 8; x++) {
		HW_activation_9_y_loop: for (y = 0; y < 8; y++) {
			HW_activation_9_m_loop: for (m = 0; m < 128; m++) {
				ifm = I_strm.read();
				if (ifm < 0) {
					O_strm.write(0);}
				else{
					O_strm.write(ifm);
				}
			}
		}
	}
}

void HW_conv2d_12(hls::stream<DATA_T> &I_strm, DATA_T W[128][128][3][3],DATA_T B[128], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[128], ofm[128];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[128][3][8];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_12_x_loop: for (x = 0; x<8 + 1; x++) {
		HW_conv2d_12_y_loop: for (y = 0; y<8 + 1 ; y++) {
			HW_conv2d_12_k_loop: for (k = 0; k<128; k++) {
			    if(x < 8 && y < 8){
	                I[k][x % 3][y] = I_strm.read();}
				if (x > 0 && y > 0){
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

					HW_conv2d_12_m_loop: for (m = 0; m<128; m++) {
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
					if (k == 128-1) {
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

void HW_activation_10(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	int num = 0;
	DATA_T ifm;
	HW_activation_10_x_loop: for (x = 0; x < 8; x++) {
		HW_activation_10_y_loop: for (y = 0; y < 8; y++) {
			HW_activation_10_m_loop: for (m = 0; m < 128; m++) {
				ifm = I_strm.read();
				if (ifm < 0) {
					O_strm.write(0);}
				else{
					O_strm.write(ifm);
				}
			}
		}
	}
}

void HW_add_5(hls::stream<DATA_T> &I1_strm, hls::stream<DATA_T> &I2_strm, hls::stream<DATA_T> &O1_strm, hls::stream<DATA_T> &O2_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm1;
	DATA_T ifm2;
	HW_add_5_x_loop_1: for (x = 0; x < 8; x++) {
	    HW_add_5_y_loop_1: for (y = 0; y < 8; y++) {
	    	HW_add_5_m_loop_1: for (m = 0; m < 128; m++) {
				ifm1 = I1_strm.read();
				ifm2 = I2_strm.read();
				O1_strm.write(ifm1 + ifm2);
				O2_strm.write(ifm1 + ifm2);
			}
		}
	}
}

void HW_conv2d_13(hls::stream<DATA_T> &I_strm, DATA_T W[128][128][3][3],DATA_T B[128], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[128], ofm[128];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[128][3][8];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_13_x_loop: for (x = 0; x<8 + 1; x++) {
		HW_conv2d_13_y_loop: for (y = 0; y<8 + 1 ; y++) {
			HW_conv2d_13_k_loop: for (k = 0; k<128; k++) {
			    if(x < 8 && y < 8){
	                I[k][x % 3][y] = I_strm.read();}
				if (x > 0 && y > 0){
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

					HW_conv2d_13_m_loop: for (m = 0; m<128; m++) {
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
					if (k == 128-1) {
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

void HW_activation_11(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	int num = 0;
	DATA_T ifm;
	HW_activation_11_x_loop: for (x = 0; x < 8; x++) {
		HW_activation_11_y_loop: for (y = 0; y < 8; y++) {
			HW_activation_11_m_loop: for (m = 0; m < 128; m++) {
				ifm = I_strm.read();
				if (ifm < 0) {
					O_strm.write(0);}
				else{
					O_strm.write(ifm);
				}
			}
		}
	}
}

void HW_conv2d_14(hls::stream<DATA_T> &I_strm, DATA_T W[128][128][3][3],DATA_T B[128], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[128], ofm[128];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[128][3][8];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_14_x_loop: for (x = 0; x<8 + 1; x++) {
		HW_conv2d_14_y_loop: for (y = 0; y<8 + 1 ; y++) {
			HW_conv2d_14_k_loop: for (k = 0; k<128; k++) {
			    if(x < 8 && y < 8){
	                I[k][x % 3][y] = I_strm.read();}
				if (x > 0 && y > 0){
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

					HW_conv2d_14_m_loop: for (m = 0; m<128; m++) {
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
					if (k == 128-1) {
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

void HW_activation_12(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	int num = 0;
	DATA_T ifm;
	HW_activation_12_x_loop: for (x = 0; x < 8; x++) {
		HW_activation_12_y_loop: for (y = 0; y < 8; y++) {
			HW_activation_12_m_loop: for (m = 0; m < 128; m++) {
				ifm = I_strm.read();
				if (ifm < 0) {
					O_strm.write(0);}
				else{
					O_strm.write(ifm);
				}
			}
		}
	}
}

void HW_add_6(hls::stream<DATA_T> &I1_strm, hls::stream<DATA_T> &I2_strm, hls::stream<DATA_T> &O1_strm, hls::stream<DATA_T> &O2_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm1;
	DATA_T ifm2;
	HW_add_6_x_loop_1: for (x = 0; x < 8; x++) {
	    HW_add_6_y_loop_1: for (y = 0; y < 8; y++) {
	    	HW_add_6_m_loop_1: for (m = 0; m < 128; m++) {
				ifm1 = I1_strm.read();
				ifm2 = I2_strm.read();
				O1_strm.write(ifm1 + ifm2);
				O2_strm.write(ifm1 + ifm2);
			}
		}
	}
}

void HW_conv2d_15(hls::stream<DATA_T> &I_strm, DATA_T W[128][128][3][3],DATA_T B[128], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[128], ofm[128];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[128][3][8];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_15_x_loop: for (x = 0; x<8 + 1; x++) {
		HW_conv2d_15_y_loop: for (y = 0; y<8 + 1 ; y++) {
			HW_conv2d_15_k_loop: for (k = 0; k<128; k++) {
			    if(x < 8 && y < 8){
	                I[k][x % 3][y] = I_strm.read();}
				if (x > 0 && y > 0){
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

					HW_conv2d_15_m_loop: for (m = 0; m<128; m++) {
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
					if (k == 128-1) {
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

void HW_activation_13(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	int num = 0;
	DATA_T ifm;
	HW_activation_13_x_loop: for (x = 0; x < 8; x++) {
		HW_activation_13_y_loop: for (y = 0; y < 8; y++) {
			HW_activation_13_m_loop: for (m = 0; m < 128; m++) {
				ifm = I_strm.read();
				if (ifm < 0) {
					O_strm.write(0);}
				else{
					O_strm.write(ifm);
				}
			}
		}
	}
}

void HW_conv2d_16(hls::stream<DATA_T> &I_strm, DATA_T W[128][128][3][3],DATA_T B[128], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[128], ofm[128];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[128][3][8];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_16_x_loop: for (x = 0; x<8 + 1; x++) {
		HW_conv2d_16_y_loop: for (y = 0; y<8 + 1 ; y++) {
			HW_conv2d_16_k_loop: for (k = 0; k<128; k++) {
			    if(x < 8 && y < 8){
	                I[k][x % 3][y] = I_strm.read();}
				if (x > 0 && y > 0){
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

					HW_conv2d_16_m_loop: for (m = 0; m<128; m++) {
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
					if (k == 128-1) {
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

void HW_activation_14(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	int num = 0;
	DATA_T ifm;
	HW_activation_14_x_loop: for (x = 0; x < 8; x++) {
		HW_activation_14_y_loop: for (y = 0; y < 8; y++) {
			HW_activation_14_m_loop: for (m = 0; m < 128; m++) {
				ifm = I_strm.read();
				if (ifm < 0) {
					O_strm.write(0);}
				else{
					O_strm.write(ifm);
				}
			}
		}
	}
}

void HW_add_7(hls::stream<DATA_T> &I1_strm, hls::stream<DATA_T> &I2_strm, hls::stream<DATA_T> &O1_strm, hls::stream<DATA_T> &O2_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm1;
	DATA_T ifm2;
	HW_add_7_x_loop_1: for (x = 0; x < 8; x++) {
	    HW_add_7_y_loop_1: for (y = 0; y < 8; y++) {
	    	HW_add_7_m_loop_1: for (m = 0; m < 128; m++) {
				ifm1 = I1_strm.read();
				ifm2 = I2_strm.read();
				O1_strm.write(ifm1 + ifm2);
				O2_strm.write(ifm1 + ifm2);
			}
		}
	}
}

void HW_conv2d_17(hls::stream<DATA_T> &I_strm, DATA_T W[256][128][3][3],DATA_T B[256], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;
	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[128], ofm[256];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[128][3][8];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_17_x_loop: for (x = 0; x<8 + 1 ; x++) {
		HW_conv2d_17_y_loop: for (y = 0; y<8 + 1 ; y++) {
			HW_conv2d_17_k_loop: for (k = 0; k<128; k++) {
				if(x < 8 && y < 8){
	                I[k][x % 3][y] = I_strm.read();
				}
				if ((x >> 1) && (y >> 1)){
				    if(x % 2 == 0 && y % 2 == 0){
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
						if (x - 2 + 1 < 8 && y - 2 + 2 < 8 ) {
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

						HW_conv2d_17_m_loop: for (m = 0; m<256; m++) {
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
						if (k == 128 -1) {
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

void HW_activation_15(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	int num = 0;
	DATA_T ifm;
	HW_activation_15_x_loop: for (x = 0; x < 4; x++) {
		HW_activation_15_y_loop: for (y = 0; y < 4; y++) {
			HW_activation_15_m_loop: for (m = 0; m < 256; m++) {
				ifm = I_strm.read();
				if (ifm < 0) {
					O_strm.write(0);}
				else{
					O_strm.write(ifm);
				}
			}
		}
	}
}

void HW_conv2d_18(hls::stream<DATA_T> &I_strm, DATA_T W[256][256][3][3],DATA_T B[256], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[256], ofm[256];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[256][3][4];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_18_x_loop: for (x = 0; x<4 + 1; x++) {
		HW_conv2d_18_y_loop: for (y = 0; y<4 + 1 ; y++) {
			HW_conv2d_18_k_loop: for (k = 0; k<256; k++) {
			    if(x < 4 && y < 4){
	                I[k][x % 3][y] = I_strm.read();}
				if (x > 0 && y > 0){
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

					HW_conv2d_18_m_loop: for (m = 0; m<256; m++) {
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
					if (k == 256-1) {
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

void HW_conv2d_19(hls::stream<DATA_T> &I_strm, DATA_T W[256][128], DATA_T B[256], hls::stream<DATA_T> &O_strm ) {
#pragma HLS INLINE
	int m, x, y, k;
	DATA_T ofm[256];
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0
	DATA_T ifm;

	HW_conv2d_19_x_loop: for (x = 0; x < 8; x++) {
		HW_conv2d_19_y_loop: for (y = 0; y < 8; y++) {
			HW_conv2d_19_k_loop: for (k = 0; k < 128; k++) {
				ifm = I_strm.read();
				if(x % 2 == 0 && y % 2 == 0){
					HW_conv2d_19_m_loop: for (m = 0; m < 256; m++) {
	#pragma HLS PIPELINE
						if (k == 0) {
							ofm[m] = B[m];}
						ofm[m] = ofm[m] + ifm * W[m][k];
					}
					if (k == 128-1) {
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

void HW_activation_16(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	int num = 0;
	DATA_T ifm;
	HW_activation_16_x_loop: for (x = 0; x < 4; x++) {
		HW_activation_16_y_loop: for (y = 0; y < 4; y++) {
			HW_activation_16_m_loop: for (m = 0; m < 256; m++) {
				ifm = I_strm.read();
				if (ifm < 0) {
					O_strm.write(0);}
				else{
					O_strm.write(ifm);
				}
			}
		}
	}
}

void HW_add_8(hls::stream<DATA_T> &I1_strm, hls::stream<DATA_T> &I2_strm, hls::stream<DATA_T> &O1_strm, hls::stream<DATA_T> &O2_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm1;
	DATA_T ifm2;
	HW_add_8_x_loop_1: for (x = 0; x < 4; x++) {
	    HW_add_8_y_loop_1: for (y = 0; y < 4; y++) {
	    	HW_add_8_m_loop_1: for (m = 0; m < 256; m++) {
				ifm1 = I1_strm.read();
				ifm2 = I2_strm.read();
				O1_strm.write(ifm1 + ifm2);
				O2_strm.write(ifm1 + ifm2);
			}
		}
	}
}

void HW_conv2d_20(hls::stream<DATA_T> &I_strm, DATA_T W[256][256][3][3],DATA_T B[256], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[256], ofm[256];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[256][3][4];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_20_x_loop: for (x = 0; x<4 + 1; x++) {
		HW_conv2d_20_y_loop: for (y = 0; y<4 + 1 ; y++) {
			HW_conv2d_20_k_loop: for (k = 0; k<256; k++) {
			    if(x < 4 && y < 4){
	                I[k][x % 3][y] = I_strm.read();}
				if (x > 0 && y > 0){
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

					HW_conv2d_20_m_loop: for (m = 0; m<256; m++) {
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
					if (k == 256-1) {
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

void HW_activation_17(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	int num = 0;
	DATA_T ifm;
	HW_activation_17_x_loop: for (x = 0; x < 4; x++) {
		HW_activation_17_y_loop: for (y = 0; y < 4; y++) {
			HW_activation_17_m_loop: for (m = 0; m < 256; m++) {
				ifm = I_strm.read();
				if (ifm < 0) {
					O_strm.write(0);}
				else{
					O_strm.write(ifm);
				}
			}
		}
	}
}

void HW_conv2d_21(hls::stream<DATA_T> &I_strm, DATA_T W[256][256][3][3],DATA_T B[256], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[256], ofm[256];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[256][3][4];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_21_x_loop: for (x = 0; x<4 + 1; x++) {
		HW_conv2d_21_y_loop: for (y = 0; y<4 + 1 ; y++) {
			HW_conv2d_21_k_loop: for (k = 0; k<256; k++) {
			    if(x < 4 && y < 4){
	                I[k][x % 3][y] = I_strm.read();}
				if (x > 0 && y > 0){
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

					HW_conv2d_21_m_loop: for (m = 0; m<256; m++) {
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
					if (k == 256-1) {
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

void HW_activation_18(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	int num = 0;
	DATA_T ifm;
	HW_activation_18_x_loop: for (x = 0; x < 4; x++) {
		HW_activation_18_y_loop: for (y = 0; y < 4; y++) {
			HW_activation_18_m_loop: for (m = 0; m < 256; m++) {
				ifm = I_strm.read();
				if (ifm < 0) {
					O_strm.write(0);}
				else{
					O_strm.write(ifm);
				}
			}
		}
	}
}

void HW_add_9(hls::stream<DATA_T> &I1_strm, hls::stream<DATA_T> &I2_strm, hls::stream<DATA_T> &O1_strm, hls::stream<DATA_T> &O2_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm1;
	DATA_T ifm2;
	HW_add_9_x_loop_1: for (x = 0; x < 4; x++) {
	    HW_add_9_y_loop_1: for (y = 0; y < 4; y++) {
	    	HW_add_9_m_loop_1: for (m = 0; m < 256; m++) {
				ifm1 = I1_strm.read();
				ifm2 = I2_strm.read();
				O1_strm.write(ifm1 + ifm2);
				O2_strm.write(ifm1 + ifm2);
			}
		}
	}
}

void HW_conv2d_22(hls::stream<DATA_T> &I_strm, DATA_T W[256][256][3][3],DATA_T B[256], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[256], ofm[256];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[256][3][4];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_22_x_loop: for (x = 0; x<4 + 1; x++) {
		HW_conv2d_22_y_loop: for (y = 0; y<4 + 1 ; y++) {
			HW_conv2d_22_k_loop: for (k = 0; k<256; k++) {
			    if(x < 4 && y < 4){
	                I[k][x % 3][y] = I_strm.read();}
				if (x > 0 && y > 0){
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

					HW_conv2d_22_m_loop: for (m = 0; m<256; m++) {
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
					if (k == 256-1) {
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

void HW_activation_19(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	int num = 0;
	DATA_T ifm;
	HW_activation_19_x_loop: for (x = 0; x < 4; x++) {
		HW_activation_19_y_loop: for (y = 0; y < 4; y++) {
			HW_activation_19_m_loop: for (m = 0; m < 256; m++) {
				ifm = I_strm.read();
				if (ifm < 0) {
					O_strm.write(0);}
				else{
					O_strm.write(ifm);
				}
			}
		}
	}
}

void HW_conv2d_23(hls::stream<DATA_T> &I_strm, DATA_T W[256][256][3][3],DATA_T B[256], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[256], ofm[256];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[256][3][4];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_23_x_loop: for (x = 0; x<4 + 1; x++) {
		HW_conv2d_23_y_loop: for (y = 0; y<4 + 1 ; y++) {
			HW_conv2d_23_k_loop: for (k = 0; k<256; k++) {
			    if(x < 4 && y < 4){
	                I[k][x % 3][y] = I_strm.read();}
				if (x > 0 && y > 0){
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

					HW_conv2d_23_m_loop: for (m = 0; m<256; m++) {
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
					if (k == 256-1) {
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

void HW_activation_20(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	int num = 0;
	DATA_T ifm;
	HW_activation_20_x_loop: for (x = 0; x < 4; x++) {
		HW_activation_20_y_loop: for (y = 0; y < 4; y++) {
			HW_activation_20_m_loop: for (m = 0; m < 256; m++) {
				ifm = I_strm.read();
				if (ifm < 0) {
					O_strm.write(0);}
				else{
					O_strm.write(ifm);
				}
			}
		}
	}
}

void HW_add_10(hls::stream<DATA_T> &I1_strm, hls::stream<DATA_T> &I2_strm, hls::stream<DATA_T> &O1_strm, hls::stream<DATA_T> &O2_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm1;
	DATA_T ifm2;
	HW_add_10_x_loop_1: for (x = 0; x < 4; x++) {
	    HW_add_10_y_loop_1: for (y = 0; y < 4; y++) {
	    	HW_add_10_m_loop_1: for (m = 0; m < 256; m++) {
				ifm1 = I1_strm.read();
				ifm2 = I2_strm.read();
				O1_strm.write(ifm1 + ifm2);
				O2_strm.write(ifm1 + ifm2);
			}
		}
	}
}

void HW_conv2d_24(hls::stream<DATA_T> &I_strm, DATA_T W[256][256][3][3],DATA_T B[256], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[256], ofm[256];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[256][3][4];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_24_x_loop: for (x = 0; x<4 + 1; x++) {
		HW_conv2d_24_y_loop: for (y = 0; y<4 + 1 ; y++) {
			HW_conv2d_24_k_loop: for (k = 0; k<256; k++) {
			    if(x < 4 && y < 4){
	                I[k][x % 3][y] = I_strm.read();}
				if (x > 0 && y > 0){
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

					HW_conv2d_24_m_loop: for (m = 0; m<256; m++) {
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
					if (k == 256-1) {
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

void HW_activation_21(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	int num = 0;
	DATA_T ifm;
	HW_activation_21_x_loop: for (x = 0; x < 4; x++) {
		HW_activation_21_y_loop: for (y = 0; y < 4; y++) {
			HW_activation_21_m_loop: for (m = 0; m < 256; m++) {
				ifm = I_strm.read();
				if (ifm < 0) {
					O_strm.write(0);}
				else{
					O_strm.write(ifm);
				}
			}
		}
	}
}

void HW_conv2d_25(hls::stream<DATA_T> &I_strm, DATA_T W[256][256][3][3],DATA_T B[256], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[256], ofm[256];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[256][3][4];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_25_x_loop: for (x = 0; x<4 + 1; x++) {
		HW_conv2d_25_y_loop: for (y = 0; y<4 + 1 ; y++) {
			HW_conv2d_25_k_loop: for (k = 0; k<256; k++) {
			    if(x < 4 && y < 4){
	                I[k][x % 3][y] = I_strm.read();}
				if (x > 0 && y > 0){
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

					HW_conv2d_25_m_loop: for (m = 0; m<256; m++) {
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
					if (k == 256-1) {
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

void HW_activation_22(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	int num = 0;
	DATA_T ifm;
	HW_activation_22_x_loop: for (x = 0; x < 4; x++) {
		HW_activation_22_y_loop: for (y = 0; y < 4; y++) {
			HW_activation_22_m_loop: for (m = 0; m < 256; m++) {
				ifm = I_strm.read();
				if (ifm < 0) {
					O_strm.write(0);}
				else{
					O_strm.write(ifm);
				}
			}
		}
	}
}

void HW_add_11(hls::stream<DATA_T> &I1_strm, hls::stream<DATA_T> &I2_strm, hls::stream<DATA_T> &O1_strm, hls::stream<DATA_T> &O2_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm1;
	DATA_T ifm2;
	HW_add_11_x_loop_1: for (x = 0; x < 4; x++) {
	    HW_add_11_y_loop_1: for (y = 0; y < 4; y++) {
	    	HW_add_11_m_loop_1: for (m = 0; m < 256; m++) {
				ifm1 = I1_strm.read();
				ifm2 = I2_strm.read();
				O1_strm.write(ifm1 + ifm2);
				O2_strm.write(ifm1 + ifm2);
			}
		}
	}
}

void HW_conv2d_26(hls::stream<DATA_T> &I_strm, DATA_T W[256][256][3][3],DATA_T B[256], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[256], ofm[256];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[256][3][4];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_26_x_loop: for (x = 0; x<4 + 1; x++) {
		HW_conv2d_26_y_loop: for (y = 0; y<4 + 1 ; y++) {
			HW_conv2d_26_k_loop: for (k = 0; k<256; k++) {
			    if(x < 4 && y < 4){
	                I[k][x % 3][y] = I_strm.read();}
				if (x > 0 && y > 0){
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

					HW_conv2d_26_m_loop: for (m = 0; m<256; m++) {
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
					if (k == 256-1) {
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

void HW_activation_23(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	int num = 0;
	DATA_T ifm;
	HW_activation_23_x_loop: for (x = 0; x < 4; x++) {
		HW_activation_23_y_loop: for (y = 0; y < 4; y++) {
			HW_activation_23_m_loop: for (m = 0; m < 256; m++) {
				ifm = I_strm.read();
				if (ifm < 0) {
					O_strm.write(0);}
				else{
					O_strm.write(ifm);
				}
			}
		}
	}
}

void HW_conv2d_27(hls::stream<DATA_T> &I_strm, DATA_T W[256][256][3][3],DATA_T B[256], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[256], ofm[256];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[256][3][4];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_27_x_loop: for (x = 0; x<4 + 1; x++) {
		HW_conv2d_27_y_loop: for (y = 0; y<4 + 1 ; y++) {
			HW_conv2d_27_k_loop: for (k = 0; k<256; k++) {
			    if(x < 4 && y < 4){
	                I[k][x % 3][y] = I_strm.read();}
				if (x > 0 && y > 0){
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

					HW_conv2d_27_m_loop: for (m = 0; m<256; m++) {
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
					if (k == 256-1) {
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

void HW_activation_24(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	int num = 0;
	DATA_T ifm;
	HW_activation_24_x_loop: for (x = 0; x < 4; x++) {
		HW_activation_24_y_loop: for (y = 0; y < 4; y++) {
			HW_activation_24_m_loop: for (m = 0; m < 256; m++) {
				ifm = I_strm.read();
				if (ifm < 0) {
					O_strm.write(0);}
				else{
					O_strm.write(ifm);
				}
			}
		}
	}
}

void HW_add_12(hls::stream<DATA_T> &I1_strm, hls::stream<DATA_T> &I2_strm, hls::stream<DATA_T> &O1_strm, hls::stream<DATA_T> &O2_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm1;
	DATA_T ifm2;
	HW_add_12_x_loop_1: for (x = 0; x < 4; x++) {
	    HW_add_12_y_loop_1: for (y = 0; y < 4; y++) {
	    	HW_add_12_m_loop_1: for (m = 0; m < 256; m++) {
				ifm1 = I1_strm.read();
				ifm2 = I2_strm.read();
				O1_strm.write(ifm1 + ifm2);
				O2_strm.write(ifm1 + ifm2);
			}
		}
	}
}

void HW_conv2d_28(hls::stream<DATA_T> &I_strm, DATA_T W[256][256][3][3],DATA_T B[256], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[256], ofm[256];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[256][3][4];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_28_x_loop: for (x = 0; x<4 + 1; x++) {
		HW_conv2d_28_y_loop: for (y = 0; y<4 + 1 ; y++) {
			HW_conv2d_28_k_loop: for (k = 0; k<256; k++) {
			    if(x < 4 && y < 4){
	                I[k][x % 3][y] = I_strm.read();}
				if (x > 0 && y > 0){
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

					HW_conv2d_28_m_loop: for (m = 0; m<256; m++) {
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
					if (k == 256-1) {
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

void HW_activation_25(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	int num = 0;
	DATA_T ifm;
	HW_activation_25_x_loop: for (x = 0; x < 4; x++) {
		HW_activation_25_y_loop: for (y = 0; y < 4; y++) {
			HW_activation_25_m_loop: for (m = 0; m < 256; m++) {
				ifm = I_strm.read();
				if (ifm < 0) {
					O_strm.write(0);}
				else{
					O_strm.write(ifm);
				}
			}
		}
	}
}

void HW_conv2d_29(hls::stream<DATA_T> &I_strm, DATA_T W[256][256][3][3],DATA_T B[256], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[256], ofm[256];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[256][3][4];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_29_x_loop: for (x = 0; x<4 + 1; x++) {
		HW_conv2d_29_y_loop: for (y = 0; y<4 + 1 ; y++) {
			HW_conv2d_29_k_loop: for (k = 0; k<256; k++) {
			    if(x < 4 && y < 4){
	                I[k][x % 3][y] = I_strm.read();}
				if (x > 0 && y > 0){
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

					HW_conv2d_29_m_loop: for (m = 0; m<256; m++) {
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
					if (k == 256-1) {
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

void HW_activation_26(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	int num = 0;
	DATA_T ifm;
	HW_activation_26_x_loop: for (x = 0; x < 4; x++) {
		HW_activation_26_y_loop: for (y = 0; y < 4; y++) {
			HW_activation_26_m_loop: for (m = 0; m < 256; m++) {
				ifm = I_strm.read();
				if (ifm < 0) {
					O_strm.write(0);}
				else{
					O_strm.write(ifm);
				}
			}
		}
	}
}

void HW_add_13(hls::stream<DATA_T> &I1_strm, hls::stream<DATA_T> &I2_strm, hls::stream<DATA_T> &O1_strm, hls::stream<DATA_T> &O2_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm1;
	DATA_T ifm2;
	HW_add_13_x_loop_1: for (x = 0; x < 4; x++) {
	    HW_add_13_y_loop_1: for (y = 0; y < 4; y++) {
	    	HW_add_13_m_loop_1: for (m = 0; m < 256; m++) {
				ifm1 = I1_strm.read();
				ifm2 = I2_strm.read();
				O1_strm.write(ifm1 + ifm2);
				O2_strm.write(ifm1 + ifm2);
			}
		}
	}
}

void HW_conv2d_30(hls::stream<DATA_T> &I_strm, DATA_T W[512][256][3][3],DATA_T B[512], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;
	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[256], ofm[512];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[256][3][4];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_30_x_loop: for (x = 0; x<4 + 1 ; x++) {
		HW_conv2d_30_y_loop: for (y = 0; y<4 + 1 ; y++) {
			HW_conv2d_30_k_loop: for (k = 0; k<256; k++) {
				if(x < 4 && y < 4){
	                I[k][x % 3][y] = I_strm.read();
				}
				if ((x >> 1) && (y >> 1)){
				    if(x % 2 == 0 && y % 2 == 0){
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
						if (x - 2 + 1 < 4 && y - 2 + 2 < 4 ) {
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

						HW_conv2d_30_m_loop: for (m = 0; m<512; m++) {
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
						if (k == 256 -1) {
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

void HW_activation_27(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	int num = 0;
	DATA_T ifm;
	HW_activation_27_x_loop: for (x = 0; x < 2; x++) {
		HW_activation_27_y_loop: for (y = 0; y < 2; y++) {
			HW_activation_27_m_loop: for (m = 0; m < 512; m++) {
				ifm = I_strm.read();
				if (ifm < 0) {
					O_strm.write(0);}
				else{
					O_strm.write(ifm);
				}
			}
		}
	}
}

void HW_conv2d_31(hls::stream<DATA_T> &I_strm, DATA_T W[512][512][3][3],DATA_T B[512], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[512], ofm[512];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[512][3][2];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_31_x_loop: for (x = 0; x<2 + 1; x++) {
		HW_conv2d_31_y_loop: for (y = 0; y<2 + 1 ; y++) {
			HW_conv2d_31_k_loop: for (k = 0; k<512; k++) {
			    if(x < 2 && y < 2){
	                I[k][x % 3][y] = I_strm.read();}
				if (x > 0 && y > 0){
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

					HW_conv2d_31_m_loop: for (m = 0; m<512; m++) {
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
					if (k == 512-1) {
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

void HW_conv2d_32(hls::stream<DATA_T> &I_strm, DATA_T W[512][256], DATA_T B[512], hls::stream<DATA_T> &O_strm ) {
#pragma HLS INLINE
	int m, x, y, k;
	DATA_T ofm[512];
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0
	DATA_T ifm;

	HW_conv2d_32_x_loop: for (x = 0; x < 4; x++) {
		HW_conv2d_32_y_loop: for (y = 0; y < 4; y++) {
			HW_conv2d_32_k_loop: for (k = 0; k < 256; k++) {
				ifm = I_strm.read();
				if(x % 2 == 0 && y % 2 == 0){
					HW_conv2d_32_m_loop: for (m = 0; m < 512; m++) {
	#pragma HLS PIPELINE
						if (k == 0) {
							ofm[m] = B[m];}
						ofm[m] = ofm[m] + ifm * W[m][k];
					}
					if (k == 256-1) {
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

void HW_activation_28(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	int num = 0;
	DATA_T ifm;
	HW_activation_28_x_loop: for (x = 0; x < 2; x++) {
		HW_activation_28_y_loop: for (y = 0; y < 2; y++) {
			HW_activation_28_m_loop: for (m = 0; m < 512; m++) {
				ifm = I_strm.read();
				if (ifm < 0) {
					O_strm.write(0);}
				else{
					O_strm.write(ifm);
				}
			}
		}
	}
}

void HW_add_14(hls::stream<DATA_T> &I1_strm, hls::stream<DATA_T> &I2_strm, hls::stream<DATA_T> &O1_strm, hls::stream<DATA_T> &O2_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm1;
	DATA_T ifm2;
	HW_add_14_x_loop_1: for (x = 0; x < 2; x++) {
	    HW_add_14_y_loop_1: for (y = 0; y < 2; y++) {
	    	HW_add_14_m_loop_1: for (m = 0; m < 512; m++) {
				ifm1 = I1_strm.read();
				ifm2 = I2_strm.read();
				O1_strm.write(ifm1 + ifm2);
				O2_strm.write(ifm1 + ifm2);
			}
		}
	}
}

void HW_conv2d_33(hls::stream<DATA_T> &I_strm, DATA_T W[512][512][3][3],DATA_T B[512], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[512], ofm[512];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[512][3][2];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_33_x_loop: for (x = 0; x<2 + 1; x++) {
		HW_conv2d_33_y_loop: for (y = 0; y<2 + 1 ; y++) {
			HW_conv2d_33_k_loop: for (k = 0; k<512; k++) {
			    if(x < 2 && y < 2){
	                I[k][x % 3][y] = I_strm.read();}
				if (x > 0 && y > 0){
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

					HW_conv2d_33_m_loop: for (m = 0; m<512; m++) {
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
					if (k == 512-1) {
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

void HW_activation_29(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	int num = 0;
	DATA_T ifm;
	HW_activation_29_x_loop: for (x = 0; x < 2; x++) {
		HW_activation_29_y_loop: for (y = 0; y < 2; y++) {
			HW_activation_29_m_loop: for (m = 0; m < 512; m++) {
				ifm = I_strm.read();
				if (ifm < 0) {
					O_strm.write(0);}
				else{
					O_strm.write(ifm);
				}
			}
		}
	}
}

void HW_conv2d_34(hls::stream<DATA_T> &I_strm, DATA_T W[512][512][3][3],DATA_T B[512], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[512], ofm[512];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[512][3][2];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_34_x_loop: for (x = 0; x<2 + 1; x++) {
		HW_conv2d_34_y_loop: for (y = 0; y<2 + 1 ; y++) {
			HW_conv2d_34_k_loop: for (k = 0; k<512; k++) {
			    if(x < 2 && y < 2){
	                I[k][x % 3][y] = I_strm.read();}
				if (x > 0 && y > 0){
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

					HW_conv2d_34_m_loop: for (m = 0; m<512; m++) {
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
					if (k == 512-1) {
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

void HW_activation_30(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	int num = 0;
	DATA_T ifm;
	HW_activation_30_x_loop: for (x = 0; x < 2; x++) {
		HW_activation_30_y_loop: for (y = 0; y < 2; y++) {
			HW_activation_30_m_loop: for (m = 0; m < 512; m++) {
				ifm = I_strm.read();
				if (ifm < 0) {
					O_strm.write(0);}
				else{
					O_strm.write(ifm);
				}
			}
		}
	}
}

void HW_add_15(hls::stream<DATA_T> &I1_strm, hls::stream<DATA_T> &I2_strm, hls::stream<DATA_T> &O1_strm, hls::stream<DATA_T> &O2_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm1;
	DATA_T ifm2;
	HW_add_15_x_loop_1: for (x = 0; x < 2; x++) {
	    HW_add_15_y_loop_1: for (y = 0; y < 2; y++) {
	    	HW_add_15_m_loop_1: for (m = 0; m < 512; m++) {
				ifm1 = I1_strm.read();
				ifm2 = I2_strm.read();
				O1_strm.write(ifm1 + ifm2);
				O2_strm.write(ifm1 + ifm2);
			}
		}
	}
}

void HW_conv2d_35(hls::stream<DATA_T> &I_strm, DATA_T W[512][512][3][3],DATA_T B[512], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[512], ofm[512];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[512][3][2];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_35_x_loop: for (x = 0; x<2 + 1; x++) {
		HW_conv2d_35_y_loop: for (y = 0; y<2 + 1 ; y++) {
			HW_conv2d_35_k_loop: for (k = 0; k<512; k++) {
			    if(x < 2 && y < 2){
	                I[k][x % 3][y] = I_strm.read();}
				if (x > 0 && y > 0){
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

					HW_conv2d_35_m_loop: for (m = 0; m<512; m++) {
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
					if (k == 512-1) {
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

void HW_activation_31(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	int num = 0;
	DATA_T ifm;
	HW_activation_31_x_loop: for (x = 0; x < 2; x++) {
		HW_activation_31_y_loop: for (y = 0; y < 2; y++) {
			HW_activation_31_m_loop: for (m = 0; m < 512; m++) {
				ifm = I_strm.read();
				if (ifm < 0) {
					O_strm.write(0);}
				else{
					O_strm.write(ifm);
				}
			}
		}
	}
}

void HW_conv2d_36(hls::stream<DATA_T> &I_strm, DATA_T W[512][512][3][3],DATA_T B[512], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[512], ofm[512];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[512][3][2];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv2d_36_x_loop: for (x = 0; x<2 + 1; x++) {
		HW_conv2d_36_y_loop: for (y = 0; y<2 + 1 ; y++) {
			HW_conv2d_36_k_loop: for (k = 0; k<512; k++) {
			    if(x < 2 && y < 2){
	                I[k][x % 3][y] = I_strm.read();}
				if (x > 0 && y > 0){
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

					HW_conv2d_36_m_loop: for (m = 0; m<512; m++) {
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
					if (k == 512-1) {
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

void HW_activation_32(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	int num = 0;
	DATA_T ifm;
	HW_activation_32_x_loop: for (x = 0; x < 2; x++) {
		HW_activation_32_y_loop: for (y = 0; y < 2; y++) {
			HW_activation_32_m_loop: for (m = 0; m < 512; m++) {
				ifm = I_strm.read();
				if (ifm < 0) {
					O_strm.write(0);}
				else{
					O_strm.write(ifm);
				}
			}
		}
	}
}

void HW_add_16(hls::stream<DATA_T> &I1_strm, hls::stream<DATA_T> &I2_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm1;
	DATA_T ifm2;
	HW_add_16_x_loop_1: for (x = 0; x < 2; x++) {
	    HW_add_16_y_loop_1: for (y = 0; y < 2; y++) {
	    	HW_add_16_m_loop_1: for (m = 0; m < 512; m++) {
				ifm1 = I1_strm.read();
				ifm2 = I2_strm.read();
				O_strm.write(ifm1 + ifm2);
			}
		}
	}
}

void HW_activation_33(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	int num = 0;
	DATA_T ifm;
	HW_activation_33_x_loop: for (x = 0; x < 2; x++) {
		HW_activation_33_y_loop: for (y = 0; y < 2; y++) {
			HW_activation_33_m_loop: for (m = 0; m < 512; m++) {
				ifm = I_strm.read();
				if (ifm < 0) {
					O_strm.write(0);}
				else{
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

  HW_global_average_pooling2d_1_x_loop: for (x=0; x<2; x++) {
	  HW_global_average_pooling2d_1_y_loop: for (y=0; y<2; y++) {
	    HW_global_average_pooling2d_1_m_1_loop: for (m=0; m<512; m++) {
#pragma HLS PIPELINE
	        ifm = I_strm.read();
	        ifm /= div;
            ofm[m] += ifm;
        }
    }
    if(x == 2 - 1){
      HW_global_average_pooling2d_1_m_2_loop: for (m=0; m<512; m++) {
#pragma HLS UNROLL
         O_strm.write(ofm[m]);
      }
    }
  }
}

void HW_dense_1(hls::stream<DATA_T> &I_strm, DATA_T W[10][512],DATA_T B[10],hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, c;
	DATA_T maximum=0;
	DATA_T ofm[10];
	DATA_T ifm;
    //Dense
	HW_dense_1_c_loop: for (c = 0; c < 512; c++){
	    ifm = I_strm.read();
		HW_dense_1_m_1_loop: for (m=0; m<10; m++) {
#pragma HLS PIPELINE
		  if(c == 0)
		    ofm[m] = B[m];
		  ofm[m] += W[m][c] * ifm;
        }
    }
    //find Max
    HW_dense_1_m_2_loop: for (m = 0; m < 10; m++){
		if(maximum<ofm[m])
		   maximum=ofm[m];
    }
    //one hot label
    HW_dense_1_m_3_loop: for (m = 0; m < 10; m++){
	    if(maximum!=ofm[m])
	        O_strm.write(0);
	    else
	        O_strm.write(1);
    }
}



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
void SW_conv2d_6(DATA_T I[64][16][16], DATA_T O[64][16][16], DATA_T W[64][64][3][3], DATA_T B[64]) {
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
void SW_conv2d_11(DATA_T I[128][8][8], DATA_T O[128][8][8], DATA_T W[128][128][3][3], DATA_T B[128]) {
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
				if (ofm < 0)
					O[m][x][y] = 0;
				else
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
				if (ofm < 0)
					O[m][x][y] = 0;
				else
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


void resnet34(DATA_T I_i[3][32][32],DATA_T W1_i[64][3][3][3],DATA_T B1_i[64], DATA_T W3_i[64][64][3][3],DATA_T B3_i[64], DATA_T W4_i[64][64][3][3],DATA_T B4_i[64], DATA_T W5_i[64][64],DATA_T B5_i[64], DATA_T W8_i[64][64][3][3],DATA_T B8_i[64], DATA_T W10_i[64][64][3][3],DATA_T B10_i[64], DATA_T W13_i[64][64][3][3],DATA_T B13_i[64], DATA_T W15_i[64][64][3][3],DATA_T B15_i[64], DATA_T W18_i[128][64][3][3],DATA_T B18_i[128], DATA_T W20_i[128][128][3][3],DATA_T B20_i[128], DATA_T W21_i[128][64],DATA_T B21_i[128], DATA_T W24_i[128][128][3][3],DATA_T B24_i[128], DATA_T W26_i[128][128][3][3],DATA_T B26_i[128], DATA_T W29_i[128][128][3][3],DATA_T B29_i[128], DATA_T W31_i[128][128][3][3],DATA_T B31_i[128], DATA_T W34_i[128][128][3][3],DATA_T B34_i[128], DATA_T W36_i[128][128][3][3],DATA_T B36_i[128], DATA_T W39_i[256][128][3][3],DATA_T B39_i[256], DATA_T W41_i[256][256][3][3],DATA_T B41_i[256], DATA_T W42_i[256][128],DATA_T B42_i[256], DATA_T W45_i[256][256][3][3],DATA_T B45_i[256], DATA_T W47_i[256][256][3][3],DATA_T B47_i[256], DATA_T W50_i[256][256][3][3],DATA_T B50_i[256], DATA_T W52_i[256][256][3][3],DATA_T B52_i[256], DATA_T W55_i[256][256][3][3],DATA_T B55_i[256], DATA_T W57_i[256][256][3][3],DATA_T B57_i[256], DATA_T W60_i[256][256][3][3],DATA_T B60_i[256], DATA_T W62_i[256][256][3][3],DATA_T B62_i[256], DATA_T W65_i[256][256][3][3],DATA_T B65_i[256], DATA_T W67_i[256][256][3][3],DATA_T B67_i[256], DATA_T W70_i[512][256][3][3],DATA_T B70_i[512], DATA_T W72_i[512][512][3][3],DATA_T B72_i[512], DATA_T W73_i[512][256],DATA_T B73_i[512], DATA_T W76_i[512][512][3][3],DATA_T B76_i[512], DATA_T W78_i[512][512][3][3],DATA_T B78_i[512], DATA_T W81_i[512][512][3][3],DATA_T B81_i[512], DATA_T W83_i[512][512][3][3],DATA_T B83_i[512], DATA_T W88_i[10][512],DATA_T B88_i[10],  DATA_T O[10])  {
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
#pragma HLS ARRAY_PARTITION variable=W18_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W18_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W18_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B18_i complete
#pragma HLS ARRAY_PARTITION variable=W20_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W20_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W20_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B20_i complete
#pragma HLS ARRAY_PARTITION variable=W21_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=B21_i complete
#pragma HLS ARRAY_PARTITION variable=W24_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W24_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W24_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B24_i complete
#pragma HLS ARRAY_PARTITION variable=W26_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W26_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W26_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B26_i complete
#pragma HLS ARRAY_PARTITION variable=W29_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W29_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W29_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B29_i complete
#pragma HLS ARRAY_PARTITION variable=W31_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W31_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W31_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B31_i complete
#pragma HLS ARRAY_PARTITION variable=W34_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W34_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W34_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B34_i complete
#pragma HLS ARRAY_PARTITION variable=W36_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W36_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W36_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B36_i complete
#pragma HLS ARRAY_PARTITION variable=W39_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W39_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W39_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B39_i complete
#pragma HLS ARRAY_PARTITION variable=W41_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W41_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W41_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B41_i complete
#pragma HLS ARRAY_PARTITION variable=W42_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=B42_i complete
#pragma HLS ARRAY_PARTITION variable=W45_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W45_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W45_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B45_i complete
#pragma HLS ARRAY_PARTITION variable=W47_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W47_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W47_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B47_i complete
#pragma HLS ARRAY_PARTITION variable=W50_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W50_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W50_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B50_i complete
#pragma HLS ARRAY_PARTITION variable=W52_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W52_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W52_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B52_i complete
#pragma HLS ARRAY_PARTITION variable=W55_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W55_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W55_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B55_i complete
#pragma HLS ARRAY_PARTITION variable=W57_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W57_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W57_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B57_i complete
#pragma HLS ARRAY_PARTITION variable=W60_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W60_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W60_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B60_i complete
#pragma HLS ARRAY_PARTITION variable=W62_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W62_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W62_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B62_i complete
#pragma HLS ARRAY_PARTITION variable=W65_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W65_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W65_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B65_i complete
#pragma HLS ARRAY_PARTITION variable=W67_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W67_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W67_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B67_i complete
#pragma HLS ARRAY_PARTITION variable=W70_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W70_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W70_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B70_i complete
#pragma HLS ARRAY_PARTITION variable=W72_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W72_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W72_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B72_i complete
#pragma HLS ARRAY_PARTITION variable=W73_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=B73_i complete
#pragma HLS ARRAY_PARTITION variable=W76_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W76_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W76_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B76_i complete
#pragma HLS ARRAY_PARTITION variable=W78_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W78_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W78_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B78_i complete
#pragma HLS ARRAY_PARTITION variable=W81_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W81_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W81_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B81_i complete
#pragma HLS ARRAY_PARTITION variable=W83_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W83_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W83_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B83_i complete
#pragma HLS ARRAY_PARTITION variable=W88_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=B88_i complete


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
	static hls::stream<DATA_T> O57_strm("O57_strm");
	static hls::stream<DATA_T> O58_strm("O58_strm");
	static hls::stream<DATA_T> O59_strm("O59_strm");
	static hls::stream<DATA_T> O60_strm("O60_strm");
	static hls::stream<DATA_T> O61_strm("O61_strm");
	static hls::stream<DATA_T> O62_strm("O62_strm");
	static hls::stream<DATA_T> O63_strm("O63_strm");
	static hls::stream<DATA_T> O64_strm("O64_strm");
	static hls::stream<DATA_T> O65_strm("O65_strm");
	static hls::stream<DATA_T> O66_strm("O66_strm");
	static hls::stream<DATA_T> O67_strm("O67_strm");
	static hls::stream<DATA_T> O68_strm("O68_strm");
	static hls::stream<DATA_T> O69_strm("O69_strm");
	static hls::stream<DATA_T> O70_strm("O70_strm");
	static hls::stream<DATA_T> O71_strm("O71_strm");
	static hls::stream<DATA_T> O72_strm("O72_strm");
	static hls::stream<DATA_T> O73_strm("O73_strm");
	static hls::stream<DATA_T> O74_strm("O74_strm");
	static hls::stream<DATA_T> O75_strm("O75_strm");
	static hls::stream<DATA_T> O76_strm("O76_strm");
	static hls::stream<DATA_T> O77_strm("O77_strm");
	static hls::stream<DATA_T> O78_strm("O78_strm");
	static hls::stream<DATA_T> O79_strm("O79_strm");
	static hls::stream<DATA_T> O80_strm("O80_strm");
	static hls::stream<DATA_T> O81_strm("O81_strm");
	static hls::stream<DATA_T> O82_strm("O82_strm");
	static hls::stream<DATA_T> O83_strm("O83_strm");
	static hls::stream<DATA_T> O84_strm("O84_strm");
	static hls::stream<DATA_T> O85_strm("O85_strm");
	static hls::stream<DATA_T> O86_strm("O86_strm");
	static hls::stream<DATA_T> O87_strm("O87_strm");
	static hls::stream<DATA_T> O88_strm("O88_strm");
	static hls::stream<DATA_T> O89_strm("O89_strm");
	static hls::stream<DATA_T> O90_strm("O90_strm");
	static hls::stream<DATA_T> O91_strm("O91_strm");
	static hls::stream<DATA_T> O92_strm("O92_strm");
	static hls::stream<DATA_T> O93_strm("O93_strm");
	static hls::stream<DATA_T> O94_strm("O94_strm");
	static hls::stream<DATA_T> O95_strm("O95_strm");
	static hls::stream<DATA_T> O96_strm("O96_strm");
	static hls::stream<DATA_T> O97_strm("O97_strm");
	static hls::stream<DATA_T> O98_strm("O98_strm");
	static hls::stream<DATA_T> O99_strm("O99_strm");
	static hls::stream<DATA_T> O100_strm("O100_strm");
	static hls::stream<DATA_T> O101_strm("O101_strm");
	static hls::stream<DATA_T> O102_strm("O102_strm");
	static hls::stream<DATA_T> O103_strm("O103_strm");
	static hls::stream<DATA_T> O104_strm("O104_strm");

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
	HW_activation_6(O18_strm ,O19_strm);
	HW_add_3(O15_strm, O19_strm, O20_strm, O21_strm); //double
	HW_conv2d_8(O20_strm, W18_i, B18_i,O22_strm);
	HW_activation_7(O22_strm ,O23_strm);
	HW_conv2d_9(O23_strm, W20_i, B20_i,O24_strm);
	HW_conv2d_10(O21_strm, W21_i, B21_i,O25_strm);
	HW_activation_8(O24_strm ,O26_strm);
	HW_add_4(O25_strm, O26_strm, O27_strm,O28_strm); //double
	HW_conv2d_11(O27_strm, W24_i, B24_i,O29_strm);
	HW_activation_9(O29_strm ,O30_strm);
	HW_conv2d_12(O30_strm, W26_i, B26_i,O31_strm);
	HW_activation_10(O31_strm ,O32_strm);
	HW_add_5(O28_strm, O32_strm, O33_strm, O34_strm); //double
	HW_conv2d_13(O33_strm, W29_i, B29_i,O35_strm);
	HW_activation_11(O35_strm ,O36_strm);
	HW_conv2d_14(O36_strm, W31_i, B31_i,O37_strm);
	HW_activation_12(O37_strm ,O38_strm);
	HW_add_6(O34_strm, O38_strm, O39_strm, O40_strm); //double
	HW_conv2d_15(O39_strm, W34_i, B34_i,O41_strm);
	HW_activation_13(O41_strm ,O42_strm);
	HW_conv2d_16(O42_strm, W36_i, B36_i,O43_strm);
	HW_activation_14(O43_strm ,O44_strm);
	HW_add_7(O40_strm, O44_strm, O45_strm, O46_strm); //double
	HW_conv2d_17(O45_strm, W39_i, B39_i,O47_strm);
	HW_activation_15(O47_strm ,O48_strm);
	HW_conv2d_18(O48_strm, W41_i, B41_i,O49_strm);
	HW_conv2d_19(O46_strm, W42_i, B42_i,O50_strm);
	HW_activation_16(O49_strm ,O51_strm);
	HW_add_8(O50_strm, O51_strm, O52_strm, O53_strm); //double
	HW_conv2d_20(O52_strm, W45_i, B45_i,O54_strm);
	HW_activation_17(O54_strm ,O55_strm);
	HW_conv2d_21(O55_strm, W47_i, B47_i,O56_strm);
	HW_activation_18(O56_strm ,O57_strm);
	HW_add_9(O53_strm, O57_strm, O58_strm, O59_strm); //double
	HW_conv2d_22(O58_strm, W50_i, B50_i,O60_strm);
	HW_activation_19(O60_strm ,O61_strm);
	HW_conv2d_23(O61_strm, W52_i, B52_i,O62_strm);
	HW_activation_20(O62_strm ,O63_strm);
	HW_add_10(O59_strm, O63_strm, O64_strm, O65_strm); //double
	HW_conv2d_24(O64_strm, W55_i, B55_i,O66_strm);
	HW_activation_21(O66_strm ,O67_strm);
	HW_conv2d_25(O67_strm, W57_i, B57_i,O68_strm);
	HW_activation_22(O68_strm ,O69_strm);
	HW_add_11(O65_strm, O69_strm, O70_strm, O71_strm); //double
	HW_conv2d_26(O70_strm, W60_i, B60_i,O72_strm);
	HW_activation_23(O72_strm ,O73_strm);
	HW_conv2d_27(O73_strm, W62_i, B62_i,O74_strm);
	HW_activation_24(O74_strm ,O75_strm);
	HW_add_12(O71_strm, O75_strm, O76_strm, O77_strm); // double
	HW_conv2d_28(O76_strm, W65_i, B65_i,O78_strm);
	HW_activation_25(O78_strm ,O79_strm);
	HW_conv2d_29(O79_strm, W67_i, B67_i,O80_strm);
	HW_activation_26(O80_strm ,O81_strm);
	HW_add_13(O77_strm, O81_strm, O82_strm,O_83_strm); // double
	HW_conv2d_30(O82_strm, W70_i, B70_i,O84_strm);
	HW_activation_27(O84_strm ,O85_strm);
	HW_conv2d_31(O85_strm, W72_i, B72_i,O86_strm);
	HW_conv2d_32(O83_strm, W73_i, B73_i,O87_strm);
	HW_activation_28(O86_strm ,O88_strm);
	HW_add_14(O87_strm, O88_strm, O89_strm, O90_strm); // double
	HW_conv2d_33(O89_strm, W76_i, B76_i,O91_strm);
	HW_activation_29(O91_strm ,O92_strm);
	HW_conv2d_34(O92_strm, W78_i, B78_i,O93_strm);
	HW_activation_30(O93_strm ,O94_strm);
	HW_add_15(O90_strm, O94_strm, O95_strm, O96_strm); // double
	HW_conv2d_35(O95_strm, W81_i, B81_i,O97_strm);
	HW_activation_31(O97_strm ,O98_strm);
	HW_conv2d_36(O98_strm, W83_i, B83_i,O99_strm);
	HW_activation_32(O99_strm ,O100_strm);
	HW_add_16(O96_strm, O100_strm, O101_strm);
	HW_activation_33(O101_strm ,O102_strm);
	HW_global_average_pooling2d_1(O102_strm ,O103_strm);
	HW_dense_1(O103_strm, W88_i, B88_i,O104_strm);
	Stream_output(O104_strm,O);
	
}

void resnet34_top(DATA_T I[3][32][32],DATA_T W1[64][3][3][3],DATA_T B1[64], DATA_T W3[64][64][3][3],DATA_T B3[64], DATA_T W4[64][64][3][3],DATA_T B4[64], DATA_T W5[64][64][1][1],DATA_T B5[64], DATA_T W8[64][64][3][3],DATA_T B8[64], DATA_T W10[64][64][3][3],DATA_T B10[64], DATA_T W13[64][64][3][3],DATA_T B13[64], DATA_T W15[64][64][3][3],DATA_T B15[64], DATA_T W18[128][64][3][3],DATA_T B18[128], DATA_T W20[128][128][3][3],DATA_T B20[128], DATA_T W21[128][64][1][1],DATA_T B21[128], DATA_T W24[128][128][3][3],DATA_T B24[128], DATA_T W26[128][128][3][3],DATA_T B26[128], DATA_T W29[128][128][3][3],DATA_T B29[128], DATA_T W31[128][128][3][3],DATA_T B31[128], DATA_T W34[128][128][3][3],DATA_T B34[128], DATA_T W36[128][128][3][3],DATA_T B36[128], DATA_T W39[256][128][3][3],DATA_T B39[256], DATA_T W41[256][256][3][3],DATA_T B41[256], DATA_T W42[256][128][1][1],DATA_T B42[256], DATA_T W45[256][256][3][3],DATA_T B45[256], DATA_T W47[256][256][3][3],DATA_T B47[256], DATA_T W50[256][256][3][3],DATA_T B50[256], DATA_T W52[256][256][3][3],DATA_T B52[256], DATA_T W55[256][256][3][3],DATA_T B55[256], DATA_T W57[256][256][3][3],DATA_T B57[256], DATA_T W60[256][256][3][3],DATA_T B60[256], DATA_T W62[256][256][3][3],DATA_T B62[256], DATA_T W65[256][256][3][3],DATA_T B65[256], DATA_T W67[256][256][3][3],DATA_T B67[256], DATA_T W70[512][256][3][3],DATA_T B70[512], DATA_T W72[512][512][3][3],DATA_T B72[512], DATA_T W73[512][256][1][1],DATA_T B73[512], DATA_T W76[512][512][3][3],DATA_T B76[512], DATA_T W78[512][512][3][3],DATA_T B78[512], DATA_T W81[512][512][3][3],DATA_T B81[512], DATA_T W83[512][512][3][3],DATA_T B83[512], DATA_T W88[10][512],DATA_T B88[10],  DATA_T O[10]) {

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
W13_i_m_loop: for (m=0; m<64; m++) {
  W13_i_k_loop: for (k=0; k<64; k++) {
  W13_i_i_loop: for (i=0; i<3; i++) {
  W13_i_j_loop: for (j=0; j<3; j++) {
  W13_i[m][k][i][j] = W13[m][k][i][j];
      }
    }
  }
}
B13_i_m_loop: for (m=0; m<64; m++) {
	B13_i[m] = B13[m];
}
W15_i_m_loop: for (m=0; m<64; m++) {
  W15_i_k_loop: for (k=0; k<64; k++) {
  W15_i_i_loop: for (i=0; i<3; i++) {
  W15_i_j_loop: for (j=0; j<3; j++) {
  W15_i[m][k][i][j] = W15[m][k][i][j];
      }
    }
  }
}
B15_i_m_loop: for (m=0; m<64; m++) {
	B15_i[m] = B15[m];
}
W18_i_m_loop: for (m=0; m<128; m++) {
  W18_i_k_loop: for (k=0; k<64; k++) {
  W18_i_i_loop: for (i=0; i<3; i++) {
  W18_i_j_loop: for (j=0; j<3; j++) {
  W18_i[m][k][i][j] = W18[m][k][i][j];
      }
    }
  }
}
B18_i_m_loop: for (m=0; m<128; m++) {
	B18_i[m] = B18[m];
}
W20_i_m_loop: for (m=0; m<128; m++) {
  W20_i_k_loop: for (k=0; k<128; k++) {
  W20_i_i_loop: for (i=0; i<3; i++) {
  W20_i_j_loop: for (j=0; j<3; j++) {
  W20_i[m][k][i][j] = W20[m][k][i][j];
      }
    }
  }
}
B20_i_m_loop: for (m=0; m<128; m++) {
	B20_i[m] = B20[m];
}
W21_i_m_loop: for (m=0; m<128; m++) {
  W21_i_k_loop: for (k=0; k<64; k++) {
  W21_i[m][k] = W21[m][k][0][0];
  }
}
B21_i_m_loop: for (m=0; m<128; m++) {
	B21_i[m] = B21[m];
}
W24_i_m_loop: for (m=0; m<128; m++) {
  W24_i_k_loop: for (k=0; k<128; k++) {
  W24_i_i_loop: for (i=0; i<3; i++) {
  W24_i_j_loop: for (j=0; j<3; j++) {
  W24_i[m][k][i][j] = W24[m][k][i][j];
      }
    }
  }
}
B24_i_m_loop: for (m=0; m<128; m++) {
	B24_i[m] = B24[m];
}
W26_i_m_loop: for (m=0; m<128; m++) {
  W26_i_k_loop: for (k=0; k<128; k++) {
  W26_i_i_loop: for (i=0; i<3; i++) {
  W26_i_j_loop: for (j=0; j<3; j++) {
  W26_i[m][k][i][j] = W26[m][k][i][j];
      }
    }
  }
}
B26_i_m_loop: for (m=0; m<128; m++) {
	B26_i[m] = B26[m];
}
W29_i_m_loop: for (m=0; m<128; m++) {
  W29_i_k_loop: for (k=0; k<128; k++) {
  W29_i_i_loop: for (i=0; i<3; i++) {
  W29_i_j_loop: for (j=0; j<3; j++) {
  W29_i[m][k][i][j] = W29[m][k][i][j];
      }
    }
  }
}
B29_i_m_loop: for (m=0; m<128; m++) {
	B29_i[m] = B29[m];
}
W31_i_m_loop: for (m=0; m<128; m++) {
  W31_i_k_loop: for (k=0; k<128; k++) {
  W31_i_i_loop: for (i=0; i<3; i++) {
  W31_i_j_loop: for (j=0; j<3; j++) {
  W31_i[m][k][i][j] = W31[m][k][i][j];
      }
    }
  }
}
B31_i_m_loop: for (m=0; m<128; m++) {
	B31_i[m] = B31[m];
}
W34_i_m_loop: for (m=0; m<128; m++) {
  W34_i_k_loop: for (k=0; k<128; k++) {
  W34_i_i_loop: for (i=0; i<3; i++) {
  W34_i_j_loop: for (j=0; j<3; j++) {
  W34_i[m][k][i][j] = W34[m][k][i][j];
      }
    }
  }
}
B34_i_m_loop: for (m=0; m<128; m++) {
	B34_i[m] = B34[m];
}
W36_i_m_loop: for (m=0; m<128; m++) {
  W36_i_k_loop: for (k=0; k<128; k++) {
  W36_i_i_loop: for (i=0; i<3; i++) {
  W36_i_j_loop: for (j=0; j<3; j++) {
  W36_i[m][k][i][j] = W36[m][k][i][j];
      }
    }
  }
}
B36_i_m_loop: for (m=0; m<128; m++) {
	B36_i[m] = B36[m];
}
W39_i_m_loop: for (m=0; m<256; m++) {
  W39_i_k_loop: for (k=0; k<128; k++) {
  W39_i_i_loop: for (i=0; i<3; i++) {
  W39_i_j_loop: for (j=0; j<3; j++) {
  W39_i[m][k][i][j] = W39[m][k][i][j];
      }
    }
  }
}
B39_i_m_loop: for (m=0; m<256; m++) {
	B39_i[m] = B39[m];
}
W41_i_m_loop: for (m=0; m<256; m++) {
  W41_i_k_loop: for (k=0; k<256; k++) {
  W41_i_i_loop: for (i=0; i<3; i++) {
  W41_i_j_loop: for (j=0; j<3; j++) {
  W41_i[m][k][i][j] = W41[m][k][i][j];
      }
    }
  }
}
B41_i_m_loop: for (m=0; m<256; m++) {
	B41_i[m] = B41[m];
}
W42_i_m_loop: for (m=0; m<256; m++) {
  W42_i_k_loop: for (k=0; k<128; k++) {
  W42_i[m][k] = W42[m][k][0][0];
  }
}
B42_i_m_loop: for (m=0; m<256; m++) {
	B42_i[m] = B42[m];
}
W45_i_m_loop: for (m=0; m<256; m++) {
  W45_i_k_loop: for (k=0; k<256; k++) {
  W45_i_i_loop: for (i=0; i<3; i++) {
  W45_i_j_loop: for (j=0; j<3; j++) {
  W45_i[m][k][i][j] = W45[m][k][i][j];
      }
    }
  }
}
B45_i_m_loop: for (m=0; m<256; m++) {
	B45_i[m] = B45[m];
}
W47_i_m_loop: for (m=0; m<256; m++) {
  W47_i_k_loop: for (k=0; k<256; k++) {
  W47_i_i_loop: for (i=0; i<3; i++) {
  W47_i_j_loop: for (j=0; j<3; j++) {
  W47_i[m][k][i][j] = W47[m][k][i][j];
      }
    }
  }
}
B47_i_m_loop: for (m=0; m<256; m++) {
	B47_i[m] = B47[m];
}
W50_i_m_loop: for (m=0; m<256; m++) {
  W50_i_k_loop: for (k=0; k<256; k++) {
  W50_i_i_loop: for (i=0; i<3; i++) {
  W50_i_j_loop: for (j=0; j<3; j++) {
  W50_i[m][k][i][j] = W50[m][k][i][j];
      }
    }
  }
}
B50_i_m_loop: for (m=0; m<256; m++) {
	B50_i[m] = B50[m];
}
W52_i_m_loop: for (m=0; m<256; m++) {
  W52_i_k_loop: for (k=0; k<256; k++) {
  W52_i_i_loop: for (i=0; i<3; i++) {
  W52_i_j_loop: for (j=0; j<3; j++) {
  W52_i[m][k][i][j] = W52[m][k][i][j];
      }
    }
  }
}
B52_i_m_loop: for (m=0; m<256; m++) {
	B52_i[m] = B52[m];
}
W55_i_m_loop: for (m=0; m<256; m++) {
  W55_i_k_loop: for (k=0; k<256; k++) {
  W55_i_i_loop: for (i=0; i<3; i++) {
  W55_i_j_loop: for (j=0; j<3; j++) {
  W55_i[m][k][i][j] = W55[m][k][i][j];
      }
    }
  }
}
B55_i_m_loop: for (m=0; m<256; m++) {
	B55_i[m] = B55[m];
}
W57_i_m_loop: for (m=0; m<256; m++) {
  W57_i_k_loop: for (k=0; k<256; k++) {
  W57_i_i_loop: for (i=0; i<3; i++) {
  W57_i_j_loop: for (j=0; j<3; j++) {
  W57_i[m][k][i][j] = W57[m][k][i][j];
      }
    }
  }
}
B57_i_m_loop: for (m=0; m<256; m++) {
	B57_i[m] = B57[m];
}
W60_i_m_loop: for (m=0; m<256; m++) {
  W60_i_k_loop: for (k=0; k<256; k++) {
  W60_i_i_loop: for (i=0; i<3; i++) {
  W60_i_j_loop: for (j=0; j<3; j++) {
  W60_i[m][k][i][j] = W60[m][k][i][j];
      }
    }
  }
}
B60_i_m_loop: for (m=0; m<256; m++) {
	B60_i[m] = B60[m];
}
W62_i_m_loop: for (m=0; m<256; m++) {
  W62_i_k_loop: for (k=0; k<256; k++) {
  W62_i_i_loop: for (i=0; i<3; i++) {
  W62_i_j_loop: for (j=0; j<3; j++) {
  W62_i[m][k][i][j] = W62[m][k][i][j];
      }
    }
  }
}
B62_i_m_loop: for (m=0; m<256; m++) {
	B62_i[m] = B62[m];
}
W65_i_m_loop: for (m=0; m<256; m++) {
  W65_i_k_loop: for (k=0; k<256; k++) {
  W65_i_i_loop: for (i=0; i<3; i++) {
  W65_i_j_loop: for (j=0; j<3; j++) {
  W65_i[m][k][i][j] = W65[m][k][i][j];
      }
    }
  }
}
B65_i_m_loop: for (m=0; m<256; m++) {
	B65_i[m] = B65[m];
}
W67_i_m_loop: for (m=0; m<256; m++) {
  W67_i_k_loop: for (k=0; k<256; k++) {
  W67_i_i_loop: for (i=0; i<3; i++) {
  W67_i_j_loop: for (j=0; j<3; j++) {
  W67_i[m][k][i][j] = W67[m][k][i][j];
      }
    }
  }
}
B67_i_m_loop: for (m=0; m<256; m++) {
	B67_i[m] = B67[m];
}
W70_i_m_loop: for (m=0; m<512; m++) {
  W70_i_k_loop: for (k=0; k<256; k++) {
  W70_i_i_loop: for (i=0; i<3; i++) {
  W70_i_j_loop: for (j=0; j<3; j++) {
  W70_i[m][k][i][j] = W70[m][k][i][j];
      }
    }
  }
}
B70_i_m_loop: for (m=0; m<512; m++) {
	B70_i[m] = B70[m];
}
W72_i_m_loop: for (m=0; m<512; m++) {
  W72_i_k_loop: for (k=0; k<512; k++) {
  W72_i_i_loop: for (i=0; i<3; i++) {
  W72_i_j_loop: for (j=0; j<3; j++) {
  W72_i[m][k][i][j] = W72[m][k][i][j];
      }
    }
  }
}
B72_i_m_loop: for (m=0; m<512; m++) {
	B72_i[m] = B72[m];
}
W73_i_m_loop: for (m=0; m<512; m++) {
  W73_i_k_loop: for (k=0; k<256; k++) {
  W73_i[m][k] = W73[m][k][0][0];
  }
}
B73_i_m_loop: for (m=0; m<512; m++) {
	B73_i[m] = B73[m];
}
W76_i_m_loop: for (m=0; m<512; m++) {
  W76_i_k_loop: for (k=0; k<512; k++) {
  W76_i_i_loop: for (i=0; i<3; i++) {
  W76_i_j_loop: for (j=0; j<3; j++) {
  W76_i[m][k][i][j] = W76[m][k][i][j];
      }
    }
  }
}
B76_i_m_loop: for (m=0; m<512; m++) {
	B76_i[m] = B76[m];
}
W78_i_m_loop: for (m=0; m<512; m++) {
  W78_i_k_loop: for (k=0; k<512; k++) {
  W78_i_i_loop: for (i=0; i<3; i++) {
  W78_i_j_loop: for (j=0; j<3; j++) {
  W78_i[m][k][i][j] = W78[m][k][i][j];
      }
    }
  }
}
B78_i_m_loop: for (m=0; m<512; m++) {
	B78_i[m] = B78[m];
}
W81_i_m_loop: for (m=0; m<512; m++) {
  W81_i_k_loop: for (k=0; k<512; k++) {
  W81_i_i_loop: for (i=0; i<3; i++) {
  W81_i_j_loop: for (j=0; j<3; j++) {
  W81_i[m][k][i][j] = W81[m][k][i][j];
      }
    }
  }
}
B81_i_m_loop: for (m=0; m<512; m++) {
	B81_i[m] = B81[m];
}
W83_i_m_loop: for (m=0; m<512; m++) {
  W83_i_k_loop: for (k=0; k<512; k++) {
  W83_i_i_loop: for (i=0; i<3; i++) {
  W83_i_j_loop: for (j=0; j<3; j++) {
  W83_i[m][k][i][j] = W83[m][k][i][j];
      }
    }
  }
}
B83_i_m_loop: for (m=0; m<512; m++) {
	B83_i[m] = B83[m];
}
W88_i_m_loop: for (m=0; m<10; m++) {
  W88_i_k_loop: for (k=0; k<512; k++) {
  W88_i[m][k] = W88[m][k];
      }
}
B88_i_m_loop: for (m=0; m<10; m++) {
	B88_i[m] = B88[m];
}


    resnet34(I_i, W1_i, B1_i, W3_i, B3_i, W4_i, B4_i, W5_i, B5_i, W8_i, B8_i, W10_i, B10_i, W13_i, B13_i, W15_i, B15_i, W18_i, B18_i, W20_i, B20_i, W21_i, B21_i, W24_i, B24_i, W26_i, B26_i, W29_i, B29_i, W31_i, B31_i, W34_i, B34_i, W36_i, B36_i, W39_i, B39_i, W41_i, B41_i, W42_i, B42_i, W45_i, B45_i, W47_i, B47_i, W50_i, B50_i, W52_i, B52_i, W55_i, B55_i, W57_i, B57_i, W60_i, B60_i, W62_i, B62_i, W65_i, B65_i, W67_i, B67_i, W70_i, B70_i, W72_i, B72_i, W73_i, B73_i, W76_i, B76_i, W78_i, B78_i, W81_i, B81_i, W83_i, B83_i, W88_i, B88_i,  O_i);

    	for(m=0; m<10; m++) { O[m] = O_i[m]; }

}

void resnet34_sw(DATA_T I[3][32][32],DATA_T W1[64][3][3][3],DATA_T B1[64], DATA_T W3[64][64][3][3],DATA_T B3[64], DATA_T W4[64][64][3][3],DATA_T B4[64], DATA_T W5[64][64][1][1],DATA_T B5[64], DATA_T W8[64][64][3][3],DATA_T B8[64], DATA_T W10[64][64][3][3],DATA_T B10[64], DATA_T W13[64][64][3][3],DATA_T B13[64], DATA_T W15[64][64][3][3],DATA_T B15[64], DATA_T W18[128][64][3][3],DATA_T B18[128], DATA_T W20[128][128][3][3],DATA_T B20[128], DATA_T W21[128][64][1][1],DATA_T B21[128], DATA_T W24[128][128][3][3],DATA_T B24[128], DATA_T W26[128][128][3][3],DATA_T B26[128], DATA_T W29[128][128][3][3],DATA_T B29[128], DATA_T W31[128][128][3][3],DATA_T B31[128], DATA_T W34[128][128][3][3],DATA_T B34[128], DATA_T W36[128][128][3][3],DATA_T B36[128], DATA_T W39[256][128][3][3],DATA_T B39[256], DATA_T W41[256][256][3][3],DATA_T B41[256], DATA_T W42[256][128][1][1],DATA_T B42[256], DATA_T W45[256][256][3][3],DATA_T B45[256], DATA_T W47[256][256][3][3],DATA_T B47[256], DATA_T W50[256][256][3][3],DATA_T B50[256], DATA_T W52[256][256][3][3],DATA_T B52[256], DATA_T W55[256][256][3][3],DATA_T B55[256], DATA_T W57[256][256][3][3],DATA_T B57[256], DATA_T W60[256][256][3][3],DATA_T B60[256], DATA_T W62[256][256][3][3],DATA_T B62[256], DATA_T W65[256][256][3][3],DATA_T B65[256], DATA_T W67[256][256][3][3],DATA_T B67[256], DATA_T W70[512][256][3][3],DATA_T B70[512], DATA_T W72[512][512][3][3],DATA_T B72[512], DATA_T W73[512][256][1][1],DATA_T B73[512], DATA_T W76[512][512][3][3],DATA_T B76[512], DATA_T W78[512][512][3][3],DATA_T B78[512], DATA_T W81[512][512][3][3],DATA_T B81[512], DATA_T W83[512][512][3][3],DATA_T B83[512], DATA_T W88[10][512],DATA_T B88[10],  DATA_T O88_SW[10]) {
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
	

    SW_conv2d_1(I,O1_SW,W1,B1);
	SW_activation_1(O1_SW,O2_SW);
	SW_res0a_branch2a(O2_SW,O3_SW,W3,B3);
	SW_conv2d_2(O3_SW,O4_SW,W4,B4);
	SW_conv2d_3(O2_SW,O5_SW,W5,B5);
	SW_activation_2(O4_SW,O6_SW);
	SW_add_1(O5_SW,O6_SW,O7_SW);
	SW_conv2d_4(O7_SW,O8_SW,W8,B8);
	SW_activation_3(O8_SW,O9_SW);
	SW_conv2d_5(O9_SW,O10_SW,W10,B10);
	SW_activation_4(O10_SW,O11_SW);
	SW_add_2(O7_SW,O11_SW,O12_SW);
	SW_conv2d_6(O12_SW,O13_SW,W13,B13);
	SW_activation_5(O13_SW,O14_SW);
	SW_conv2d_7(O14_SW,O15_SW,W15,B15);
	SW_activation_6(O15_SW,O16_SW);
	SW_add_3(O12_SW,O16_SW,O17_SW);
	SW_conv2d_8(O17_SW,O18_SW,W18,B18);
	SW_activation_7(O18_SW,O19_SW);
	SW_conv2d_9(O19_SW,O20_SW,W20,B20);
	SW_conv2d_10(O17_SW,O21_SW,W21,B21);
	SW_activation_8(O20_SW,O22_SW);
	SW_add_4(O21_SW,O22_SW,O23_SW);
	SW_conv2d_11(O23_SW,O24_SW,W24,B24);
	SW_activation_9(O24_SW,O25_SW);
	SW_conv2d_12(O25_SW,O26_SW,W26,B26);
	SW_activation_10(O26_SW,O27_SW);
	SW_add_5(O23_SW,O27_SW,O28_SW);
	SW_conv2d_13(O28_SW,O29_SW,W29,B29);
	SW_activation_11(O29_SW,O30_SW);
	SW_conv2d_14(O30_SW,O31_SW,W31,B31);
	SW_activation_12(O31_SW,O32_SW);
	SW_add_6(O28_SW,O32_SW,O33_SW);
	SW_conv2d_15(O33_SW,O34_SW,W34,B34);
	SW_activation_13(O34_SW,O35_SW);
	SW_conv2d_16(O35_SW,O36_SW,W36,B36);
	SW_activation_14(O36_SW,O37_SW);
	SW_add_7(O33_SW,O37_SW,O38_SW);
	SW_conv2d_17(O38_SW,O39_SW,W39,B39);
	SW_activation_15(O39_SW,O40_SW);
	SW_conv2d_18(O40_SW,O41_SW,W41,B41);
	SW_conv2d_19(O38_SW,O42_SW,W42,B42);
	SW_activation_16(O41_SW,O43_SW);
	SW_add_8(O42_SW,O43_SW,O44_SW);
	SW_conv2d_20(O44_SW,O45_SW,W45,B45);
	SW_activation_17(O45_SW,O46_SW);
	SW_conv2d_21(O46_SW,O47_SW,W47,B47);
	SW_activation_18(O47_SW,O48_SW);
	SW_add_9(O44_SW,O48_SW,O49_SW);
	SW_conv2d_22(O49_SW,O50_SW,W50,B50);
	SW_activation_19(O50_SW,O51_SW);
	SW_conv2d_23(O51_SW,O52_SW,W52,B52);
	SW_activation_20(O52_SW,O53_SW);
	SW_add_10(O49_SW,O53_SW,O54_SW);
	SW_conv2d_24(O54_SW,O55_SW,W55,B55);
	SW_activation_21(O55_SW,O56_SW);
	SW_conv2d_25(O56_SW,O57_SW,W57,B57);
	SW_activation_22(O57_SW,O58_SW);
	SW_add_11(O54_SW,O58_SW,O59_SW);
	SW_conv2d_26(O59_SW,O60_SW,W60,B60);
	SW_activation_23(O60_SW,O61_SW);
	SW_conv2d_27(O61_SW,O62_SW,W62,B62);
	SW_activation_24(O62_SW,O63_SW);
	SW_add_12(O59_SW,O63_SW,O64_SW);
	SW_conv2d_28(O64_SW,O65_SW,W65,B65);
	SW_activation_25(O65_SW,O66_SW);
	SW_conv2d_29(O66_SW,O67_SW,W67,B67);
	SW_activation_26(O67_SW,O68_SW);
	SW_add_13(O64_SW,O68_SW,O69_SW);
	SW_conv2d_30(O69_SW,O70_SW,W70,B70);
	SW_activation_27(O70_SW,O71_SW);
	SW_conv2d_31(O71_SW,O72_SW,W72,B72);
	SW_conv2d_32(O69_SW,O73_SW,W73,B73);
	SW_activation_28(O72_SW,O74_SW);
	SW_add_14(O73_SW,O74_SW,O75_SW);
	SW_conv2d_33(O75_SW,O76_SW,W76,B76);
	SW_activation_29(O76_SW,O77_SW);
	SW_conv2d_34(O77_SW,O78_SW,W78,B78);
	SW_activation_30(O78_SW,O79_SW);
	SW_add_15(O75_SW,O79_SW,O80_SW);
	SW_conv2d_35(O80_SW,O81_SW,W81,B81);
	SW_activation_31(O81_SW,O82_SW);
	SW_conv2d_36(O82_SW,O83_SW,W83,B83);
	SW_activation_32(O83_SW,O84_SW);
	SW_add_16(O80_SW,O84_SW,O85_SW);
	SW_activation_33(O85_SW,O86_SW);
	SW_global_average_pooling2d_1(O86_SW,O87_SW);
	SW_dense_1(O87_SW,O88_SW,W88,B88);
	
}