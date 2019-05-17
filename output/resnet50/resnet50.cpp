#include <iostream>
#include <ap_int.h>
#include <hls_stream.h>

typedef int DATA_T;
typedef ap_uint<256> uint256_t;
typedef ap_uint<512> uint512_t;

void Stream_input(DATA_T I[3][13][13], hls::stream<DATA_T> &I_strm) {
  int k, x, y;

#pragma HLS ARRAY_PARTITION variable=I complete dim=1
	Stream_input_x_loop: for (x=0; x<13; x++) {
	 	Stream_input_y_loop: for (y=0; y<13; y++) {
#pragma HLS PIPELINE
	 		Stream_input_m_loop: for (k=0; k<3; k++) {
	 			I_strm.write(I[k][x][y]);
	    }
	  }
	}
}

void Stream_output(hls::stream<DATA_T> &O_strm, DATA_T O[5][3][3]) {
  int m, x, y;
#pragma HLS ARRAY_PARTITION variable=O complete dim=1

	Stream_input_x_loop: for (x=0; x<3; x++) {
	 	Stream_input_y_loop: for (y=0; y<3; y++) {
#pragma HLS PIPELINE
	 		Stream_input_m_loop: for (m=0; m<5; m++) {
	 			O[m][x][y] = O_strm.read();
	    }
	  }
	}
}


static DATA_T I_i[3][13][13];
	static DATA_T W2_i[4][3][7][7];
	static DATA_T B2_i[4];
	static DATA_T W5_i[4][4];
	static DATA_T B5_i[4];
	static DATA_T W7_i[4][4][3][3];
	static DATA_T B7_i[4];
	static DATA_T W9_i[5][4];
	static DATA_T B9_i[5];
	static DATA_T W10_i[5][4];
	static DATA_T B10_i[5];
	

void SW_conv1_pad(DATA_T I[3][13][13],DATA_T O[3][19][19]) {
	int m, x, y, i, j;
	for (m = 0; m < 3; m++) {
		for (x = 0; x < 19; x++) {
			for (y = 0; y < 19; y++) {
				O[m][x][y] = 0;
			}
		}
	}
	for (m = 0; m < 3; m++) {
		for (x = 0; x < 13; x++) {
			for (y = 0; y < 13; y++) {
				O[m][x+ 3][y+ 3] = I[m][x][y];
			}
		}
	}
}
void HW_conv1(hls::stream<DATA_T> &I_strm, DATA_T W[4][3][7][7], DATA_T B[4], hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y, i, j, k;
	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6;
	DATA_T I_7, I_8, I_9, I_10, I_11, I_12, I_13;
	DATA_T I_14, I_15, I_16, I_17, I_18, I_19, I_20;
	DATA_T I_21, I_22, I_23, I_24, I_25, I_26, I_27;
	DATA_T I_28, I_29, I_30, I_31, I_32, I_33, I_34;
	DATA_T I_35, I_36, I_37, I_38, I_39, I_40, I_41;
	DATA_T I_42, I_43, I_44, I_45, I_46, I_47, I_48;
	DATA_T ifm[3], ofm[4];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[3][7][19];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_conv1_x_loop: for (x = 0; x < 19; x++) {
		HW_conv1_y_loop: for (y = 0; y < 19; y++) {
			HW_conv1_k_loop: for (k = 0; k < 3; k++) {
				I[k][x % 7][y] = I_strm.read();
				if (x >= 6 && y >= 6) {
					if (x % 2 == 0 && y % 2 == 0) {
						I_0 = I[k][(x - 6) % 7][(y - 6)];
						I_1 = I[k][(x - 5) % 7][(y - 6)];
						I_2 = I[k][(x - 4) % 7][(y - 6)];
						I_3 = I[k][(x - 3) % 7][(y - 6)];
						I_4 = I[k][(x - 2) % 7][(y - 6)];
						I_5 = I[k][(x - 1) % 7][(y - 6)];
						I_6 = I[k][(x) % 7][(y - 6)];
						I_7 = I[k][(x - 6) % 7][(y - 5)];
						I_8 = I[k][(x - 5) % 7][(y - 5)];
						I_9 = I[k][(x - 4) % 7][(y - 5)];
						I_10 = I[k][(x - 3) % 7][(y - 5)];
						I_11 = I[k][(x - 2) % 7][(y - 5)];
						I_12 = I[k][(x - 1) % 7][(y - 5)];
						I_13 = I[k][(x) % 7][(y - 5)];
						I_14 = I[k][(x - 6) % 7][(y - 4)];
						I_15 = I[k][(x - 5) % 7][(y - 4)];
						I_16 = I[k][(x - 4) % 7][(y - 4)];
						I_17 = I[k][(x - 3) % 7][(y - 4)];
						I_18 = I[k][(x - 2) % 7][(y - 4)];
						I_19 = I[k][(x - 1) % 7][(y - 4)];
						I_20 = I[k][(x) % 7][(y - 4)];
						I_21 = I[k][(x - 6) % 7][(y - 3)];
						I_22 = I[k][(x - 5) % 7][(y - 3)];
						I_23 = I[k][(x - 4) % 7][(y - 3)];
						I_24 = I[k][(x - 3) % 7][(y - 3)];
						I_25 = I[k][(x - 2) % 7][(y - 3)];
						I_26 = I[k][(x - 1) % 7][(y - 3)];
						I_27 = I[k][(x) % 7][(y - 3)];
						I_28 = I[k][(x - 6) % 7][(y - 2)];
						I_29 = I[k][(x - 5) % 7][(y - 2)];
						I_30 = I[k][(x - 4) % 7][(y - 2)];
						I_31 = I[k][(x - 3) % 7][(y - 2)];
						I_32 = I[k][(x - 2) % 7][(y - 2)];
						I_33 = I[k][(x - 1) % 7][(y - 2)];
						I_34 = I[k][(x) % 7][(y - 2)];
						I_35 = I[k][(x - 6) % 7][(y - 1)];
						I_36 = I[k][(x - 5) % 7][(y - 1)];
						I_37 = I[k][(x - 4) % 7][(y - 1)];
						I_38 = I[k][(x - 3) % 7][(y - 1)];
						I_39 = I[k][(x - 2) % 7][(y - 1)];
						I_40 = I[k][(x - 1) % 7][(y - 1)];
						I_41 = I[k][(x) % 7][(y - 1)];
						I_42 = I[k][(x - 6) % 7][(y)];
						I_43 = I[k][(x - 5) % 7][(y)];
						I_44 = I[k][(x - 4) % 7][(y)];
						I_45 = I[k][(x - 3) % 7][(y)];
						I_46 = I[k][(x - 2) % 7][(y)];
						I_47 = I[k][(x - 1) % 7][(y)];
						I_48 = I[k][(x) % 7][(y)];

						HW_conv1_m_loop: for (m = 0; m < 4; m++) {
#pragma HLS PIPELINE
							if (k == 0) {
								ofm[m] = B[m];
							}
							ofm[m] = ofm[m] + I_0 * W[m][k][0][0];
							ofm[m] = ofm[m] + I_1 * W[m][k][1][0];
							ofm[m] = ofm[m] + I_2 * W[m][k][2][0];
							ofm[m] = ofm[m] + I_3 * W[m][k][3][0];
							ofm[m] = ofm[m] + I_4 * W[m][k][4][0];
							ofm[m] = ofm[m] + I_5 * W[m][k][5][0];
							ofm[m] = ofm[m] + I_6 * W[m][k][6][0];
							ofm[m] = ofm[m] + I_7 * W[m][k][0][1];
							ofm[m] = ofm[m] + I_8 * W[m][k][1][1];
							ofm[m] = ofm[m] + I_9 * W[m][k][2][1];
							ofm[m] = ofm[m] + I_10 * W[m][k][3][1];
							ofm[m] = ofm[m] + I_11 * W[m][k][4][1];
							ofm[m] = ofm[m] + I_12 * W[m][k][5][1];
							ofm[m] = ofm[m] + I_13 * W[m][k][6][1];
							ofm[m] = ofm[m] + I_14 * W[m][k][0][2];
							ofm[m] = ofm[m] + I_15 * W[m][k][1][2];
							ofm[m] = ofm[m] + I_16 * W[m][k][2][2];
							ofm[m] = ofm[m] + I_17 * W[m][k][3][2];
							ofm[m] = ofm[m] + I_18 * W[m][k][4][2];
							ofm[m] = ofm[m] + I_19 * W[m][k][5][2];
							ofm[m] = ofm[m] + I_20 * W[m][k][6][2];
							ofm[m] = ofm[m] + I_21 * W[m][k][0][3];
							ofm[m] = ofm[m] + I_22 * W[m][k][1][3];
							ofm[m] = ofm[m] + I_23 * W[m][k][2][3];
							ofm[m] = ofm[m] + I_24 * W[m][k][3][3];
							ofm[m] = ofm[m] + I_25 * W[m][k][4][3];
							ofm[m] = ofm[m] + I_26 * W[m][k][5][3];
							ofm[m] = ofm[m] + I_27 * W[m][k][6][3];
							ofm[m] = ofm[m] + I_28 * W[m][k][0][4];
							ofm[m] = ofm[m] + I_29 * W[m][k][1][4];
							ofm[m] = ofm[m] + I_30 * W[m][k][2][4];
							ofm[m] = ofm[m] + I_31 * W[m][k][3][4];
							ofm[m] = ofm[m] + I_32 * W[m][k][4][4];
							ofm[m] = ofm[m] + I_33 * W[m][k][5][4];
							ofm[m] = ofm[m] + I_34 * W[m][k][6][4];
							ofm[m] = ofm[m] + I_35 * W[m][k][0][5];
							ofm[m] = ofm[m] + I_36 * W[m][k][1][5];
							ofm[m] = ofm[m] + I_37 * W[m][k][2][5];
							ofm[m] = ofm[m] + I_38 * W[m][k][3][5];
							ofm[m] = ofm[m] + I_39 * W[m][k][4][5];
							ofm[m] = ofm[m] + I_40 * W[m][k][5][5];
							ofm[m] = ofm[m] + I_41 * W[m][k][6][5];
							ofm[m] = ofm[m] + I_42 * W[m][k][0][6];
							ofm[m] = ofm[m] + I_43 * W[m][k][1][6];
							ofm[m] = ofm[m] + I_44 * W[m][k][2][6];
							ofm[m] = ofm[m] + I_45 * W[m][k][3][6];
							ofm[m] = ofm[m] + I_46 * W[m][k][4][6];
							ofm[m] = ofm[m] + I_47 * W[m][k][5][6];
							ofm[m] = ofm[m] + I_48 * W[m][k][6][6];
						}
						if (k == 3 - 1) {
							for (m = 0; m < 4; m++) {
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
void HW_activation_1(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	int num = 0;
	DATA_T ifm;
	HW_activation_1_x_loop: for (x = 0; x < 7; x++) {
		HW_activation_1_y_loop: for (y = 0; y < 7; y++) {
			HW_activation_1_m_loop: for (m = 0; m < 4; m++) {
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

void HW_max_pooling2d_1(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y ;
	ap_uint<32> max;
	DATA_T I[4][3][7];
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_max_pooling2d_1_x_loop: for (x = 0; x < 7; x++) {
		HW_max_pooling2d_1_y_loop: for (y = 0; y < 7; y++) {
			HW_max_pooling2d_1_m_loop: for (m = 0; m < 4; m++) {
#pragma HLS PIPELINE
				I[m][x % 3][y] = I_strm.read();
				if (x >= 2 && y >= 2) {
					if (x % 2 == 0 && y % 2 == 0) {
						max = I[m][(x - 2) % 3][y - 2];
						if (I[m][(x - 1) % 3][y - 2] > max) {
							max = I[m][(x - 1) % 3][y - 2];
						}
						if (I[m][(x) % 3][y - 2] > max) {
							max = I[m][(x) % 3][y - 2];
						}
						if (I[m][(x - 2) % 3][y - 1] > max) {
							max = I[m][(x - 2) % 3][y - 1];
						}
						if (I[m][(x - 1) % 3][y - 1] > max) {
							max = I[m][(x - 1) % 3][y - 1];
						}
						if (I[m][(x) % 3][y - 1] > max) {
							max = I[m][(x) % 3][y - 1];
						}
						if (I[m][(x - 2) % 3][y] > max) {
							max = I[m][(x - 2) % 3][y];
						}
						if (I[m][(x - 1) % 3][y] > max) {
							max = I[m][(x - 1) % 3][y];
						}
						if (I[m][(x) % 3][y] > max) {
							max = I[m][(x) % 3][y];
						}
						O_strm.write(max);
					}
				}
			}
		}
	}
}

void HW_res2a_branch2a(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm, DATA_T W[4][4], DATA_T B[4]) {
#pragma HLS INLINE
	int m, x, y, k;
	DATA_T ofm[4];
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[4][3];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2

	HW_res2a_branch2a_x_loop: for (x = 0; x < 3; x++) {
		HW_res2a_branch2a_y_loop: for (y = 0; y < 3; y++) {
			HW_res2a_branch2a_k_loop: for (k = 0; k < 4; k++) {
				I[k][y] = I_strm.read();
				HW_res2a_branch2a_m_loop: for (m = 0; m < 4; m++) {
#pragma HLS PIPELINE
					if (k == 0) {
						ofm[m] = B[m];}
					ofm[m] = ofm[m] + I[k][y] * W[m][k];
				}
				if (k == 3) {
					for (m = 0; m < 4; m++) {
#pragma HLS UNROLL
						O_strm.write(ofm[m]);
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
	HW_activation_2_x_loop: for (x = 0; x < 3; x++) {
		HW_activation_2_y_loop: for (y = 0; y < 3; y++) {
			HW_activation_2_m_loop: for (m = 0; m < 4; m++) {
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

void HW_res2a_branch2b(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm, DATA_T W[4][4][3][3],DATA_T B[4]) {
#pragma HLS INLINE
	int m, x, y, k;

	DATA_T I_0, I_1, I_2, I_3, I_4, I_5, I_6, I_7, I_8;
	DATA_T ifm[4], ofm[4];
#pragma HLS ARRAY_PARTITION variable=ifm complete dim=0
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[4][3][3];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2
#pragma HLS ARRAY_PARTITION variable=I complete dim=3

	HW_res2a_branch2b_x_loop: for (x = 0; x<3 ; x++) {
		HW_res2a_branch2b_y_loop: for (y = 0; y<3 ; y++) {
			HW_res2a_branch2b_k_loop: for (k = 0; k<4; k++) {
				if (x<3 && y<3) {
					I[k][x % 3][y] = I_strm.read();
				}
				if (x > 0 && y > 0){
#pragma HLS PIPELINE
					if (x - 2 + 0 < 3 && y - 2 + 0 < 3 && x - 2 + 0 >= 0 && y - 2 + 0 >= 0) {
						I_0 = I[k][(x - 2) % 3][(y - 2)];
					}
					else {
						I_0 = 0; // zero padding
					}
					if (x - 2 + 0 < 3 && y - 2 + 1 < 3 && x - 2 + 0 >= 0 && y - 2 + 1 >= 0) {
						I_1 = I[k][(x - 2) % 3][(y - 1)];
					}
					else {
						I_1 = 0; // zero padding
					}
					if (x - 2 + 0 < 3 && y - 2 + 2 < 3 && x - 2 + 0 >= 0 && y - 2 + 2 >= 0) {
						I_2 = I[k][(x - 2) % 3][(y)];
					}
					else {
						I_2 = 0; // zero padding
					}

					if (x - 2 + 1 < 3 && y - 2 + 0 < 3 && x - 2 + 1 >= 0 && y - 2 + 0 >= 0) {
						I_3 = I[k][(x - 1) % 3][(y - 2)];
					}
					else {
						I_3 = 0; // zero padding
					}
					if (x - 2 + 1 < 3 && y - 2 + 1 < 3 && x - 2 + 1 >= 0 && y - 2 + 1 >= 0) {
						I_4 = I[k][(x - 1) % 3][(y - 1)];
					}
					else {
						I_4 = 0; // zero padding
					}
					if (x - 2 + 1 < 3 && y - 2 + 2 < 3 && x - 2 + 1 >= 0 && y - 2 + 2 >= 0) {
						I_5 = I[k][(x - 1) % 3][(y)];
					}
					else {
						I_5 = 0; // zero padding
					}
					if (x - 2 + 2 < 3 && y - 2 + 0 < 3 && x - 2 + 2 >= 0 && y - 2 + 0 >= 0) {
						I_6 = I[k][(x) % 3][(y - 2)];
					}
					else {
						I_6 = 0; // zero padding
					}
					if (x - 2 + 2 < 3 && y - 2 + 1 < 3 && x - 2 + 2 >= 0 && y - 2 + 1 >= 0) {
						I_7 = I[k][(x) % 3][(y - 1)];
					}
					else {
						I_7 = 0; // zero padding
					}
					if (x - 2 + 2 < 3 && y - 2 + 2 < 3 && x - 2 + 2 >= 0 && y - 2 + 2 >= 0) {
						I_8 = I[k][(x) % 3][(y)];
					}
					else {
						I_8 = 0; // zero padding
					}

					HW_res2a_branch2b_m_loop: for (m = 0; m<4; m++) {
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
					if (k == 3) {
						for (m = 0; m<4; m++) {
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
	HW_activation_3_x_loop: for (x = 0; x < 3; x++) {
		HW_activation_3_y_loop: for (y = 0; y < 3; y++) {
			HW_activation_3_m_loop: for (m = 0; m < 4; m++) {
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

void HW_res2a_branch2c(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm, DATA_T W[5][4], DATA_T B[5]) {
#pragma HLS INLINE
	int m, x, y, k;
	DATA_T ofm[5];
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[4][3];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2

	HW_res2a_branch2c_x_loop: for (x = 0; x < 3; x++) {
		HW_res2a_branch2c_y_loop: for (y = 0; y < 3; y++) {
			HW_res2a_branch2c_k_loop: for (k = 0; k < 4; k++) {
				I[k][y] = I_strm.read();
				HW_res2a_branch2c_m_loop: for (m = 0; m < 5; m++) {
#pragma HLS PIPELINE
					if (k == 0) {
						ofm[m] = B[m];}
					ofm[m] = ofm[m] + I[k][y] * W[m][k];
				}
				if (k == 3) {
					for (m = 0; m < 5; m++) {
#pragma HLS UNROLL
						O_strm.write(ofm[m]);
					}
				}
			}
		}
	}
}

void HW_res2a_branch1(hls::stream<DATA_T> &I_strm, hls::stream<DATA_T> &O_strm, DATA_T W[5][4], DATA_T B[5]) {
#pragma HLS INLINE
	int m, x, y, k;
	DATA_T ofm[5];
#pragma HLS ARRAY_PARTITION variable=ofm complete dim=0

	DATA_T I[4][3];
#pragma HLS ARRAY_PARTITION variable=I complete dim=1
#pragma HLS ARRAY_PARTITION variable=I complete dim=2

	HW_res2a_branch1_x_loop: for (x = 0; x < 3; x++) {
		HW_res2a_branch1_y_loop: for (y = 0; y < 3; y++) {
			HW_res2a_branch1_k_loop: for (k = 0; k < 4; k++) {
				I[k][y] = I_strm.read();
				HW_res2a_branch1_m_loop: for (m = 0; m < 5; m++) {
#pragma HLS PIPELINE
					if (k == 0) {
						ofm[m] = B[m];}
					ofm[m] = ofm[m] + I[k][y] * W[m][k];
				}
				if (k == 3) {
					for (m = 0; m < 5; m++) {
#pragma HLS UNROLL
						O_strm.write(ofm[m]);
					}
				}
			}
		}
	}
}

void HW_add_1(hls::stream<DATA_T> &I1_strm, hls::stream<DATA_T> I2_strm, hls::stream<DATA_T> &O_strm) {
#pragma HLS INLINE
	int m, x, y;
	DATA_T ifm1;
	DATA_T ifm2;
	HW_add_1_x_loop_1: for (x = 0; x < 3; x++) {
	    HW_add_1_y_loop_1: for (y = 0; y < 3; y++) {
	    	HW_add_1_m_loop_1: for (m = 0; m < 5; m++) {
				ifm1 = I1_strm.read();
				ifm2 = I2_strm.read();
				O_strm.write(ifm1 + ifm2);
			}
		}
	}
}



void SW_conv1_pad(DATA_T I[3][13][13],DATA_T O[3][19][19]) {
	int m, x, y, i, j;
	for (m = 0; m < 3; m++) {
		for (x = 0; x < 19; x++) {
			for (y = 0; y < 19; y++) {
				O[m][x][y] = 0;
			}
		}
	}
	for (m = 0; m < 3; m++) {
		for (x = 0; x < 13; x++) {
			for (y = 0; y < 13; y++) {
				O[m][x+ 3][y+ 3] = I[m][x][y];
			}
		}
	}
}
void SW_conv1(DATA_T I[3][19][19], DATA_T O[4][7][7], DATA_T W[4][3][7][7], DATA_T B[4]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<4; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ofm = B[m];
				for (k = 0; k<3; k++) {
					for (i = 0; i<7; i++) {
						for (j = 0; j<7; j++) {
							if (x + i <= 19 && y + j <= 19) {
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

void SW_activation_1(DATA_T I[4][7][7], DATA_T O[4][7][7]) {
	int m, x, y, i, j, k;
	for (m = 0; m<4; m++) {
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

void SW_max_pooling2d_1(DATA_T I[4][7][7], DATA_T O[4][3][3])
{
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<4; m++) {
		for (x = 0; x<3; x++) {
			for (y = 0; y<3; y++) {
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
void SW_res2a_branch2a(DATA_T I[4][3][3], DATA_T O[4][3][3], DATA_T W[4][4][1][1], DATA_T B[4]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<4; m++) {
		for (x = 0; x<3; x++) {
			for (y = 0; y<3; y++) {
				ofm = B[m];
				for (k = 0; k<4; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 3 && y + j <= 3) {
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

void SW_activation_2(DATA_T I[4][3][3], DATA_T O[4][3][3]) {
	int m, x, y, i, j, k;
	for (m = 0; m<4; m++) {
		for (x = 0; x<3; x++) {
			for (y = 0; y<3; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_res2a_branch2b(DATA_T I[4][3][3], DATA_T O[4][3][3], DATA_T W[4][4][3][3], DATA_T B[4]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(3 - 1) - 3 + 3)/2;
	for (m = 0; m<4; m++) {
		for (x = 0; x<3; x++) {
			for (y = 0; y<3; y++) {
				ofm = B[m];
				for (k = 0; k<4; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x*1 + i < 3 + p && y*1 + j < 3 + p && x*1 + i -p >= 0 && y*1 + j -p >= 0) {
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

void SW_activation_3(DATA_T I[4][3][3], DATA_T O[4][3][3]) {
	int m, x, y, i, j, k;
	for (m = 0; m<4; m++) {
		for (x = 0; x<3; x++) {
			for (y = 0; y<3; y++) {
				if (I[m][x][y] < 0)
					O[m][x][y] = 0;
				else
				    O[m][x][y] = I[m][x][y];
			}
		}
	}
}

void SW_res2a_branch2c(DATA_T I[4][3][3], DATA_T O[5][3][3], DATA_T W[5][4][1][1], DATA_T B[5]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<5; m++) {
		for (x = 0; x<3; x++) {
			for (y = 0; y<3; y++) {
				ofm = B[m];
				for (k = 0; k<4; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 3 && y + j <= 3) {
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

void SW_res2a_branch1(DATA_T I[4][3][3], DATA_T O[5][3][3], DATA_T W[5][4][1][1], DATA_T B[5]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<5; m++) {
		for (x = 0; x<3; x++) {
			for (y = 0; y<3; y++) {
				ofm = B[m];
				for (k = 0; k<4; k++) {
					for (i = 0; i<1; i++) {
						for (j = 0; j<1; j++) {
							if (x + i <= 3 && y + j <= 3) {
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

void SW_add_1(DATA_T I1[5][3][3], DATA_T I2[5][3][3], DATA_T O[5][3][3]) {
	int m, x, y;
	for (m = 0; m< 5; m++) {
		for (x = 0; x < 3; x++) {
			for(y = 0; y < 3; y++) {
				O[m][x][y] = I1[m][x][y] + I2[m][x][y];
			}
		}
	}
}


void resnet50(DATA_T I_i[3][13][13],DATA_T W2_i[4][3][7][7],DATA_T B2_i[4], DATA_T W5_i[4][4],DATA_T B5_i[4], DATA_T W7_i[4][4][3][3],DATA_T B7_i[4], DATA_T W9_i[5][4],DATA_T B9_i[5], DATA_T W10_i[5][4],DATA_T B10_i[5],  DATA_T O[5][3][3])  {
    #pragma HLS ARRAY_PARTITION variable=W2_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W2_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W2_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B2_i complete
#pragma HLS ARRAY_PARTITION variable=W5_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=B5_i complete
#pragma HLS ARRAY_PARTITION variable=W7_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=W7_i complete dim=3
#pragma HLS ARRAY_PARTITION variable=W7_i complete dim=4
#pragma HLS ARRAY_PARTITION variable=B7_i complete
#pragma HLS ARRAY_PARTITION variable=W9_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=B9_i complete
#pragma HLS ARRAY_PARTITION variable=W10_i complete dim=1
#pragma HLS ARRAY_PARTITION variable=B10_i complete


    #pragma HLS DATAFLOW

    	hls::stream<DATA_T> O0_strm("O0_strm");
	hls::stream<DATA_T> O1_strm("O1_strm");
	hls::stream<DATA_T> O2_strm("O2_strm");
	hls::stream<DATA_T> O3_strm("O3_strm");
	hls::stream<DATA_T> O4_strm("O4_strm");
	hls::stream<DATA_T> O5_strm("O5_strm");
	hls::stream<DATA_T> O6_strm("O6_strm");
	hls::stream<DATA_T> O7_strm("O7_strm");
	hls::stream<DATA_T> O8_strm("O8_strm");
	hls::stream<DATA_T> O9_strm("O9_strm");
	hls::stream<DATA_T> O10_strm("O10_strm");
	hls::stream<DATA_T> O11_strm("O11_strm");


    	Stream_input(I_i,O0_strm);
	HW_conv1_pad(O0_strm ,O1_strm);
	HW_conv1(O1_strm, W2_i, B2_i,O2_strm);
	HW_activation_1(O2_strm ,O3_strm);
	HW_max_pooling2d_1(O3_strm ,O4_strm);
	HW_res2a_branch2a(O4_strm, W5_i, B5_i,O5_strm);
	HW_activation_2(O5_strm ,O6_strm);
	HW_res2a_branch2b(O6_strm, W7_i, B7_i,O7_strm);
	HW_activation_3(O7_strm ,O8_strm);
	HW_res2a_branch2c(O8_strm, W9_i, B9_i,O9_strm);
	HW_res2a_branch1(O4_strm, W10_i, B10_i,O10_strm);
	HW_add_1(O9_strm, O10_strm, O11_strm);
	Stream_output(O11_strm,O);
	
}

void resnet50_top(DATA_T I[3][13][13],DATA_T W2[4][3][7][7],DATA_T B2[4], DATA_T W5[4][4][1][1],DATA_T B5[4], DATA_T W7[4][4][3][3],DATA_T B7[4], DATA_T W9[5][4][1][1],DATA_T B9[5], DATA_T W10[5][4][1][1],DATA_T B10[5],  DATA_T O[5][3][3]) {

    DATA_T O_i[5][3][3];
    int m, x, y, i, j, k;

    	I_i_k_loop: for (k=0; k<3; k++) {
  I_i_x_loop: for (x=0; x<13; x++) {
  I_i_y_loop: for (y=0; y<13; y++) {
  I_i[k][x][y] = I[k][x][y];
    }
  }
}
W2_i_m_loop: for (m=0; m<4; m++) {
  W2_i_k_loop: for (k=0; k<3; k++) {
  W2_i_i_loop: for (i=0; i<7; i++) {
  W2_i_j_loop: for (j=0; j<7; j++) {
  W2_i[m][k][i][j] = W2[m][k][i][j];
      }
    }
  }
}
B2_i_m_loop: for (m=0; m<4; m++) {
	B2_i[m] = B2[m];
}
W5_i_m_loop: for (m=0; m<4; m++) {
  W5_i_k_loop: for (k=0; k<4; k++) {
  W5_i[m][k] = W5[m][k];
  }
}
B5_i_m_loop: for (m=0; m<4; m++) {
	B5_i[m] = B5[m];
}
W7_i_m_loop: for (m=0; m<4; m++) {
  W7_i_k_loop: for (k=0; k<4; k++) {
  W7_i_i_loop: for (i=0; i<3; i++) {
  W7_i_j_loop: for (j=0; j<3; j++) {
  W7_i[m][k][i][j] = W7[m][k][i][j];
      }
    }
  }
}
B7_i_m_loop: for (m=0; m<4; m++) {
	B7_i[m] = B7[m];
}
W9_i_m_loop: for (m=0; m<5; m++) {
  W9_i_k_loop: for (k=0; k<4; k++) {
  W9_i[m][k] = W9[m][k];
  }
}
B9_i_m_loop: for (m=0; m<5; m++) {
	B9_i[m] = B9[m];
}
W10_i_m_loop: for (m=0; m<5; m++) {
  W10_i_k_loop: for (k=0; k<4; k++) {
  W10_i[m][k] = W10[m][k];
  }
}
B10_i_m_loop: for (m=0; m<5; m++) {
	B10_i[m] = B10[m];
}


    resnet50(I_i, W2_i, B2_i, W5_i, B5_i, W7_i, B7_i, W9_i, B9_i, W10_i, B10_i,  O_i);

    	for(m=0; m<5; m++) {
		for(x=0;x<3;x++){
			for(y=0;y<3;y++){ O[m][x][y] = O_i[m][x][y]; 
}}}

}

void resnet50_sw(DATA_T I[3][13][13],DATA_T W2[4][3][7][7],DATA_T B2[4], DATA_T W5[4][4][1][1],DATA_T B5[4], DATA_T W7[4][4][3][3],DATA_T B7[4], DATA_T W9[5][4][1][1],DATA_T B9[5], DATA_T W10[5][4][1][1],DATA_T B10[5],  DATA_T O11_SW[5][3][3]) {
    static DATA_T O1_SW[3][19][19];
	static DATA_T O2_SW[4][7][7];
	static DATA_T O3_SW[4][7][7];
	static DATA_T O4_SW[4][3][3];
	static DATA_T O5_SW[4][3][3];
	static DATA_T O6_SW[4][3][3];
	static DATA_T O7_SW[4][3][3];
	static DATA_T O8_SW[4][3][3];
	static DATA_T O9_SW[5][3][3];
	static DATA_T O10_SW[5][3][3];
	

    SW_conv1_pad(I,O1_SW);
	SW_conv1(O1_SW,O2_SW,W2,B2);
	SW_activation_1(O2_SW,O3_SW);
	SW_max_pooling2d_1(O3_SW,O4_SW);
	SW_res2a_branch2a(O4_SW,O5_SW,W5,B5);
	SW_activation_2(O5_SW,O6_SW);
	SW_res2a_branch2b(O6_SW,O7_SW,W7,B7);
	SW_activation_3(O7_SW,O8_SW);
	SW_res2a_branch2c(O8_SW,O9_SW,W9,B9);
	SW_res2a_branch1(O4_SW,O10_SW,W10,B10);
	SW_add_1(O9_SW,O10_SW,O11_SW);
	
}