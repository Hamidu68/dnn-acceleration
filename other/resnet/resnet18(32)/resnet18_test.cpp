#include <iostream>
#include <ap_int.h>
#include "hls_stream.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
using namespace std;


typedef ap_int<32> DATA_IN;
typedef ap_int<80> DATA_T;

void resnet18_top(DATA_T I[3][32][32], DATA_T W1[64][3][3][3], DATA_T B1[64], DATA_T W3[64][64][3][3], DATA_T B3[64], DATA_T W4[64][64][3][3], DATA_T B4[64], DATA_T W5[64][64][1][1], DATA_T B5[64], DATA_T W8[64][64][3][3], DATA_T B8[64], DATA_T W10[64][64][3][3], DATA_T B10[64], DATA_T W13[128][64][3][3], DATA_T B13[128], DATA_T W15[128][128][3][3], DATA_T B15[128], DATA_T W16[128][64][1][1], DATA_T B16[128], DATA_T W19[128][128][3][3], DATA_T B19[128], DATA_T W21[128][128][3][3], DATA_T B21[128], DATA_T W24[256][128][3][3], DATA_T B24[256], DATA_T W26[256][256][3][3], DATA_T B26[256], DATA_T W27[256][128][1][1], DATA_T B27[256], DATA_T W30[256][256][3][3], DATA_T B30[256], DATA_T W32[256][256][3][3], DATA_T B32[256], DATA_T W35[512][256][3][3], DATA_T B35[512], DATA_T W37[512][512][3][3], DATA_T B37[512], DATA_T W38[512][256][1][1], DATA_T B38[512], DATA_T W41[512][512][3][3], DATA_T B41[512], DATA_T W43[512][512][3][3], DATA_T B43[512], DATA_T W48[10][512], DATA_T B48[10], DATA_T O[10]);
void resnet18_sw(DATA_T I[3][32][32], DATA_T W1[64][3][3][3], DATA_T B1[64], DATA_T W3[64][64][3][3], DATA_T B3[64], DATA_T W4[64][64][3][3], DATA_T B4[64], DATA_T W5[64][64][1][1], DATA_T B5[64], DATA_T W8[64][64][3][3], DATA_T B8[64], DATA_T W10[64][64][3][3], DATA_T B10[64], DATA_T W13[128][64][3][3], DATA_T B13[128], DATA_T W15[128][128][3][3], DATA_T B15[128], DATA_T W16[128][64][1][1], DATA_T B16[128], DATA_T W19[128][128][3][3], DATA_T B19[128], DATA_T W21[128][128][3][3], DATA_T B21[128], DATA_T W24[256][128][3][3], DATA_T B24[256], DATA_T W26[256][256][3][3], DATA_T B26[256], DATA_T W27[256][128][1][1], DATA_T B27[256], DATA_T W30[256][256][3][3], DATA_T B30[256], DATA_T W32[256][256][3][3], DATA_T B32[256], DATA_T W35[512][256][3][3], DATA_T B35[512], DATA_T W37[512][512][3][3], DATA_T B37[512], DATA_T W38[512][256][1][1], DATA_T B38[512], DATA_T W41[512][512][3][3], DATA_T B41[512], DATA_T W43[512][512][3][3], DATA_T B43[512], DATA_T W48[10][512], DATA_T B48[10], DATA_T O[10]);

int main() {

	int m, x, y, i, j, k;
	DATA_IN trash;

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
	static DATA_T O_SW[10];
	static DATA_T O_HW[10];


	FILE *w_stream = fopen("init_weight.bin", "r");
	if (w_stream == NULL) printf("weight file was not opened");
	FILE *i_stream = fopen("init_input.bin", "r");
	if (i_stream == NULL) printf("input file was not opened");

	for (k = 0; k < 32; k++) {
		for (x = 0; x < 32; x++) {
			for (y = 0; y < 3; y++) {
				fread(&trash, sizeof(DATA_IN), 1, i_stream);
				I[y][k][x] = (DATA_T)trash;
			}
		}
	}
	for (m = 0; m < 3; m++) {
		for (k = 0; k < 3; k++) {
			for (i = 0; i < 3; i++) {
				for (j = 0; j < 64; j++) {
					fread(&trash, sizeof(DATA_IN), 1, w_stream);
					W1[j][i][m][k] = (DATA_T)trash;
				}
			}
		}
	}

	for (m = 0; m < 64; m++) {
		fread(&trash, sizeof(DATA_IN), 1, w_stream);
		B1[m] = (DATA_T)trash;
	}

	for (m = 0; m < 3; m++) {
		for (k = 0; k < 3; k++) {
			for (i = 0; i < 64; i++) {
				for (j = 0; j < 64; j++) {
					fread(&trash, sizeof(DATA_IN), 1, w_stream);
					W3[j][i][m][k] = (DATA_T)trash;
				}
			}
		}
	}

	for (m = 0; m < 64; m++) {
		fread(&trash, sizeof(DATA_IN), 1, w_stream);
		B3[m] = (DATA_T)trash;
	}

	for (m = 0; m < 3; m++) {
		for (k = 0; k < 3; k++) {
			for (i = 0; i < 64; i++) {
				for (j = 0; j < 64; j++) {
					fread(&trash, sizeof(DATA_IN), 1, w_stream);
					W4[j][i][m][k] = (DATA_T)trash;
				}
			}
		}
	}

	for (m = 0; m < 64; m++) {
		fread(&trash, sizeof(DATA_IN), 1, w_stream);
		B4[m] = (DATA_T)trash;
	}

	for (m = 0; m < 1; m++) {
		for (k = 0; k < 1; k++) {
			for (i = 0; i < 64; i++) {
				for (j = 0; j < 64; j++) {
					fread(&trash, sizeof(DATA_IN), 1, w_stream);
					W5[j][i][m][k] = (DATA_T)trash;
				}
			}
		}
	}

	for (m = 0; m < 64; m++) {
		fread(&trash, sizeof(DATA_IN), 1, w_stream);
		B5[m] = (DATA_T)trash;
	}

	for (m = 0; m < 3; m++) {
		for (k = 0; k < 3; k++) {
			for (i = 0; i < 64; i++) {
				for (j = 0; j < 64; j++) {
					fread(&trash, sizeof(DATA_IN), 1, w_stream);
					W8[j][i][m][k] = (DATA_T)trash;
				}
			}
		}
	}

	for (m = 0; m < 64; m++) {
		fread(&trash, sizeof(DATA_IN), 1, w_stream);
		B8[m] = (DATA_T)trash;
	}

	for (m = 0; m < 3; m++) {
		for (k = 0; k < 3; k++) {
			for (i = 0; i < 64; i++) {
				for (j = 0; j < 64; j++) {
					fread(&trash, sizeof(DATA_IN), 1, w_stream);
					W10[j][i][m][k] = (DATA_T)trash;
				}
			}
		}
	}

	for (m = 0; m < 64; m++) {
		fread(&trash, sizeof(DATA_IN), 1, w_stream);
		B10[m] = (DATA_T)trash;
	}

	for (m = 0; m < 3; m++) {
		for (k = 0; k < 3; k++) {
			for (i = 0; i < 64; i++) {
				for (j = 0; j < 128; j++) {
					fread(&trash, sizeof(DATA_IN), 1, w_stream);
					W13[j][i][m][k] = (DATA_T)trash;
				}
			}
		}
	}

	for (m = 0; m < 128; m++) {
		fread(&trash, sizeof(DATA_IN), 1, w_stream);
		B13[m] = (DATA_T)trash;
	}

	for (m = 0; m < 3; m++) {
		for (k = 0; k < 3; k++) {
			for (i = 0; i < 128; i++) {
				for (j = 0; j < 128; j++) {
					fread(&trash, sizeof(DATA_IN), 1, w_stream);
					W15[j][i][m][k] = (DATA_T)trash;
				}
			}
		}
	}

	for (m = 0; m < 128; m++) {
		fread(&trash, sizeof(DATA_IN), 1, w_stream);
		B15[m] = (DATA_T)trash;
	}

	for (m = 0; m < 1; m++) {
		for (k = 0; k < 1; k++) {
			for (i = 0; i < 64; i++) {
				for (j = 0; j < 128; j++) {
					fread(&trash, sizeof(DATA_IN), 1, w_stream);
					W16[j][i][m][k] = (DATA_T)trash;
				}
			}
		}
	}

	for (m = 0; m < 128; m++) {
		fread(&trash, sizeof(DATA_IN), 1, w_stream);
		B16[m] = (DATA_T)trash;
	}

	for (m = 0; m < 3; m++) {
		for (k = 0; k < 3; k++) {
			for (i = 0; i < 128; i++) {
				for (j = 0; j < 128; j++) {
					fread(&trash, sizeof(DATA_IN), 1, w_stream);
					W19[j][i][m][k] = (DATA_T)trash;
				}
			}
		}
	}

	for (m = 0; m < 128; m++) {
		fread(&trash, sizeof(DATA_IN), 1, w_stream);
		B19[m] = (DATA_T)trash;
	}

	for (m = 0; m < 3; m++) {
		for (k = 0; k < 3; k++) {
			for (i = 0; i < 128; i++) {
				for (j = 0; j < 128; j++) {
					fread(&trash, sizeof(DATA_IN), 1, w_stream);
					W21[j][i][m][k] = (DATA_T)trash;
				}
			}
		}
	}

	for (m = 0; m < 128; m++) {
		fread(&trash, sizeof(DATA_IN), 1, w_stream);
		B21[m] = (DATA_T)trash;
	}

	for (m = 0; m < 3; m++) {
		for (k = 0; k < 3; k++) {
			for (i = 0; i < 128; i++) {
				for (j = 0; j < 256; j++) {
					fread(&trash, sizeof(DATA_IN), 1, w_stream);
					W24[j][i][m][k] = (DATA_T)trash;
				}
			}
		}
	}

	for (m = 0; m < 256; m++) {
		fread(&trash, sizeof(DATA_IN), 1, w_stream);
		B24[m] = (DATA_T)trash;
	}

	for (m = 0; m < 3; m++) {
		for (k = 0; k < 3; k++) {
			for (i = 0; i < 256; i++) {
				for (j = 0; j < 256; j++) {
					fread(&trash, sizeof(DATA_IN), 1, w_stream);
					W26[j][i][m][k] = (DATA_T)trash;
				}
			}
		}
	}

	for (m = 0; m < 256; m++) {
		fread(&trash, sizeof(DATA_IN), 1, w_stream);
		B26[m] = (DATA_T)trash;
	}

	for (m = 0; m < 1; m++) {
		for (k = 0; k < 1; k++) {
			for (i = 0; i < 128; i++) {
				for (j = 0; j < 256; j++) {
					fread(&trash, sizeof(DATA_IN), 1, w_stream);
					W27[j][i][m][k] = (DATA_T)trash;
				}
			}
		}
	}

	for (m = 0; m < 256; m++) {
		fread(&trash, sizeof(DATA_IN), 1, w_stream);
		B27[m] = (DATA_T)trash;
	}

	for (m = 0; m < 3; m++) {
		for (k = 0; k < 3; k++) {
			for (i = 0; i < 256; i++) {
				for (j = 0; j < 256; j++) {
					fread(&trash, sizeof(DATA_IN), 1, w_stream);
					W30[j][i][m][k] = (DATA_T)trash;
				}
			}
		}
	}

	for (m = 0; m < 256; m++) {
		fread(&trash, sizeof(DATA_IN), 1, w_stream);
		B30[m] = (DATA_T)trash;
	}

	for (m = 0; m < 3; m++) {
		for (k = 0; k < 3; k++) {
			for (i = 0; i < 256; i++) {
				for (j = 0; j < 256; j++) {
					fread(&trash, sizeof(DATA_IN), 1, w_stream);
					W32[j][i][m][k] = (DATA_T)trash;
				}
			}
		}
	}

	for (m = 0; m < 256; m++) {
		fread(&trash, sizeof(DATA_IN), 1, w_stream);
		B32[m] = (DATA_T)trash;
	}

	for (m = 0; m < 3; m++) {
		for (k = 0; k < 3; k++) {
			for (i = 0; i < 256; i++) {
				for (j = 0; j < 512; j++) {
					fread(&trash, sizeof(DATA_IN), 1, w_stream);
					W35[j][i][m][k] = (DATA_T)trash;
				}
			}
		}
	}

	for (m = 0; m < 512; m++) {
		fread(&trash, sizeof(DATA_IN), 1, w_stream);
		B35[m] = (DATA_T)trash;
	}

	for (m = 0; m < 3; m++) {
		for (k = 0; k < 3; k++) {
			for (i = 0; i < 512; i++) {
				for (j = 0; j < 512; j++) {
					fread(&trash, sizeof(DATA_IN), 1, w_stream);
					W37[j][i][m][k] = (DATA_T)trash;
				}
			}
		}
	}

	for (m = 0; m < 512; m++) {
		fread(&trash, sizeof(DATA_IN), 1, w_stream);
		B37[m] = (DATA_T)trash;
	}

	for (m = 0; m < 1; m++) {
		for (k = 0; k < 1; k++) {
			for (i = 0; i < 256; i++) {
				for (j = 0; j < 512; j++) {
					fread(&trash, sizeof(DATA_IN), 1, w_stream);
					W38[j][i][m][k] = (DATA_T)trash;
				}
			}
		}
	}

	for (m = 0; m < 512; m++) {
		fread(&trash, sizeof(DATA_IN), 1, w_stream);
		B38[m] = (DATA_T)trash;
	}

	for (m = 0; m < 3; m++) {
		for (k = 0; k < 3; k++) {
			for (i = 0; i < 512; i++) {
				for (j = 0; j < 512; j++) {
					fread(&trash, sizeof(DATA_IN), 1, w_stream);
					W41[j][i][m][k] = (DATA_T)trash;
				}
			}
		}
	}

	for (m = 0; m < 512; m++) {
		fread(&trash, sizeof(DATA_IN), 1, w_stream);
		B41[m] = (DATA_T)trash;
	}

	for (m = 0; m < 3; m++) {
		for (k = 0; k < 3; k++) {
			for (i = 0; i < 512; i++) {
				for (j = 0; j < 512; j++) {
					fread(&trash, sizeof(DATA_IN), 1, w_stream);
					W43[j][i][m][k] = (DATA_T)trash;
				}
			}
		}
	}

	for (m = 0; m < 512; m++) {
		fread(&trash, sizeof(DATA_IN), 1, w_stream);
		B43[m] = (DATA_T)trash;
	}

	for (m = 0; m < 512; m++) {
		for (k = 0; k < 10; k++) {
			fread(&trash, sizeof(DATA_IN), 1, w_stream);
			W48[k][m] = (DATA_T)trash;
		}
	}

	for (m = 0; m < 10; m++) {
		fread(&trash, sizeof(DATA_IN), 1, w_stream);
		B48[m] = (DATA_T)trash;
	}


	resnet18_sw(I, W1, B1, W3, B3, W4, B4, W5, B5, W8, B8, W10, B10, W13, B13, W15, B15, W16, B16, W19, B19, W21, B21, W24, B24, W26, B26, W27, B27, W30, B30, W32, B32, W35, B35, W37, B37, W38, B38, W41, B41, W43, B43, W48, B48, O_SW);
	resnet18_top(I, W1, B1, W3, B3, W4, B4, W5, B5, W8, B8, W10, B10, W13, B13, W15, B15, W16, B16, W19, B19, W21, B21, W24, B24, W26, B26, W27, B27, W30, B30, W32, B32, W35, B35, W37, B37, W38, B38, W41, B41, W43, B43, W48, B48, O_HW);

	int err_cnt = 0;
	for (m = 0; m<10; m++) {
		cout << "SW: O[" << m << "] = " << O_SW[m];
		cout << "HW: O[" << m << "] = " << O_HW[m];
		if (O_HW[m] != O_SW[m]) {
			err_cnt++;
		}
	}

	int ret_val;
	if (err_cnt == 0) {
		printf("*** TEST PASSED ***\n");
		ret_val = 0;
	}
	else {
		printf("!!! TEST FAILED - %d mismatches detected !!!\n\n", err_cnt);
		ret_val = -1;
	}

	fclose(w_stream);
	fclose(i_stream);
	return ret_val;
}
