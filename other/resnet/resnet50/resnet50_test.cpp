#include <iostream>
#include <ap_int.h>
#include "hls_stream.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
using namespace std;

typedef int DATA_IN;
typedef int DATA_T;

void resnet50_top(DATA_T I[3][13][13], DATA_T W1[4][3][7][7], DATA_T B1[4], DATA_T W2[4][4][1][1], DATA_T B2[4], DATA_T W3[4][4][3][3], DATA_T B3[4], DATA_T W4[5][4][1][1], DATA_T B4[5], DATA_T W5[5][4][1][1], DATA_T B5[5], DATA_T O[5][3][3]);
void resnet50_sw(DATA_T I[3][13][13], DATA_T W1[4][3][7][7], DATA_T B1[4], DATA_T W2[4][4][1][1], DATA_T B2[4], DATA_T W3[4][4][3][3], DATA_T B3[4], DATA_T W4[5][4][1][1], DATA_T B4[5], DATA_T W5[5][4][1][1], DATA_T B5[5], DATA_T O[5][3][3]);

int main()
{
	int m, x, y, i, j, k;
	DATA_T temp;
	int trash;

	DATA_T I[3][13][13];
	DATA_T W1[4][3][7][7];
	DATA_T B1[4];
	DATA_T W2[4][4][1][1];
	DATA_T B2[4];
	DATA_T W3[4][4][3][3];
	DATA_T B3[4];
	DATA_T W4[5][4][1][1];
	DATA_T B4[5];
	DATA_T W5[5][4][1][1];
	DATA_T B5[5];
	DATA_T O_SW[5][3][3];
	DATA_T O_HW[5][3][3];

	FILE *w_stream = fopen("init_Weight.bin", "r");
	if (w_stream == NULL) printf("weight file was not opened");
	FILE *i_stream = fopen("init_Input.bin", "r");
	if (i_stream == NULL) printf("input file was not opened");

	//initialization
	for (k = 0; k < 13; k++) {
		for (x = 0; x < 13; x++) {
			for (y = 0; y < 3; y++) {
				fread(&trash, sizeof(int), 1, i_stream);
				I[y][k][x] = (DATA_T)trash;
			}
		}
	}

	for (m = 0; m < 7; m++) {
		for (k = 0; k < 7; k++) {
			for (i = 0; i < 3; i++) {
				for (j = 0; j < 4; j++) {
					fread(&trash, sizeof(int), 1, w_stream);
					W1[j][i][m][k] = (DATA_T)trash;
				}
			}
		}
	}

	for (m = 0; m < 4; m++) {
		fread(&trash, sizeof(int), 1, w_stream);
		B1[m] = (DATA_T)trash;
	}

	for (m = 0; m < 1; m++) {
		for (k = 0; k < 1; k++) {
			for (i = 0; i < 4; i++) {
				for (j = 0; j < 4; j++) {
					fread(&trash, sizeof(int), 1, w_stream);
					W2[j][i][m][k] = (DATA_T)trash;
				}
			}
		}
	}

	for (m = 0; m < 4; m++) {
		fread(&trash, sizeof(int), 1, w_stream);
		B2[m] = (DATA_T)trash;
	}

	for (m = 0; m < 3; m++) {
		for (k = 0; k < 3; k++) {
			for (i = 0; i < 4; i++) {
				for (j = 0; j < 4; j++) {
					fread(&trash, sizeof(int), 1, w_stream);
					W3[j][i][m][k] = (DATA_T)trash;
				}
			}
		}
	}

	for (m = 0; m < 4; m++) {
		fread(&trash, sizeof(int), 1, w_stream);
		B3[m] = (DATA_T)trash;
	}

	for (m = 0; m < 1; m++) {
		for (k = 0; k < 1; k++) {
			for (i = 0; i < 4; i++) {
				for (j = 0; j < 5; j++) {
					fread(&trash, sizeof(int), 1, w_stream);
					W4[j][i][m][k] = (DATA_T)trash;
				}
			}
		}
	}

	for (m = 0; m < 5; m++) {
		fread(&trash, sizeof(int), 1, w_stream);
		B4[m] = (DATA_T)trash;
	}

	for (m = 0; m < 1; m++) {
		for (k = 0; k < 1; k++) {
			for (i = 0; i < 4; i++) {
				for (j = 0; j < 5; j++) {
					fread(&trash, sizeof(int), 1, w_stream);
					W5[j][i][m][k] = (DATA_T)trash;
				}
			}
		}
	}

	for (m = 0; m < 5; m++) {
		fread(&trash, sizeof(int), 1, w_stream);
		B5[m] = (DATA_T)trash;
	}

	resnet50_top(I,W1,B1,W2,B2,W3,B3,W4,B4,W5,B5,O_HW);
	resnet50_sw(I, W1, B1, W2, B2, W3, B3, W4, B4, W5, B5, O_SW);

	int err_cnt = 0;
	for (m = 0; m<5; m++) {
		for (x = 0; x<3; x++) {
			for (y = 0; y<3; y++) {
				cout << "SW: O[" << m << "][" << x << "][" << y << "] = " << O_SW[m][x][y];
				cout << " HW: O[" << m << "][" << x << "][" << y << "] = " << O_HW[m][x][y];
				cout<<endl;
				if (O_HW[m][x][y] != O_SW[m][x][y]) {
					err_cnt++;
				}
			}
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
