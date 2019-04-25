#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <math.h>

using namespace std;

typedef float DATA_T;

// define functions of each layer
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
							if (x + i < 3 + p && y + j < 3 + p && x + i -p >= 0 && y + j -p >= 0) {
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


//argv[1] = init_weight.txt, argv[2] = init_input.txt
int main(int argc, char *argv[]){

    DATA_T temp;
    int m, x, y, i, j, k, l;
    int trash;

    // declare array variables of input, weight and bias (static variables)
    static DATA_T I[3][13][13];
	static DATA_T W2[4][3][7][7];
	static DATA_T B2[4];
	static DATA_T W5[4][4][1][1];
	static DATA_T B5[4];
	static DATA_T W7[4][4][3][3];
	static DATA_T B7[4];
	static DATA_T W9[5][4][1][1];
	static DATA_T B9[5];
	static DATA_T W10[5][4][1][1];
	static DATA_T B10[5];
	

    // declare array variables of output (static variables)
    static DATA_T O0_SW[3][13][13];
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
	static DATA_T O11_SW[5][3][3];
	

    FILE *w_stream = fopen(argv[1], "rb");
    if (w_stream == NULL) printf("weight file was not opened");
    FILE *i_stream = fopen(argv[2], "rb");
    if (i_stream == NULL) printf("input file was not opened");
    FILE *o_stream = fopen("./output/resnet50/output_value/c_output.txt", "w");
    if (o_stream == NULL) printf("Output file was not opened");
    FILE *c_num = fopen("./output/resnet50/output_value/c_output_num.txt", "w");
    if (c_num == NULL) printf("Output file was not opened");

    // initialize input, weight, bias variables using fread
    printf("[c_verifier.cpp]Start Initialzation\n");
    for (k = 0; k <  13 ; k++) {
	for (x = 0; x < 13 ; x++) {
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
			for (j = 0; j < 4 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W2[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 4 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B2[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 4 ; i++) {
			for (j = 0; j < 4 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W5[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 4 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B5[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 4 ; i++) {
			for (j = 0; j < 4 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W7[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 4 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B7[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 4 ; i++) {
			for (j = 0; j < 5 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W9[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 5 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B9[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 4 ; i++) {
			for (j = 0; j < 5 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W10[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 5 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B10[m] = (DATA_T) trash;
}

	
    printf("[c_verifier.cpp]Finish Initialization\n");

    // call function of each layer based on csv file which containts layer information
    printf("[c_verifier.cpp]InputLayer\n");
	printf("[c_verifier.cpp]Calculate ZeroPadding2D1\n");
	SW_conv1_pad(O0_SW,O1_SW);
	printf("[c_verifier.cpp]Calculate Conv2D2\n");
	SW_conv1(O1_SW,O2_SW,W2,B2);
	printf("[c_verifier.cpp]Calculate Activation(Relu)3\n");
	SW_activation_1(O2_SW,O3_SW);
	printf("[c_verifier.cpp]Calculate MaxPooling2D4\n");
	SW_max_pooling2d_1(O3_SW,O4_SW);
	printf("[c_verifier.cpp]Calculate Conv2D5\n");
	SW_res2a_branch2a(O4_SW,O5_SW,W5,B5);
	printf("[c_verifier.cpp]Calculate Activation(Relu)6\n");
	SW_activation_2(O5_SW,O6_SW);
	printf("[c_verifier.cpp]Calculate Conv2D7\n");
	SW_res2a_branch2b(O6_SW,O7_SW,W7,B7);
	printf("[c_verifier.cpp]Calculate Activation(Relu)8\n");
	SW_activation_3(O7_SW,O8_SW);
	printf("[c_verifier.cpp]Calculate Conv2D9\n");
	SW_res2a_branch2c(O8_SW,O9_SW,W9,B9);
	printf("[c_verifier.cpp]Calculate Conv2D10\n");
	SW_res2a_branch1(O4_SW,O10_SW,W10,B10);
	printf("[c_verifier.cpp]Calculate Add11\n");
	SW_add_1(O9_SW,O10_SW,O11_SW);
	

    // print each element of output variables
    printf("[c_verifier.cpp]Print Result\n");


    fprintf(o_stream,"%s","InputLayer : [[");
for (k = 0; k < 13 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 13 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 3 ; y++) {
			fprintf(o_stream,"%.6f ",O0_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O0_SW[y][k][x]);
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


	fprintf(o_stream,"%s","ZeroPadding2D : [[");
for (k = 0; k < 19 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 19 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 3 ; y++) {
			fprintf(o_stream,"%.6f ",O1_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O1_SW[y][k][x]);
		}
		if(x != 19 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 19 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 7 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 7 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 4 ; y++) {
			fprintf(o_stream,"%.6f ",O2_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O2_SW[y][k][x]);
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
		for(y = 0; y < 4 ; y++) {
			fprintf(o_stream,"%.6f ",O3_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O3_SW[y][k][x]);
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
for (k = 0; k < 3 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 3 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 4 ; y++) {
			fprintf(o_stream,"%.6f ",O4_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O4_SW[y][k][x]);
		}
		if(x != 3 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 3 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 3 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 3 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 4 ; y++) {
			fprintf(o_stream,"%.6f ",O5_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O5_SW[y][k][x]);
		}
		if(x != 3 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 3 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 3 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 3 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 4 ; y++) {
			fprintf(o_stream,"%.6f ",O6_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O6_SW[y][k][x]);
		}
		if(x != 3 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 3 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 3 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 3 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 4 ; y++) {
			fprintf(o_stream,"%.6f ",O7_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O7_SW[y][k][x]);
		}
		if(x != 3 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 3 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Activation : [[");
for (k = 0; k < 3 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 3 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 4 ; y++) {
			fprintf(o_stream,"%.6f ",O8_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O8_SW[y][k][x]);
		}
		if(x != 3 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 3 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 3 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 3 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 5 ; y++) {
			fprintf(o_stream,"%.6f ",O9_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O9_SW[y][k][x]);
		}
		if(x != 3 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 3 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 3 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 3 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 5 ; y++) {
			fprintf(o_stream,"%.6f ",O10_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O10_SW[y][k][x]);
		}
		if(x != 3 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 3 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Add : [[");
for (k = 0; k < 3 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 3 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 5 ; y++) {
			fprintf(o_stream,"%.6f ",O11_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O11_SW[y][k][x]);
		}
		if(x != 3 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 3 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	


    fclose(w_stream);
    fclose(i_stream);
    fclose(o_stream);
    fclose(c_num);

    return 0;
}
