#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <string.h>
#include <math.h>

using namespace std;

typedef float DATA_T;

// define functions of each layer
void SW_block1_conv1(DATA_T I[3][227][227], DATA_T O[96][55][55], DATA_T W[96][3][11][11], DATA_T B[96]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<96; m++) {
		for (x = 0; x<55; x++) {
			for (y = 0; y<55; y++) {
				ofm = B[m];
				for (k = 0; k<3; k++) {
					for (i = 0; i<11; i++) {
						for (j = 0; j<11; j++) {
							if (x + i <= 227 && y + j <= 227) {
								ifm = I[k][x*4 + i][y*4 + j];
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

void SW_block1_pool(DATA_T I[96][55][55], DATA_T O[96][27][27])
{
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<96; m++) {
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
void SW_block2_conv1(DATA_T I[96][27][27], DATA_T O[256][27][27], DATA_T W[256][96][5][5], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(27 - 1) - 27 + 5)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<27; x++) {
			for (y = 0; y<27; y++) {
				ofm = B[m];
				for (k = 0; k<96; k++) {
					for (i = 0; i<5; i++) {
						for (j = 0; j<5; j++) {
							if (x + i < 27 + p && y + j < 27 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block2_pool(DATA_T I[256][27][27], DATA_T O[256][13][13])
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
void SW_block3_conv1(DATA_T I[256][13][13], DATA_T O[384][13][13], DATA_T W[384][256][3][3], DATA_T B[384]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(13 - 1) - 13 + 3)/2;
	for (m = 0; m<384; m++) {
		for (x = 0; x<13; x++) {
			for (y = 0; y<13; y++) {
				ofm = B[m];
				for (k = 0; k<256; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 13 + p && y + j < 13 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block4_conv1(DATA_T I[384][13][13], DATA_T O[384][13][13], DATA_T W[384][384][3][3], DATA_T B[384]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(13 - 1) - 13 + 3)/2;
	for (m = 0; m<384; m++) {
		for (x = 0; x<13; x++) {
			for (y = 0; y<13; y++) {
				ofm = B[m];
				for (k = 0; k<384; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 13 + p && y + j < 13 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block5_conv1(DATA_T I[384][13][13], DATA_T O[256][13][13], DATA_T W[256][384][3][3], DATA_T B[256]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(13 - 1) - 13 + 3)/2;
	for (m = 0; m<256; m++) {
		for (x = 0; x<13; x++) {
			for (y = 0; y<13; y++) {
				ofm = B[m];
				for (k = 0; k<384; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 13 + p && y + j < 13 + p && x + i -p >= 0 && y + j -p >= 0) {
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

void SW_block5_pool(DATA_T I[256][13][13], DATA_T O[256][6][6])
{
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<256; m++) {
		for (x = 0; x<6; x++) {
			for (y = 0; y<6; y++) {
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
void SW_flatten(DATA_T I[256][6][6], DATA_T O[9216]){
	int i, j, x, y;
	i = 0;
	 for(x=0; x<6;x++)
		for(y=0; y<6; y++)
			for (j=0; j<256; j++) {
				O[i] = I[j][x][y];
				i++;
			}
}

void SW_fc1(DATA_T I[9216], DATA_T O[4096], DATA_T W[4096][9216])
{
    //Dense
	int m, c;
	for(m=0; m<4096; m++){
        O[m] = 0;
		for (c = 0; c < 9216; c++){
            O[m] += W[m][c] * I[c];
        }
        if (O[m] < 0) //Relu
            O[m] = 0;
     }
}
void SW_fc2(DATA_T I[4096], DATA_T O[4096], DATA_T W[4096][4096])
{
    //Dense
	int m, c;
	for(m=0; m<4096; m++){
        O[m] = 0;
		for (c = 0; c < 4096; c++){
            O[m] += W[m][c] * I[c];
        }
        if (O[m] < 0) //Relu
            O[m] = 0;
     }
}
void SW_predictions(DATA_T I[4096], DATA_T O[1000], DATA_T W[1000][4096])
{
    //Dense
	int m, c;
	DATA_T denom=0;
	DATA_T maximum;
	for(m=0; m<1000; m++){
        O[m] = 0;
		for (c = 0; c < 4096; c++){
			O[m] += W[m][c] * I[c];
         }
        denom+=O[m];

    }
    //Softmax

    maximum=O[0]/denom;
    for (m = 0; m < 1000; m++){
		O[m]=O[m]/denom;
		if(maximum<O[m])
		    maximum=O[m];
    }

    for (m = 0; m < 1000; m++){
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
    int trash;

    // declare array variables of input, weight and bias (static variables)
    static DATA_T I[3][227][227];
	static DATA_T W1[96][3][11][11];
	static DATA_T B1[96];
	static DATA_T W3[256][96][5][5];
	static DATA_T B3[256];
	static DATA_T W5[384][256][3][3];
	static DATA_T B5[384];
	static DATA_T W6[384][384][3][3];
	static DATA_T B6[384];
	static DATA_T W7[256][384][3][3];
	static DATA_T B7[256];
	static DATA_T W10[4096][9216];
	static DATA_T W11[4096][4096];
	static DATA_T W12[1000][4096];
	

    // declare array variables of output (static variables)
    static DATA_T O0_SW[3][227][227];
	static DATA_T O1_SW[96][55][55];
	static DATA_T O2_SW[96][27][27];
	static DATA_T O3_SW[256][27][27];
	static DATA_T O4_SW[256][13][13];
	static DATA_T O5_SW[384][13][13];
	static DATA_T O6_SW[384][13][13];
	static DATA_T O7_SW[256][13][13];
	static DATA_T O8_SW[256][6][6];
	static DATA_T O9_SW[9216];
	static DATA_T O10_SW[4096];
	static DATA_T O11_SW[4096];
	static DATA_T O12_SW[1000];
	

    FILE *w_stream = fopen(argv[1], "rb");
    if (w_stream == NULL) printf("weight file was not opened");
    FILE *i_stream = fopen(argv[2], "rb");
    if (i_stream == NULL) printf("input file was not opened");
    FILE *o_stream = fopen("./output/alexnet/output_value/c_output.txt", "w");
    if (o_stream == NULL) printf("Output file was not opened");
    FILE *c_num = fopen("./output/alexnet/output_value/c_output_num.txt", "w");
    if (c_num == NULL) printf("Output file was not opened");

    // initialize input, weight, bias variables using fread
    printf("[c_verifier.cpp]Start Initialzation\n");
    for (k = 0; k <  227 ; k++) {
	for (x = 0; x < 227 ; x++) {
		for(y = 0; y < 3 ; y++) {
			fread(&trash, sizeof(int), 1, i_stream);
            I[y][k][x] = (DATA_T) trash;
			O0_SW[y][k][x] = (DATA_T) trash;
		}
	}
}
	for (m = 0; m <  11 ; m++) {
	for (k = 0; k < 11 ; k++) {
		for (i = 0; i < 3 ; i++) {
			for (j = 0; j < 96 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W1[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 96 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B1[m] = (DATA_T) trash;
}

	for (m = 0; m <  5 ; m++) {
	for (k = 0; k < 5 ; k++) {
		for (i = 0; i < 96 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W3[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B3[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 384 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W5[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 384 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B5[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 384 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W6[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 384 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B6[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 384 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(int), 1, w_stream);
                W7[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B7[m] = (DATA_T) trash;
}

	for (m = 0; m <  9216 ; m++) {
	for (k = 0; k < 4096 ; k++) {
		fread(&trash, sizeof(int), 1, w_stream);
        W10[k][m] = (DATA_T) trash;
	}
}
/*
for (m = 0; m < 4096 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B10[m] = (DATA_T) trash;
}
*/
	for (m = 0; m <  4096 ; m++) {
	for (k = 0; k < 4096 ; k++) {
		fread(&trash, sizeof(int), 1, w_stream);
        W11[k][m] = (DATA_T) trash;
	}
}
/*
for (m = 0; m < 4096 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B11[m] = (DATA_T) trash;
}
*/
	for (m = 0; m <  4096 ; m++) {
	for (k = 0; k < 1000 ; k++) {
		fread(&trash, sizeof(int), 1, w_stream);
        W12[k][m] = (DATA_T) trash;
	}
}
/*
for (m = 0; m < 1000 ; m++) {
    fread(&trash, sizeof(int), 1, w_stream);
    B12[m] = (DATA_T) trash;
}
*/
	
    printf("[c_verifier.cpp]Finish Initialization\n");

    // call function of each layer based on csv file which containts layer information
    printf("[c_verifier.cpp]InputLayer\n");
	printf("[c_verifier.cpp]Calculate Conv2D1\n");
	SW_block1_conv1(O0_SW,O1_SW,W1,B1);
	printf("[c_verifier.cpp]Calculate MaxPooling2D2\n");
	SW_block1_pool(O1_SW,O2_SW);
	printf("[c_verifier.cpp]Calculate Conv2D3\n");
	SW_block2_conv1(O2_SW,O3_SW,W3,B3);
	printf("[c_verifier.cpp]Calculate MaxPooling2D4\n");
	SW_block2_pool(O3_SW,O4_SW);
	printf("[c_verifier.cpp]Calculate Conv2D5\n");
	SW_block3_conv1(O4_SW,O5_SW,W5,B5);
	printf("[c_verifier.cpp]Calculate Conv2D6\n");
	SW_block4_conv1(O5_SW,O6_SW,W6,B6);
	printf("[c_verifier.cpp]Calculate Conv2D7\n");
	SW_block5_conv1(O6_SW,O7_SW,W7,B7);
	printf("[c_verifier.cpp]Calculate MaxPooling2D8\n");
	SW_block5_pool(O7_SW,O8_SW);
	printf("[c_verifier.cpp]Calculate Flatten9\n");
	SW_flatten(O8_SW,O9_SW);
	printf("[c_verifier.cpp]Calculate Dense10\n");
	SW_fc1(O9_SW,O10_SW,W10);
	printf("[c_verifier.cpp]Calculate Dense11\n");
	SW_fc2(O10_SW,O11_SW,W11);
	printf("[c_verifier.cpp]Calculate Dense12\n");
	SW_predictions(O11_SW,O12_SW,W12);
	

    // print each element of output variables
    printf("[c_verifier.cpp]Print Result\n");


    fprintf(o_stream,"%s","InputLayer : [[");
for (k = 0; k < 227 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 227 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 3 ; y++) {
			fprintf(o_stream,"%.6f ",O0_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O0_SW[y][k][x]);
		}
		if(x != 227 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 227 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Conv2D : [[");
for (k = 0; k < 55 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 55 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 96 ; y++) {
			fprintf(o_stream,"%.6f ",O1_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O1_SW[y][k][x]);
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
		for(y = 0; y < 96 ; y++) {
			fprintf(o_stream,"%.6f ",O2_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O2_SW[y][k][x]);
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
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O3_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O3_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O4_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O4_SW[y][k][x]);
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
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O5_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O5_SW[y][k][x]);
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
		for(y = 0; y < 384 ; y++) {
			fprintf(o_stream,"%.6f ",O6_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O6_SW[y][k][x]);
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
			fprintf(o_stream,"%.6f ",O7_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O7_SW[y][k][x]);
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


	fprintf(o_stream,"%s","MaxPooling2D : [[");
for (k = 0; k < 6 ; k++) {
	fprintf(o_stream,"%s","[");
	for (x = 0; x < 6 ; x++) {
		fprintf(o_stream,"%s","[");
		for(y = 0; y < 256 ; y++) {
			fprintf(o_stream,"%.6f ",O8_SW[y][k][x]);
			fprintf(c_num,"%.6f ",O8_SW[y][k][x]);
		}
		if(x != 6 -1 )
			fprintf(o_stream,"%s","]\n   ");
		else
			fprintf(o_stream,"%s","]");
	}
	if(k != 6 -1 )
		fprintf(o_stream,"%s","]\n\n  ");
}
fprintf(o_stream,"%s","]]]\n\n");


	fprintf(o_stream,"%s","Flatten : [[");
for (k = 0; k <  9216 ; k++) {
	fprintf(o_stream,"%.6f ",O9_SW[k]);
	fprintf(c_num,"%.6f ",O9_SW[k]);
}
fprintf(o_stream,"%s","]]\n\n");


	fprintf(o_stream,"%s","Dense : [[");
for (k = 0; k <  4096 ; k++) {
	fprintf(o_stream,"%.6f ",O10_SW[k]);
	fprintf(c_num,"%.6f ",O10_SW[k]);
}
fprintf(o_stream,"%s","]]\n\n");


	fprintf(o_stream,"%s","Dense : [[");
for (k = 0; k <  4096 ; k++) {
	fprintf(o_stream,"%.6f ",O11_SW[k]);
	fprintf(c_num,"%.6f ",O11_SW[k]);
}
fprintf(o_stream,"%s","]]\n\n");


	fprintf(o_stream,"%s","Dense : [[");
for (k = 0; k <  1000 ; k++) {
	fprintf(o_stream,"%.6f ",O12_SW[k]);
	fprintf(c_num,"%.6f ",O12_SW[k]);
}
fprintf(o_stream,"%s","]]\n\n");


	


    fclose(w_stream);
    fclose(i_stream);
    fclose(o_stream);
    fclose(c_num);

    return 0;
}
