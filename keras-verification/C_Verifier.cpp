#include <stdio.h>
#include<iostream>
#include <stdlib.h>
#include<string>
#include<math.h>
using namespace std;

typedef float DATA_T;

void SW_block1_conv1(DATA_T I[3][10][10], DATA_T O[4][8][8], DATA_T B[4], DATA_T W[4][3][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
	for (m = 0; m<4; m++) {
		for (x = 0; x<8; x++) {
			for (y = 0; y<8; y++) {
				ofm = B[m];
				for (k = 0; k<3; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 10 && y + j < 10) { 
								ifm = I[k][x*1 + i][y*1 + j];
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

void SW_bn2a_branch2a(DATA_T I[4][8][8], DATA_T O[4][8][8]) {
	int m, x, y;
	DATA_T temp;
	DATA_T mean = 0;
	DATA_T dsd_dev = 1;
	DATA_T size = 4 * 8 * 8;
	DATA_T epsilon = 1e-3;
for (m = 0; m < 4; m++){
		for (x = 0; x < 8; x++){
			for (y = 0; y < 8; y++){
				O[m][x][y] = (I[m][x][y]-mean)/sqrt(dsd_dev+epsilon);
			}
		}
	}
}
void SW_block2_conv1(DATA_T I[4][8][8], DATA_T O[8][8][8], DATA_T B[8], DATA_T W[8][4][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(8 - 1) - 8 + 3)/2; 
	for (m = 0; m<8; m++) {
		for (x = 0; x<8; x++) {
			for (y = 0; y<8; y++) {
				ofm = B[m];
				for (k = 0; k<4; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 8 + p && y + j < 8 + p && x + i -p >= 0 && y + j -p >= 0) { 
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

void SW_block1_pool(DATA_T I[8][8][8], DATA_T O[8][7][7])
{ 
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<8; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				max = I[m][x*1][y*1];
				for (i = 0; i<2; i++) {
					for (j = 0; j<2; j++) {
						if (I[m][x*1 + i][y*1 + j] > max) {
							max = I[m][x*1 + i][y*1 + j];
						}
					}
				}
				O[m][x][y] = max;
			}
		}
	}
}
void SW_activation_2(DATA_T I[8][7][7], DATA_T O[8][7][7])
{
	int m, x, y;
	DATA_T ifm;
	for (m = 0; m<8; m++) {
		for (x = 0; x<7; x++) {
			for (y = 0; y<7; y++) {
				ifm = I[m][x][y];
				if (ifm < 0)
					O[m][x][y] = 0;
				else
					O[m][x][y] = ifm;
			}
		}
	}
}
void SW_avg_pool(DATA_T I[8][7][7], DATA_T O[8][1][1])
{ 
	int m, x, y, i, j;
	DATA_T sum;
    DATA_T div = 7*7 ; 
	for (m = 0; m<8; m++) {
		for (x = 0; x<1; x++) {
			for (y = 0; y<1; y++) {
                sum = 0;
				for (i = 0; i<7; i++) {
					for (j = 0; j<7; j++) {
						sum += I[m][x*7 + i][y*7 + j];
					}
				}
				O[m][x][y] = sum/div;
			}
		}
	}
}


//argv[1] = init_weight.txt , argv[2] = init_bias.txt , argv[3] = init_input.txt
int main(int argc, char *argv[]){
 
  DATA_T temp;
  int m, x, y, i, j, k, l;
  
  static DATA_T I[3][10][10];
	static DATA_T W1[4][3][3][3];
	static DATA_T B1[4];
	static DATA_T W3[8][4][3][3];
	static DATA_T B3[8];
	
 
  static DATA_T O0_SW[3][10][10];
	static DATA_T O1_SW[4][8][8];
	static DATA_T O2_SW[4][8][8];
	static DATA_T O3_SW[8][8][8];
	static DATA_T O4_SW[8][7][7];
	static DATA_T O5_SW[8][7][7];
	static DATA_T O6_SW[8][1][1];
	 

  FILE *w_stream = fopen(argv[1], "r");
  if (w_stream == NULL) printf("weight file was not opened");
  FILE *b_stream = fopen(argv[2], "r");
  if (b_stream == NULL) printf("bias file was not opened");
  FILE *i_stream = fopen(argv[3], "r");
  if (i_stream == NULL) printf("i_stream file was not opened");
  FILE *o_stream = fopen("Output/C_output.txt", "w");
  if (o_stream == NULL) printf("Output file was not opened");
  
  for (k = 0; k <  3 ; k++) {
	for (x = 0; x < 10 ; x++) {
		for(y = 0; y < 10 ; y++) {
			fscanf(i_stream, "%f", &temp);
			I[k][x][y] = temp;
			O0_SW[k][x][y] = temp;
		}
	}
}

	for (m = 0; m <  4 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 3 ; i++) {
			for (j = 0; j < 3 ; j++) {
				fscanf(w_stream, "%f", &temp);
				W1[m][k][i][j] = temp;
			}
		}
	}
	fscanf(b_stream, "%f", &temp);
	B1[m] = temp;      
}

	for (m = 0; m <  8 ; m++) {
	for (k = 0; k < 4 ; k++) {
		for (i = 0; i < 3 ; i++) {
			for (j = 0; j < 3 ; j++) {
				fscanf(w_stream, "%f", &temp);
				W3[m][k][i][j] = temp;
			}
		}
	}
	fscanf(b_stream, "%f", &temp);
	B3[m] = temp;      
}

	
 
  SW_block1_conv1(O0_SW,O1_SW,B1,W1);
	SW_bn2a_branch2a(O1_SW,O2_SW);
	SW_block2_conv1(O2_SW,O3_SW,B3,W3);
	SW_block1_pool(O3_SW,O4_SW);
	SW_activation_2(O4_SW,O5_SW);
	SW_avg_pool(O5_SW,O6_SW);
	
  
  string show_result;
  
  show_result= show_result + "InputLayer" + " : [[";
for (k = 0; k <  3 ; k++) {
	show_result += "[";
	for (x = 0; x < 10 ; x++) {
		show_result += "[";
		for(y = 0; y < 10 ; y++) {
			show_result = show_result + " " + to_string((int)(O0_SW[k][x][y]));
		}
		show_result += "]";
		if(x != 10 -1 )
			show_result += "\n   ";
	}
	if(k != 3 -1 )
		show_result += "]\n\n  ";
}
show_result += "]]]\n\n";

show_result= show_result + "Convolution2D" + " : [[";
for (k = 0; k <  4 ; k++) {
	show_result += "[";
	for (x = 0; x < 8 ; x++) {
		show_result += "[";
		for(y = 0; y < 8 ; y++) {
			show_result = show_result + " " + to_string((int)(O1_SW[k][x][y]));
		}
		show_result += "]";
		if(x != 8 -1 )
			show_result += "\n   ";
	}
	if(k != 4 -1 )
		show_result += "]\n\n  ";
}
show_result += "]]]\n\n";

show_result= show_result + "BatchNormalization" + " : [[";
for (k = 0; k <  4 ; k++) {
	show_result += "[";
	for (x = 0; x < 8 ; x++) {
		show_result += "[";
		for(y = 0; y < 8 ; y++) {
			show_result = show_result + " " + to_string((int)(O2_SW[k][x][y]));
		}
		show_result += "]";
		if(x != 8 -1 )
			show_result += "\n   ";
	}
	if(k != 4 -1 )
		show_result += "]\n\n  ";
}
show_result += "]]]\n\n";

show_result= show_result + "Convolution2D" + " : [[";
for (k = 0; k <  8 ; k++) {
	show_result += "[";
	for (x = 0; x < 8 ; x++) {
		show_result += "[";
		for(y = 0; y < 8 ; y++) {
			show_result = show_result + " " + to_string((int)(O3_SW[k][x][y]));
		}
		show_result += "]";
		if(x != 8 -1 )
			show_result += "\n   ";
	}
	if(k != 8 -1 )
		show_result += "]\n\n  ";
}
show_result += "]]]\n\n";

show_result= show_result + "MaxPooling2D" + " : [[";
for (k = 0; k <  8 ; k++) {
	show_result += "[";
	for (x = 0; x < 7 ; x++) {
		show_result += "[";
		for(y = 0; y < 7 ; y++) {
			show_result = show_result + " " + to_string((int)(O4_SW[k][x][y]));
		}
		show_result += "]";
		if(x != 7 -1 )
			show_result += "\n   ";
	}
	if(k != 8 -1 )
		show_result += "]\n\n  ";
}
show_result += "]]]\n\n";

show_result= show_result + "Activations.Relu" + " : [[";
for (k = 0; k <  8 ; k++) {
	show_result += "[";
	for (x = 0; x < 7 ; x++) {
		show_result += "[";
		for(y = 0; y < 7 ; y++) {
			show_result = show_result + " " + to_string((int)(O5_SW[k][x][y]));
		}
		show_result += "]";
		if(x != 7 -1 )
			show_result += "\n   ";
	}
	if(k != 8 -1 )
		show_result += "]\n\n  ";
}
show_result += "]]]\n\n";

show_result= show_result + "AveragePooling2D" + " : [[";
for (k = 0; k <  8 ; k++) {
	show_result += "[";
	for (x = 0; x < 1 ; x++) {
		show_result += "[";
		for(y = 0; y < 1 ; y++) {
			show_result = show_result + " " + to_string((int)(O6_SW[k][x][y]));
		}
		show_result += "]";
		if(x != 1 -1 )
			show_result += "\n   ";
	}
	if(k != 8 -1 )
		show_result += "]\n\n  ";
}
show_result += "]]]\n\n";


  
  fprintf(o_stream,"%s",show_result.c_str());
  
  fclose(w_stream);
  fclose(b_stream);
  fclose(i_stream);
  fclose(o_stream);
  return 0;
}

