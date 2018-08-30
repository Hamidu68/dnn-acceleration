#include <stdio.h>
#include<iostream>
#include <stdlib.h>
#include<string>
#include<string.h>
#include<math.h>
using namespace std;

typedef float DATA_T;

void SW_block1_conv1(DATA_T I[3][224][224], DATA_T O[64][224][224], DATA_T B[64], DATA_T W[64][3][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(224 - 1) - 224 + 3)/2; 
	for (m = 0; m<64; m++) {
		for (x = 0; x<224; x++) {
			for (y = 0; y<224; y++) {
				ofm = B[m];
				for (k = 0; k<3; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 224 + p && y + j < 224 + p && x + i -p >= 0 && y + j -p >= 0) { 
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

void SW_block1_conv2(DATA_T I[64][224][224], DATA_T O[64][224][224], DATA_T B[64], DATA_T W[64][64][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(224 - 1) - 224 + 3)/2; 
	for (m = 0; m<64; m++) {
		for (x = 0; x<224; x++) {
			for (y = 0; y<224; y++) {
				ofm = B[m];
				for (k = 0; k<64; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 224 + p && y + j < 224 + p && x + i -p >= 0 && y + j -p >= 0) { 
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

void SW_block1_pool(DATA_T I[64][224][224], DATA_T O[64][112][112])
{ 
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<64; m++) {
		for (x = 0; x<112; x++) {
			for (y = 0; y<112; y++) {
				max = I[m][x*2][y*2];
				for (i = 0; i<2; i++) {
					for (j = 0; j<2; j++) {
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
void SW_block2_conv1(DATA_T I[64][112][112], DATA_T O[128][112][112], DATA_T B[128], DATA_T W[128][64][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(112 - 1) - 112 + 3)/2; 
	for (m = 0; m<128; m++) {
		for (x = 0; x<112; x++) {
			for (y = 0; y<112; y++) {
				ofm = B[m];
				for (k = 0; k<64; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 112 + p && y + j < 112 + p && x + i -p >= 0 && y + j -p >= 0) { 
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

void SW_block2_conv2(DATA_T I[128][112][112], DATA_T O[128][112][112], DATA_T B[128], DATA_T W[128][128][3][3]) {
	int m, x, y, i, j, k;
	DATA_T ifm, ofm;
    int p = (1 *(112 - 1) - 112 + 3)/2; 
	for (m = 0; m<128; m++) {
		for (x = 0; x<112; x++) {
			for (y = 0; y<112; y++) {
				ofm = B[m];
				for (k = 0; k<128; k++) {
					for (i = 0; i<3; i++) {
						for (j = 0; j<3; j++) {
							if (x + i < 112 + p && y + j < 112 + p && x + i -p >= 0 && y + j -p >= 0) { 
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

void SW_block2_pool(DATA_T I[128][112][112], DATA_T O[128][56][56])
{ 
	int m, x, y, i, j;
	DATA_T max;
	for (m = 0; m<128; m++) {
		for (x = 0; x<56; x++) {
			for (y = 0; y<56; y++) {
				max = I[m][x*2][y*2];
				for (i = 0; i<2; i++) {
					for (j = 0; j<2; j++) {
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


//argv[1] = init_weight.txt , argv[2] = init_bias.txt , argv[3] = init_input.txt
int main(int argc, char *argv[]){
 
  DATA_T temp;
  int m, x, y, i, j, k, l;
  char * w_line, * b_line, *i_line;
  size_t b_len, w_len, i_len;
  char *i_v, * w_v, *b_v;
  static DATA_T I[3][224][224];
	static DATA_T W1[64][3][3][3];
	static DATA_T B1[64];
	static DATA_T W2[64][64][3][3];
	static DATA_T B2[64];
	static DATA_T W4[128][64][3][3];
	static DATA_T B4[128];
	static DATA_T W5[128][128][3][3];
	static DATA_T B5[128];
	
 
  static DATA_T O0_SW[3][224][224];
	static DATA_T O1_SW[64][224][224];
	static DATA_T O2_SW[64][224][224];
	static DATA_T O3_SW[64][112][112];
	static DATA_T O4_SW[128][112][112];
	static DATA_T O5_SW[128][112][112];
	static DATA_T O6_SW[128][56][56];
	 

  FILE *w_stream = fopen(argv[1], "r");
  if (w_stream == NULL) printf("weight file was not opened");
  FILE *b_stream = fopen(argv[2], "r");
  if (b_stream == NULL) printf("bias file was not opened");
  FILE *i_stream = fopen(argv[3], "r");
  if (i_stream == NULL) printf("i_stream file was not opened");
  FILE *o_stream = fopen("Output/C_output.txt", "w");
  if (o_stream == NULL) printf("Output file was not opened");
 
  printf("[C_verifier.cpp]Start Initialzation\n\n");
  i_line = NULL;
i_len = 0;
getline(&i_line, &i_len, i_stream);
for (k = 0; k <  3 ; k++) {
	for (x = 0; x < 224 ; x++) {
		for(y = 0; y < 224 ; y++) {
			i_v = strtok_r(i_line, " ", &i_line);
                        I[k][x][y] = atof(i_v);
			O0_SW[k][x][y] = atof(i_v);
		}
	}
}

	w_line = b_line = NULL;
w_len = b_len = 0;
getline(&w_line, &w_len, w_stream);
getline(&b_line, &b_len, b_stream);
for (m = 0; m <  64 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 3 ; i++) {
			for (j = 0; j < 3 ; j++) {
                                w_v = strtok_r(w_line," ",&w_line);
				W1[m][k][i][j] = atof(w_v);       
			}
		}
	}
	b_v = strtok_r(b_line," ",&b_line);
        B1[m] = atof(b_v);            
}

	w_line = b_line = NULL;
w_len = b_len = 0;
getline(&w_line, &w_len, w_stream);
getline(&b_line, &b_len, b_stream);
for (m = 0; m <  64 ; m++) {
	for (k = 0; k < 64 ; k++) {
		for (i = 0; i < 3 ; i++) {
			for (j = 0; j < 3 ; j++) {
                                w_v = strtok_r(w_line," ",&w_line);
				W2[m][k][i][j] = atof(w_v);       
			}
		}
	}
	b_v = strtok_r(b_line," ",&b_line);
        B2[m] = atof(b_v);            
}

	w_line = b_line = NULL;
w_len = b_len = 0;
getline(&w_line, &w_len, w_stream);
getline(&b_line, &b_len, b_stream);
for (m = 0; m <  128 ; m++) {
	for (k = 0; k < 64 ; k++) {
		for (i = 0; i < 3 ; i++) {
			for (j = 0; j < 3 ; j++) {
                                w_v = strtok_r(w_line," ",&w_line);
				W4[m][k][i][j] = atof(w_v);       
			}
		}
	}
	b_v = strtok_r(b_line," ",&b_line);
        B4[m] = atof(b_v);            
}

	w_line = b_line = NULL;
w_len = b_len = 0;
getline(&w_line, &w_len, w_stream);
getline(&b_line, &b_len, b_stream);
for (m = 0; m <  128 ; m++) {
	for (k = 0; k < 128 ; k++) {
		for (i = 0; i < 3 ; i++) {
			for (j = 0; j < 3 ; j++) {
                                w_v = strtok_r(w_line," ",&w_line);
				W5[m][k][i][j] = atof(w_v);       
			}
		}
	}
	b_v = strtok_r(b_line," ",&b_line);
        B5[m] = atof(b_v);            
}

	
  printf("[C_verifier.cpp]Finish Initialization\n\n");
  printf("[C_verifier.cpp]InputLayer\n\n");
	printf("[C_verifier.cpp]Calcluate Conv2D1\n\n");
	SW_block1_conv1(O0_SW,O1_SW,B1,W1);
	printf("[C_verifier.cpp]Calcluate Conv2D2\n\n");
	SW_block1_conv2(O1_SW,O2_SW,B2,W2);
	printf("[C_verifier.cpp]Calcluate MaxPooling2D3\n\n");
	SW_block1_pool(O2_SW,O3_SW);
	printf("[C_verifier.cpp]Calcluate Conv2D4\n\n");
	SW_block2_conv1(O3_SW,O4_SW,B4,W4);
	printf("[C_verifier.cpp]Calcluate Conv2D5\n\n");
	SW_block2_conv2(O4_SW,O5_SW,B5,W5);
	printf("[C_verifier.cpp]Calcluate MaxPooling2D6\n\n");
	SW_block2_pool(O5_SW,O6_SW);
	  string show_result;
  printf("[C_verifier.cpp]Print Result\n");
  show_result= show_result + "InputLayer" + " : [[";
for (k = 0; k <  3 ; k++) {
	show_result += "[";
	for (x = 0; x < 224 ; x++) {
		show_result += "[";
		for(y = 0; y < 224 ; y++) {
			show_result = show_result +" "+to_string((int)(O0_SW[k][x][y]));
		}
		show_result += "]";
		if(x != 224 -1 )
			show_result += "\n   ";
	}
	if(k != 3 -1 )
		show_result += "]\n\n  ";
}
show_result += "]]]\n\n";

show_result= show_result + "Convolution2D" + " : [[";
for (k = 0; k <  64 ; k++) {
	show_result += "[";
	for (x = 0; x < 224 ; x++) {
		show_result += "[";
		for(y = 0; y < 224 ; y++) {
			show_result = show_result +" "+to_string((int)(O1_SW[k][x][y]));
		}
		show_result += "]";
		if(x != 224 -1 )
			show_result += "\n   ";
	}
	if(k != 64 -1 )
		show_result += "]\n\n  ";
}
show_result += "]]]\n\n";

show_result= show_result + "Convolution2D" + " : [[";
for (k = 0; k <  64 ; k++) {
	show_result += "[";
	for (x = 0; x < 224 ; x++) {
		show_result += "[";
		for(y = 0; y < 224 ; y++) {
			show_result = show_result +" "+to_string((int)(O2_SW[k][x][y]));
		}
		show_result += "]";
		if(x != 224 -1 )
			show_result += "\n   ";
	}
	if(k != 64 -1 )
		show_result += "]\n\n  ";
}
show_result += "]]]\n\n";

show_result= show_result + "MaxPooling2D" + " : [[";
for (k = 0; k <  64 ; k++) {
	show_result += "[";
	for (x = 0; x < 112 ; x++) {
		show_result += "[";
		for(y = 0; y < 112 ; y++) {
			show_result = show_result +" "+to_string((int)(O3_SW[k][x][y]));
		}
		show_result += "]";
		if(x != 112 -1 )
			show_result += "\n   ";
	}
	if(k != 64 -1 )
		show_result += "]\n\n  ";
}
show_result += "]]]\n\n";

show_result= show_result + "Convolution2D" + " : [[";
for (k = 0; k <  128 ; k++) {
	show_result += "[";
	for (x = 0; x < 112 ; x++) {
		show_result += "[";
		for(y = 0; y < 112 ; y++) {
			show_result = show_result +" "+to_string((int)(O4_SW[k][x][y]));
		}
		show_result += "]";
		if(x != 112 -1 )
			show_result += "\n   ";
	}
	if(k != 128 -1 )
		show_result += "]\n\n  ";
}
show_result += "]]]\n\n";

show_result= show_result + "Convolution2D" + " : [[";
for (k = 0; k <  128 ; k++) {
	show_result += "[";
	for (x = 0; x < 112 ; x++) {
		show_result += "[";
		for(y = 0; y < 112 ; y++) {
			show_result = show_result +" "+to_string((int)(O5_SW[k][x][y]));
		}
		show_result += "]";
		if(x != 112 -1 )
			show_result += "\n   ";
	}
	if(k != 128 -1 )
		show_result += "]\n\n  ";
}
show_result += "]]]\n\n";

show_result= show_result + "MaxPooling2D" + " : [[";
for (k = 0; k <  128 ; k++) {
	show_result += "[";
	for (x = 0; x < 56 ; x++) {
		show_result += "[";
		for(y = 0; y < 56 ; y++) {
			show_result = show_result +" "+to_string((int)(O6_SW[k][x][y]));
		}
		show_result += "]";
		if(x != 56 -1 )
			show_result += "\n   ";
	}
	if(k != 128 -1 )
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

