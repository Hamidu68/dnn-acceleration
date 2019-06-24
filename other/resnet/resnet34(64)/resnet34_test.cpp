#include <iostream>
#include <ap_int.h>
#include "hls_stream.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
using namespace std;


typedef int DATA_IN;
typedef int DATA_T;

void resnet34_top(DATA_T I[3][64][64],DATA_T W1[64][3][3][3],DATA_T B1[64], DATA_T W3[64][64][3][3],DATA_T B3[64], DATA_T W4[64][64][3][3],DATA_T B4[64], DATA_T W5[64][64][1][1],DATA_T B5[64], DATA_T W8[64][64][3][3],DATA_T B8[64], DATA_T W10[64][64][3][3],DATA_T B10[64], DATA_T W13[64][64][3][3],DATA_T B13[64], DATA_T W15[64][64][3][3],DATA_T B15[64], DATA_T W18[128][64][3][3],DATA_T B18[128], DATA_T W20[128][128][3][3],DATA_T B20[128], DATA_T W21[128][64][1][1],DATA_T B21[128], DATA_T W24[128][128][3][3],DATA_T B24[128], DATA_T W26[128][128][3][3],DATA_T B26[128], DATA_T W29[128][128][3][3],DATA_T B29[128], DATA_T W31[128][128][3][3],DATA_T B31[128], DATA_T W34[128][128][3][3],DATA_T B34[128], DATA_T W36[128][128][3][3],DATA_T B36[128], DATA_T W39[256][128][3][3],DATA_T B39[256], DATA_T W41[256][256][3][3],DATA_T B41[256], DATA_T W42[256][128][1][1],DATA_T B42[256], DATA_T W45[256][256][3][3],DATA_T B45[256], DATA_T W47[256][256][3][3],DATA_T B47[256], DATA_T W50[256][256][3][3],DATA_T B50[256], DATA_T W52[256][256][3][3],DATA_T B52[256], DATA_T W55[256][256][3][3],DATA_T B55[256], DATA_T W57[256][256][3][3],DATA_T B57[256], DATA_T W60[256][256][3][3],DATA_T B60[256], DATA_T W62[256][256][3][3],DATA_T B62[256], DATA_T W65[256][256][3][3],DATA_T B65[256], DATA_T W67[256][256][3][3],DATA_T B67[256], DATA_T W70[512][256][3][3],DATA_T B70[512], DATA_T W72[512][512][3][3],DATA_T B72[512], DATA_T W73[512][256][1][1],DATA_T B73[512], DATA_T W76[512][512][3][3],DATA_T B76[512], DATA_T W78[512][512][3][3],DATA_T B78[512], DATA_T W81[512][512][3][3],DATA_T B81[512], DATA_T W83[512][512][3][3],DATA_T B83[512], DATA_T W88[200][512],DATA_T B88[200], DATA_T O[200]);
void resnet34_sw(DATA_T I[3][64][64],DATA_T W1[64][3][3][3],DATA_T B1[64], DATA_T W3[64][64][3][3],DATA_T B3[64], DATA_T W4[64][64][3][3],DATA_T B4[64], DATA_T W5[64][64][1][1],DATA_T B5[64], DATA_T W8[64][64][3][3],DATA_T B8[64], DATA_T W10[64][64][3][3],DATA_T B10[64], DATA_T W13[64][64][3][3],DATA_T B13[64], DATA_T W15[64][64][3][3],DATA_T B15[64], DATA_T W18[128][64][3][3],DATA_T B18[128], DATA_T W20[128][128][3][3],DATA_T B20[128], DATA_T W21[128][64][1][1],DATA_T B21[128], DATA_T W24[128][128][3][3],DATA_T B24[128], DATA_T W26[128][128][3][3],DATA_T B26[128], DATA_T W29[128][128][3][3],DATA_T B29[128], DATA_T W31[128][128][3][3],DATA_T B31[128], DATA_T W34[128][128][3][3],DATA_T B34[128], DATA_T W36[128][128][3][3],DATA_T B36[128], DATA_T W39[256][128][3][3],DATA_T B39[256], DATA_T W41[256][256][3][3],DATA_T B41[256], DATA_T W42[256][128][1][1],DATA_T B42[256], DATA_T W45[256][256][3][3],DATA_T B45[256], DATA_T W47[256][256][3][3],DATA_T B47[256], DATA_T W50[256][256][3][3],DATA_T B50[256], DATA_T W52[256][256][3][3],DATA_T B52[256], DATA_T W55[256][256][3][3],DATA_T B55[256], DATA_T W57[256][256][3][3],DATA_T B57[256], DATA_T W60[256][256][3][3],DATA_T B60[256], DATA_T W62[256][256][3][3],DATA_T B62[256], DATA_T W65[256][256][3][3],DATA_T B65[256], DATA_T W67[256][256][3][3],DATA_T B67[256], DATA_T W70[512][256][3][3],DATA_T B70[512], DATA_T W72[512][512][3][3],DATA_T B72[512], DATA_T W73[512][256][1][1],DATA_T B73[512], DATA_T W76[512][512][3][3],DATA_T B76[512], DATA_T W78[512][512][3][3],DATA_T B78[512], DATA_T W81[512][512][3][3],DATA_T B81[512], DATA_T W83[512][512][3][3],DATA_T B83[512], DATA_T W88[200][512],DATA_T B88[200], DATA_T O[200]);

int main() {

	int m, x, y, i, j, k;
	DATA_T temp;
	int trash;

	static DATA_T I[3][64][64];
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
	static DATA_T W13[64][64][3][3];
	static DATA_T B13[64];
	static DATA_T W15[64][64][3][3];
	static DATA_T B15[64];
	static DATA_T W18[128][64][3][3];
	static DATA_T B18[128];
	static DATA_T W20[128][128][3][3];
	static DATA_T B20[128];
	static DATA_T W21[128][64][1][1];
	static DATA_T B21[128];
	static DATA_T W24[128][128][3][3];
	static DATA_T B24[128];
	static DATA_T W26[128][128][3][3];
	static DATA_T B26[128];
	static DATA_T W29[128][128][3][3];
	static DATA_T B29[128];
	static DATA_T W31[128][128][3][3];
	static DATA_T B31[128];
	static DATA_T W34[128][128][3][3];
	static DATA_T B34[128];
	static DATA_T W36[128][128][3][3];
	static DATA_T B36[128];
	static DATA_T W39[256][128][3][3];
	static DATA_T B39[256];
	static DATA_T W41[256][256][3][3];
	static DATA_T B41[256];
	static DATA_T W42[256][128][1][1];
	static DATA_T B42[256];
	static DATA_T W45[256][256][3][3];
	static DATA_T B45[256];
	static DATA_T W47[256][256][3][3];
	static DATA_T B47[256];
	static DATA_T W50[256][256][3][3];
	static DATA_T B50[256];
	static DATA_T W52[256][256][3][3];
	static DATA_T B52[256];
	static DATA_T W55[256][256][3][3];
	static DATA_T B55[256];
	static DATA_T W57[256][256][3][3];
	static DATA_T B57[256];
	static DATA_T W60[256][256][3][3];
	static DATA_T B60[256];
	static DATA_T W62[256][256][3][3];
	static DATA_T B62[256];
	static DATA_T W65[256][256][3][3];
	static DATA_T B65[256];
	static DATA_T W67[256][256][3][3];
	static DATA_T B67[256];
	static DATA_T W70[512][256][3][3];
	static DATA_T B70[512];
	static DATA_T W72[512][512][3][3];
	static DATA_T B72[512];
	static DATA_T W73[512][256][1][1];
	static DATA_T B73[512];
	static DATA_T W76[512][512][3][3];
	static DATA_T B76[512];
	static DATA_T W78[512][512][3][3];
	static DATA_T B78[512];
	static DATA_T W81[512][512][3][3];
	static DATA_T B81[512];
	static DATA_T W83[512][512][3][3];
	static DATA_T B83[512];
	static DATA_T W88[200][512];
	static DATA_T B88[200];
	static DATA_T O0_SW[3][64][64];
	static DATA_T O_SW[200];
	static DATA_T O_HW[200];
	

	FILE *w_stream = fopen("init_weight.bin", "r");
	if (w_stream == NULL) printf("weight file was not opened");
	FILE *i_stream = fopen("init_input.bin", "r");
	if (i_stream == NULL) printf("input file was not opened");

 	for (k = 0; k <  64 ; k++) {
	for (x = 0; x < 64 ; x++) {
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
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W1[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B1[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W3[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B3[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W4[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B4[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W5[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B5[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W8[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B8[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W10[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B10[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W13[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B13[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 64 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W15[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 64 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B15[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W18[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B18[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W20[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B20[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 64 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W21[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B21[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W24[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B24[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W26[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B26[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W29[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B29[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W31[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B31[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W34[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B34[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 128 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W36[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 128 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B36[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W39[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B39[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W41[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B41[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 128 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W42[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B42[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W45[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B45[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W47[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B47[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W50[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B50[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W52[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B52[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W55[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B55[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W57[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B57[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W60[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B60[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W62[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B62[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W65[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B65[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 256 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W67[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 256 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B67[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W70[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B70[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W72[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B72[m] = (DATA_T) trash;
}

	for (m = 0; m <  1 ; m++) {
	for (k = 0; k < 1 ; k++) {
		for (i = 0; i < 256 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W73[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B73[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W76[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B76[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W78[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B78[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W81[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B81[m] = (DATA_T) trash;
}

	for (m = 0; m <  3 ; m++) {
	for (k = 0; k < 3 ; k++) {
		for (i = 0; i < 512 ; i++) {
			for (j = 0; j < 512 ; j++) {
				fread(&trash, sizeof(DATA_T), 1, w_stream);
                W83[j][i][m][k] = (DATA_T) trash;
			}
		}
	}
}

for (m = 0; m < 512 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B83[m] = (DATA_T) trash;
}

	for (m = 0; m <  512 ; m++) {
	for (k = 0; k < 200 ; k++) {
		fread(&trash, sizeof(DATA_T), 1, w_stream);
        W88[k][m] = (DATA_T) trash;
	}
}

for (m = 0; m < 200 ; m++) {
    fread(&trash, sizeof(DATA_T), 1, w_stream);
    B88[m] = (DATA_T) trash;
}

	

	resnet34_top(I, W1, B1, W3, B3, W4, B4, W5, B5, W8, B8, W10, B10, W13, B13, W15, B15, W18, B18, W20, B20, W21, B21, W24, B24, W26, B26, W29, B29, W31, B31, W34, B34, W36, B36, W39, B39, W41, B41, W42, B42, W45, B45, W47, B47, W50, B50, W52, B52, W55, B55, W57, B57, W60, B60, W62, B62, W65, B65, W67, B67, W70, B70, W72, B72, W73, B73, W76, B76, W78, B78, W81, B81, W83, B83, W88, B88,  O_HW);
	resnet34_sw(I, W1, B1, W3, B3, W4, B4, W5, B5, W8, B8, W10, B10, W13, B13, W15, B15, W18, B18, W20, B20, W21, B21, W24, B24, W26, B26, W29, B29, W31, B31, W34, B34, W36, B36, W39, B39, W41, B41, W42, B42, W45, B45, W47, B47, W50, B50, W52, B52, W55, B55, W57, B57, W60, B60, W62, B62, W65, B65, W67, B67, W70, B70, W72, B72, W73, B73, W76, B76, W78, B78, W81, B81, W83, B83, W88, B88,  O_SW);

	int err_cnt =0;
	for(m=0; m<200; m++){
	    cout<<"SW: O["<<m<<"] = "<<O_SW[m];
	    cout<<"HW: O["<<m<<"] = "<<O_HW[m];
	    if (O_HW[m] != O_SW[m]){
	        err_cnt++;
	    }
	}

	int ret_val;
	if (err_cnt == 0) {
		printf("*** TEST PASSED ***\n");
		ret_val = 0;
	} else {
		printf("!!! TEST FAILED - %d mismatches detected !!!\n\n",err_cnt);
		ret_val = -1;
	}

	fclose(w_stream);
	fclose(i_stream);
	return ret_val;
}