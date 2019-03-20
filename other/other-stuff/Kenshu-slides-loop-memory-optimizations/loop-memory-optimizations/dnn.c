#include <stdio.h>

#define M 64
#define F 64 
#define E 64 
#define R 3
#define S 3
#define C 64

// need padding
int O1[M][F+2*(R/2)][E+2*(S/2)];
int O[M][F][E];
int I[C][F+2*(R/2)][E+2*(S/2)];


int main() {

  int B[M];
  int W[M][C][R][S];

  int m, x, y, i, j, k;


  for (m=0; m<M; m++) {
    B[m] = ((m+3)>>1) % 255;
  }

  for (m=0; m<M; m++) {
    for (k=0; k<C; k++) {
      for (i=0; i<R; i++) {
        for (j=0; j<S; j++) {
          W[m][k][i][j] = ((i+1)^2+j*7+k^2+(m+5)) % 255;
        }
      }
    }
  }

  for (k=0; k<C; k++) {
    for (i=0; i<R; i++) {
      for (j=0; j<S; j++) {
        I[k][i][j] = (i*3+j^+k*7) % 255;
      }
    }
  }


#pragma scop

  for (m=0; m<M; m++) {
    for (x=0; x<F; x++) {
      for (y=0; y<E; y++) {
        for (k=0; k<C; k++) { 
          if (k==0) {
            O1[m][x][y] = B[m];
          }
          O1[m][x][y] = O1[m][x][y] + I[k][x+0][y+0] * W[m][k][0][0];
          O1[m][x][y] = O1[m][x][y] + I[k][x+0][y+1] * W[m][k][0][1];
          O1[m][x][y] = O1[m][x][y] + I[k][x+0][y+2] * W[m][k][0][2];
          O1[m][x][y] = O1[m][x][y] + I[k][x+1][y+0] * W[m][k][1][0];
          O1[m][x][y] = O1[m][x][y] + I[k][x+1][y+1] * W[m][k][1][1];
          O1[m][x][y] = O1[m][x][y] + I[k][x+1][y+2] * W[m][k][1][2];
          O1[m][x][y] = O1[m][x][y] + I[k][x+2][y+0] * W[m][k][2][0];
          O1[m][x][y] = O1[m][x][y] + I[k][x+2][y+1] * W[m][k][2][1];
          O1[m][x][y] = O1[m][x][y] + I[k][x+2][y+2] * W[m][k][2][2];
        }
      }
    }
  }


  for (m=0; m<M; m++) {
    for (x=0; x<F; x++) {
      for (y=0; y<E; y++) {
        for (k=0; k<C; k++) { 
          if (k==0) {
            O[m][x][y] = B[m];
          }
          O[m][x][y] = O[m][x][y] + O1[k][x+0][y+0] * W[m][k][0][0];
          O[m][x][y] = O[m][x][y] + O1[k][x+0][y+1] * W[m][k][0][1];
          O[m][x][y] = O[m][x][y] + O1[k][x+0][y+2] * W[m][k][0][2];
          O[m][x][y] = O[m][x][y] + O1[k][x+1][y+0] * W[m][k][1][0];
          O[m][x][y] = O[m][x][y] + O1[k][x+1][y+1] * W[m][k][1][1];
          O[m][x][y] = O[m][x][y] + O1[k][x+1][y+2] * W[m][k][1][2];
          O[m][x][y] = O[m][x][y] + O1[k][x+2][y+0] * W[m][k][2][0];
          O[m][x][y] = O[m][x][y] + O1[k][x+2][y+1] * W[m][k][2][1];
          O[m][x][y] = O[m][x][y] + O1[k][x+2][y+2] * W[m][k][2][2];
        }
      }
    }
  }





#pragma endscop

  for (m=0; m<M; m++) {
    for (x=1; x<F; x++) {
      for (y=1; y<E; y++) {
        printf("O[%d][%d][%d] = %d\n", m, x, y, O[m][x][y]);
      }
    }
  }

  return 0;
}

//O[n][m][x][y] = O[n][m][x][y] + I[n][k][x+i][y+j] + W[m][k][i][j];
