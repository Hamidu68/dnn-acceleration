#include <iostream>
#include <hls_stream.h>

using namespace std;

typedef int DATA_T;

void func1(DATA_T I[16][16], DATA_T W1[16][16], /*DATA_T*/ hls::stream<DATA_T> &T1) {
#pragma HLS INLINE
  int i, j;

  for (int T=0; T<16; T+=4) {
    func1_i_loop:for (i=0; i<16; i++) {
      func1_j_loop: for (j=0; j<4; j++) {
#pragma HLS PIPELINE  //what if we move it up (before the loop)?
	    //T1[i][T+j] = I[i][T+j] + 1 + W1[i][T+j];
  	    T1.write(I[i][T+j] + 1 + W1[i][T+j]);
      }
    }
  }
}

//T2 is not streamed -> we use buffer for that?
void func2(/*DATA_T*/ hls::stream<DATA_T> &T1, DATA_T W2[16][16], DATA_T T2[16][16]) {
//void func2(/*DATA_T*/ hls::stream<DATA_T> &T1[16][16], DATA_T W2[16][16], DATA_T T2[16][16]) {
#pragma HLS INLINE
  int i, j;
  for (int T=0; T<16; T+=4) {
    func2_i_loop:for (i=0; i<16; i++) {
      func2_j_loop: for (j=0; j<4; j++) {
#pragma HLS PIPELINE
    	T2[i][T+j] = T1.read() + 2 + W2[i][T+j];
    	//T2[i][T+j] = T1[i][T+j] + 2 + W2[i][T+j];
      }
    }
  }
}

void func3(DATA_T T2[16][16], DATA_T T3[16][16]) {
#pragma HLS INLINE
  int i, j;

  func3_i_loop:for (i=0; i<16; i++) {
    func3_j_loop: for (j=0; j<16; j++) {
#pragma HLS PIPELINE
    	T3[i][j] = T2[i][j] + 3;
    }
  }
}

void func4(DATA_T T3[16][16], DATA_T W3[16][16], DATA_T O[16][16]) {
#pragma HLS INLINE
  int i, j;

  func4_i_loop:for (i=0; i<16; i++) {
    func4_j_loop: for (j=0; j<16; j++) {
#pragma HLS PIPELINE
    	O[i][j] = T3[i][j] + 3 + W3[i][j];
    }
  }
}

//--

static DATA_T I_i[16][16];
static DATA_T W1_i[16][16];
static DATA_T W2_i[16][16];
static DATA_T W3_i[16][16];


void sample(DATA_T I[16][16], DATA_T W1_i[16][16], DATA_T W2_i[16][16], DATA_T W3_i[16][16], DATA_T O[16][16]) {
//void sample(DATA_T I[16][16], /* DATA_T W1[16][16],*/  DATA_T O[16][16]) {
#pragma HLS DATAFLOW

  //DATA_T T1[16][16];
  hls::stream<DATA_T> T1;
  DATA_T T2[16][16];
  DATA_T T3[16][16];
//#pragma HLS stream variable=T1 depth=1
//#pragma HLS stream variable=T2 depth=1
//#pragma HLS stream variable=T3 depth=1
  func1(I, W1_i, T1);
  func2(T1, W2_i, T2);
  func3(T2, T3);
  func4(T3, W3_i, O); //why don't we use streaming for the output?

}



void sample_top(DATA_T I[16][16], DATA_T W1[16][16], DATA_T W2[16][16], DATA_T W3[16][16], DATA_T O[16][16]) {

  // internal memories in accelerator
  static DATA_T O_i[16][16];

  int i, j;

  I_i_i_loop: for (i=0; i<16; i++) {
    I_i_j_loop: for (j=0; j<16; j++) {
      I_i[i][j] = I[i][j];
    }
  }
  W1_i_i_loop: for (i=0; i<16; i++) {
    W1_i_j_loop: for (j=0; j<16; j++) {
      W1_i[i][j] = W1[i][j];
    }
  }
  W2_i_i_loop: for (i=0; i<16; i++) {
    W2_i_j_loop: for (j=0; j<16; j++) {
      W2_i[i][j] = W2[i][j];
    }
  }
  W3_i_i_loop: for (i=0; i<16; i++) {
    W3_i_j_loop: for (j=0; j<16; j++) {
      W3_i[i][j] = W3[i][j];
    }
  }
  sample(I_i, W1_i, W2_i, W3_i, O_i);
  //sample(I_i, /*W1_i,*/ O_i);
  //sample(I, /*W1_i,*/ O_i);

  O_i_i_loop: for (i=0; i<16; i++) {
    O_i_j_loop: for (j=0; j<16; j++) {
      O[i][j] = O_i[i][j];
    }
  }

}


void sample_sw(DATA_T I[16][16], DATA_T W1[16][16], DATA_T W2[16][16],  DATA_T W3[16][16], DATA_T O[16][16]) {

  hls::stream<DATA_T> T1;
  //static DATA_T T1[16][16];
  static DATA_T T2[16][16];
  static DATA_T T3[16][16];


  func1(I, W1, T1);
  func2(T1, W2, T2);
  func3(T2, T3);
  func4(T3, W3, O);


}

