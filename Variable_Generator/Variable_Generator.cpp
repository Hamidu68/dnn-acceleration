#include<fstream>
#include<string>
#include<stdlib.h>
#include<stdio.h>
#include<time.h>
#include<iostream>
using namespace std;

// argv[1] = Test.csv (layer file) 
// argv[2] = Init_weight.bin (Output-weight file)
// argv[3] = Init_Bias.bin (Output-weight file)
// argv[4] = Init_Input.bin (Output-weight file)
// argv[5] = 20 (Variable Random Range)
int main(int argc, char *argv[])
{
	srand(time(NULL));

	//////////////////////////////////////
	//////////Define Variable/////////////
	//////////////////////////////////////

	size_t pos;
	string line,batch, temp;
	string layer_type, batch_input_shape, batch_output_shape, filter, kernel_size;
	int i_w, i_h, i_c;  // Input_width, Input_height, Input_channel
	int o_c;            // output_channel
	int f_w, f_h;       // filter_width, filter_height
	int input_size, weight_size, bias_size;  //Size of varibles that have to be initialized

	//open csv (layer info)
	ifstream csv_file;
	csv_file.open(argv[1]);
	int random_range = atoi(argv[5]);
	int layer_count = 0;
	////////////////////////////////////////////////////////////////////////////////////////
	/////////Read the layer file line by line and Generate the necessary variables./////////
	////////////////////////////////////////////////////////////////////////////////////////

	//Open files
	FILE *input_f = fopen(argv[4], "wb");
	FILE *weight_f = fopen(argv[2], "wb");
	FILE *bias_f = fopen(argv[3], "wb");
	
	getline(csv_file, line); //first line (Ignore label)
	while (getline(csv_file, line)){ //other lines
		printf("Generate layer variables : %d\n", ++layer_count);
		size_t pos = 0;
		pos = line.find(',');
		line = line.substr(pos + 1); //Delete name
		pos = line.find(',');
		layer_type = line.substr(0, pos); //Set layer_type
		line = line.substr(pos + 3); // Delete layer_type
		
		//Make variable depending on layer type
		if (layer_type == "InputLayer")
		{
			//Read line
			pos = line.find(')');
			batch_input_shape = line.substr(0, pos); //Set batch_Input_shape 
			
			//batch_Input_shape  ("batch, width, height, channel")

			//Divide Input value
			temp = batch_input_shape;
			pos = temp.find(",");
			batch = temp.substr(1, pos); //Set batch
			temp = temp.substr(pos + 1);
			pos = temp.find(","); 
			i_w = stoi(temp.substr(0, pos)); //Set Input_width
			temp = temp.substr(pos + 1);
			pos = temp.find(",");
			i_h = stoi(temp.substr(0, pos)); //Set Input_height
			i_c = stoi(temp.substr(pos + 1)); //Set Input_channel

			//variable initialize = Input
			input_size = i_c * i_w * i_h; // Input size
			//save
			int trash;
			for (int i = 0; i < input_size; i++) {
				trash = (rand() % random_range) + 1;
				fwrite(&trash, sizeof(int), 1, input_f);
			}
		}
		else if (layer_type == "Conv2D")
		{
            //Read line
			pos = line.find(')');
			batch_input_shape = line.substr(0, pos); //Set batch_Input_shape
			line = line.substr(pos + 5);
			pos = line.find(')');
			batch_output_shape = line.substr(0, pos); //Set batch_Output_shape
			line = line.substr(pos + 3);
			pos = line.find(",");
			filter = line.substr(0, pos); //Set filter
			line = line.substr(pos + 3);
			pos = line.find(')');
			kernel_size = line.substr(0, pos); //Set kernel_size
			
			//batch_Input_shape  ("batch, width, height, channel")

			//Divide Input value
			temp = batch_input_shape;
			pos = temp.find(",");
			batch = temp.substr(1, pos); //Set batch
			temp = temp.substr(pos + 1);
			pos = temp.find(",");
			i_w = stoi(temp.substr(0, pos)); //Set Input_width
			temp = temp.substr(pos + 1);
			pos = temp.find(",");
			i_h = stoi(temp.substr(0, pos)); //Set Input_height
			i_c = stoi(temp.substr(pos + 1)); //Set Input_channel

			//filters
			o_c = stoi(filter);  //Set Output_channel

			//kernel size
			temp = kernel_size;
			pos = temp.find(",");
			f_w = stoi(temp.substr(0, pos)); //Set Filter_width
			f_h = stoi(temp.substr(pos + 1)); //Set Filter_height

			//variable initialize : Weight
			//save
			weight_size = i_c * o_c * f_w * f_h;
			int trash;
			for (int i = 0; i < weight_size; i++) {
				trash = (rand() % random_range) + 1;
				fwrite(&trash, sizeof(int), 1, weight_f);
			}
			
			//variable initialize : Bias
			//save
			bias_size = o_c;
			
			for (int i = 0; i < bias_size; i++) {
				trash = (rand() % random_range) + 1;
				fwrite(&trash, sizeof(int), 1, bias_f);
			}
			
		}

		else if (layer_type == "Dense")
		{
                        //Read line
			pos = line.find(')');
			batch_input_shape = line.substr(0, pos);
			line = line.substr(pos + 5);
			pos = line.find(')');
			batch_output_shape = line.substr(0, pos);

			//input
			temp = batch_input_shape;
			pos = temp.find(",");
			batch = temp.substr(0, pos); // Set Batch
			i_c = stoi(temp.substr(pos + 1)); // Set input_channel

			//output
			temp = batch_output_shape;
			pos = temp.find(",");
			batch = temp.substr(0, pos); // Set Batch
			o_c = stoi(temp.substr(pos + 1)); // Set Output_channel

			//variable initialize : Weight
			weight_size = i_c * o_c;
			int trash;
			for (int i = 0; i < weight_size; i++) {
				trash = (rand() % random_range) + 1;
				fwrite(&trash, sizeof(int), 1, weight_f);
			}
			
			//variable initialize : Bias
			bias_size = o_c;
			
			for (int i = 0; i < bias_size; i++) {
				trash = (rand() % random_range) + 1;
				fwrite(&trash, sizeof(int), 1, bias_f);
			}
			
		}
		else // Nothing
		{
		}
	}

	//Close
	fclose(input_f);
	fclose(weight_f);
	fclose(bias_f);

	return 0;
}
