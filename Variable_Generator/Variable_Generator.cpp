#include<fstream>
#include<string>
#include<stdlib.h>
#include<stdio.h>
#include<time.h>
#include<iostream>
using namespace std;

// argv[1] = Test.csv (layer file) 
// argv[2] = Init_weight.txt (Output-weight file)
// argv[3] = Init_Bias.txt (Output-weight file)
// argv[4] = Init_Input.txt (Output-weight file)
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
	string bias_t, weight_t, input_t;        //Gather initialzied values to string
	bias_t = "";
	weight_t = "";
	input_t = "";

	//open csv (layer info)
	ifstream csv_file;
	csv_file.open(argv[1]);
	int random_range = atoi(argv[5]);
	int layer_count = 0;
	////////////////////////////////////////////////////////////////////////////////////////
	/////////Read the layer file line by line and Generate the necessary variables./////////
	////////////////////////////////////////////////////////////////////////////////////////

	getline(csv_file, line); //first line (Ignore label)
	while (getline(csv_file, line)){ //other lines
		printf("layer : %d\n", ++layer_count);
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
			input_t += to_string(rand() % random_range  + 1);
			for (int i = 1; i < input_size; i++) {
				input_t += " " +to_string(rand() % random_range + 1);
			}
			input_t += "\n";
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
			weight_t += to_string((rand() % random_range) + 1);
			for (int i = 1; i < weight_size; i++) {
				weight_t += " " + to_string((rand() % random_range) + 1);
			}
			weight_t += "\n";

			//variable initialize : Bias
			//save
			bias_size = o_c;
			bias_t += to_string((rand() % random_range) + 1);
			for (int i = 1; i < bias_size; i++) {
				bias_t += " " + to_string((rand() % random_range) + 1);
			}
			bias_t += "\n";

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
			weight_t += to_string((rand() % random_range) + 1);
			for (int i = 1; i < weight_size; i++) {
				weight_t += " " + to_string(rand() %random_range + 1);
			}
			weight_t += "\n";

			//variable initialize : Bias
			bias_size = o_c;
			bias_t += to_string((rand() % random_range) + 1);
			for (int i = 1; i < bias_size; i++) {
				bias_t += " " + to_string((rand() % random_range) + 1);
			}
			bias_t += "\n";
		}
		else // Nothing
		{
		}
	}

	///////////////////////////////////////////////////
	//////////////// Write file ///////////////////////
	///////////////////////////////////////////////////

	ofstream Weight, Bias, Input;

	//Open output(weight,bias,input) txt file
	Weight.open(argv[2]);
	Bias.open(argv[3]);
	Input.open(argv[4]);

	//Write
	Input << input_t;
	Bias << bias_t;
	Weight << weight_t;

	//Close
	Weight.close();
	Bias.close();
	Input.close();

	return 0;
}
