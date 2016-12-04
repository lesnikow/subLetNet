* What I did was basically running the shell script on the tensorflow site (that you shared me). 
  This script gives some std output as follows. 

    W tensorflow/core/framework/op_def_util.cc:332] Op BatchNormWithGlobalNormalization is deprecated. It will cease to work in GraphDef version 9. Use tf.nn.batch_normalization().
	I tensorflow/examples/label_image/main.cc:206] indoor (1): 0.973283
	I tensorflow/examples/label_image/main.cc:206] outdoor (0): 0.0267166
	../share/subLetNet/par11_000/quart1/1000434.jpg

  I'm no bash expert and I don't have better idea to pick those information out. 
  Thus I restored the std output of the script in a file. Then I wrote a python code to parse it and gather those key information. *
 	

1. run -> bazel build tensorflow/examples/label_image:label_image (run it in tensorflow dir)

2. go to inference.sh
	a) modify FILES as the path to the image directory

	b) modify graph dir and label file dir (where you put output_graph.pb and output_labels.txt)

3. run ./inference.sh &> output.txt (store std output to file)

4. go to parse.py and change the path to the output file generated in 3)

5. run parse.py 
	a) you will find filtered images in 'inference_result' folder