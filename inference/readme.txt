1. run -> bazel build tensorflow/examples/label_image:label_image (run it in tensorflow dir)

2. go to inference.sh
	a) modify FILES as the path to the image directory

	b) modify graph dir and label file dir (where you put output_graph.pb and output_labels.txt)

3. run ./inference.sh &> output.txt (store std output to file)

4. go to parse.py and change the path to the output file generated in 3)

5. run parse.py (not completed yet)