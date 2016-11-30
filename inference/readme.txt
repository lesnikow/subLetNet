1. run -> bazel build tensorflow/examples/label_image:label_image (run it in tensorflow dir)

2. go to inference.sh
	b) modify FILES as the path to the image directory

	a) modify graph dir and label file dir (where you put output_graph.pb and output_labels.txt)

4. run ./inference.sh &> output.txt (store std output to file)