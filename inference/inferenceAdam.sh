FILES=../par11_000/quart1

for f in $FILES/*.jpg
do
output=`~/tfSource/bazel-bin/tensorflow/examples/label_image/label_image \
    --graph=output_graph.pb --labels=output_labels.txt \
    --output_layer=final_result \
    --image=$f` && echo $f
done
