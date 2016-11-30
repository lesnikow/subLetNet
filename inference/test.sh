FILES=../share/subLetNet/par11_000/quart1

for f in $FILES/*.jpg
do
output=`bazel-bin/tensorflow/examples/label_image/label_image \
    --graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt \
    --output_layer=final_result \
    --image=$f` && echo $f
done