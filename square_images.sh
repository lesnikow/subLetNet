#!/bin/bash
d=64
for f in *.jpg
do 
	echo $f
	convert $f -resize $((d))x$((d))! $((d))_$f
done
