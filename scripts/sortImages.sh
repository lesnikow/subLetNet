#!/bin/bash

for image in *.jpg;
    do res=$(identify -format %wx%h\\n $image);
    mkdir -p $res;
    cp $image $res;
done
