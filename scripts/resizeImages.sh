# Resizes images in pwd by 25%
for f in *.jpg; do convert $f -resize 25% output/$f; done
