find . -iname "*.jpg" -type f | xargs -I{} identify -format '%w %h %i \n' {} | awk '$1<300 || $2<300'

