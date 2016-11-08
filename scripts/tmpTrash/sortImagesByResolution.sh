#!/bin/bash

NumberOfBaseDir="0"
SubDirNumber="0"
CountAll="0"
NumDir="0"

echo Creating destination.
Destination="./SortedImages"
echo mkdir -v "${Destination}" -p 
echo Created destination with status: $?

echo Your set of limitations.
SDN=50; echo Limit files in a subdir: $SDN
NBD=50; echo Limit subdirs in a file type named destination: $NBD

SourceDataFile="collect-info-about-images.txt"
echo Source file with a necessery data: $SourceDataFile

if [ 'XX' == 'XX'"$SourceDataFile" ] ; then 
 echo The '$SourceDataFile' variable is empty
 exit 1
else 
 if [ ! -f "$SourceDataFile" ]; then 
  echo The "$SourceDataFile" file doesn"'"t exist
  exit 2
 fi;
fi;

echo Populating an array from a file
ArrayFillCount=0;
while read line ; do
  tmpWb="${line/|H|*/}";
W="${tmpWb/*W|/}";
  tmpHb="${line/|Format|*/}";
H="${tmpHb/*|/}";

#if (( "$W" >= "800" )) && (( "$W" <=  "1000" )); then
#if (( "$H" >= "800" )) && (( "$H" <=  "1000" )); then
  ArrayOfFiles[$ArrayFillCount]="$line";
  ArrayFillCount=$((ArrayFillCount+1))
DupLimitKeeper[$W,$H]="0";
#fi;fi;

done < $SourceDataFile;
echo Done with extracting of necessary data about resolutions.

echo Starting loop of restoration
XX=${#ArrayOfFiles[@]}
while [  "${XX}" != "${CountAl}l" ] ; do
  preType=${ArrayOfFiles[$CountAll]/*"|Format|"/};
ImageType=${preType/|*/}
  preW=${ArrayOfFiles[$CountAll]/*"|W|"/};Width=${preW/|*/};preH=${ArrayOfFiles[$CountAll]/*"|H|"/} 
Height=${preH/|*/};
PathToFile=${ArrayOfFiles[$CountAll]/"|"*/}

DupLimitKeeper[Width,Height]=$((DupLimitKeeper[Width,Height]+1));

IfExist="${Destination}/${ImageType}${NumberOfBaseDir}/Resolution_${Width}x${Height}_DirN${SubDirNumber}"

if [ ! -d "$IfExist"  ];then 
  echo mkdir -vp "$IfExist"
NumDir=$((NumDir+1));
fi

## Creating a new numbered file type folders
if [ "${DupLimitKeeper[Width,Height]}" -gt $SDN ]; then 
  SubDirNumber=$((SubDirNumber+1));
  DupLimitKeeper[$Width,$Height]="0";
fi

## Adding a file number 
FileNameOnly="${PathToFile##*/}"
NewFileName="N${CountAll}C${FileNameOnly}"
#NewFileName="${FileNameOnly}"

## Creating a new sub-dir when limit of files in a sub-folder is reached
if [ $NumDir -gt $NBD ];then 
  NumberOfBaseDir=$((NumberOfBaseDir+1));
  NumDir="0";
fi
##
if [ -f "${PathToFile}" ];then
  echo mv -v "${PathToFile}" "$IfExist/$NewFileName";
# echo cp -v "${PathToFile}" "$IfExist/$NewFileName";
fi

CountAll=$((CountAll+1)) 
done
echo Total processed files: $CountAll
