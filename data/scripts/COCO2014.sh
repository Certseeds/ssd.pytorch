#!/bin/bash
set -e
start=$(date +%s)

function create_format() {
  mkdir -p ./data
  cd ./data/
  mkdir -p ./coco
  cd ./coco
  mkdir -p ./images
  mkdir -p ./annotations
  cd ./../..
}
origin="$(pwd)"
data_path="${HOME}"
if [[ -z "$1" ]]; then
  temp=$(pwd)
else
  if [[ ! -d "$1" ]]; then
    echo $1 " is not a valid directory"
    exit 0
  else
    data_path="$1"
  fi
fi
cd "${data_path}"
create_format
cd "${origin}"

cd "${data_path}/data/coco"
# Download the image data.
cd ./images

echo "Downloading MSCOCO train images ..."
# curl -LOC http://images.cocodataset.org/zips/train2014.zip
echo "Downloading MSCOCO val images ..."
# curl -LO http://images.cocodataset.org/zips/val2014.zip

cd ./../

# Download the annotation data.
cd ./annotations
echo "Downloading MSCOCO train/val annotations ..."
#curl -LO http://images.cocodataset.org/annotations/annotations_trainval2014.zip
echo "Finished downloading. Now extracting ..."

# Unzip data
echo "Extracting train images ..."
unzip ../images/train2014.zip -d ../images
echo "Extracting val images ..."
unzip ../images/val2014.zip -d ../images
echo "Extracting annotations ..."
unzip ./annotations_trainval2014.zip

echo "Removing zip files ..."
# rm ../images/train2014.zip
# rm ../images/val2014.zip
# rm ./annotations_trainval2014.zip

echo "Creating trainval35k dataset..."

# Download annotations json
echo "Downloading trainval35k annotations from S3"
# curl -LO https://s3.amazonaws.com/amdegroot-datasets/instances_trainval35k.json.zip

# combine train and val
echo "Combining train and val images"
mkdir ../images/trainval35k
cd ../images/train2014
find -maxdepth 1 -name '*.jpg' -exec cp -t ../trainval35k {} + # dir too large for cp
cd ../val2014
find -maxdepth 1 -name '*.jpg' -exec cp -t ../trainval35k {} +

end=$(date +%s)
runtime=$((end - start))

echo "Completed in " $runtime " seconds"
