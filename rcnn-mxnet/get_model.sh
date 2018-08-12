#! /bin/bash
#SYMBOL_FILE=http://data.dmlc.ml/models/imagenet/inception-bn/Inception-BN-symbol.json
#PARAM_FILE=http://data.dmlc.ml/models/imagenet/inception-bn/Inception-BN-0126.params

#MEAN_FILE=https://github.com/dmlc/web-data/raw/master/mxnet/example/feature_extract/mean_224.nd
#SYNSET_FILE=http://data.dmlc.ml/models/imagenet/synset.txt

SYMBOL_FILE=http://data.dmlc.ml/models/imagenet/resnet/50-layers/resnet-50-symbol.json
PARAM_FILE=http://data.dmlc.ml/models/imagenet/resnet/50-layers/resnet-50-0000.params

MEAN_FILE=https://github.com/dmlc/web-data/raw/master/mxnet/example/feature_extract/mean_224.nd
SYNSET_FILE=http://data.dmlc.ml/models/imagenet/resnet/synset.txt


mkdir -p model
wget -nc -O model/model_symbol.json $SYMBOL_FILE
wget -nc -O model/model_param.params $PARAM_FILE
wget -nc -O model/synset.txt $SYNSET_FILE
wget -nc -O model/mean_224.nd $MEAN_FILE

