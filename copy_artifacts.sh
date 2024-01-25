#!/bin/bash
mkdir $BUILD_DIR/model_pipeline/ && cp $BUILD_DIR/pretrained_network.py $BUILD_DIR/preprocessing.py $BUILD_DIR/model_pipeline/

zip -r $BUILD_DIR/model_pipeline.zip $BUILD_DIR/model_pipeline
pushd $BUILD_DIR; zip -r model_pipeline.zip model_pipeline; popd

echo "copied artifacts to build dir"