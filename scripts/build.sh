BUILD_DIR=cmake-build

rm -r ${BUILD_DIR}
cmake -H. -B${BUILD_DIR}
cd ${BUILD_DIR} && make -j12
cd bin && ./tests
cd ../..