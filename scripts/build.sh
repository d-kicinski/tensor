BUILD_DIR=cmake-build

rm -r ${BUILD_DIR}
cmake -H. -B${BUILD_DIR} -GNinja
cd ${BUILD_DIR} && ninja -j12 && cd ..
if [ -f ${BUILD_DIR}/bin/tests ]; then
  cd ${BUILD_DIR}/bin && ./tests && cd ../..
fi

cp -r ${BUILD_DIR}/bin/libtensor.cpython*.so python/src/tensor
