BUILD_DIR=cmake-build

rm -r ${BUILD_DIR}
cmake -H. -B${BUILD_DIR}
cd ${BUILD_DIR} && make -j12
if [ -f ${BUILD_DIR}/bin/tests ]; then
  cd bin && ./tests && cd ..
fi
cd ..

cp -r ${BUILD_DIR}/bin/pytensor* python/lib/
