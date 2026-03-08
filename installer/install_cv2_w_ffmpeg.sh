#!/bin/bash
set -e

python3.11 -m venv ../.cv2
source ../.cv2/bin/activate
echo "Virtual Env: ${VIRTUAL_ENV}"

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
pip uninstall -y opencv-python

export OPENH_264_VER="2.5.1"
curl -LO https://github.com/cisco/openh264/archive/refs/tags/${OPENH_264_VER}.tar.gz \
  && tar -xvzf ${OPENH_264_VER}.tar.gz \
  && cd openh264-${OPENH_264_VER} \
  && make -j4 \
  && make install PREFIX=$VIRTUAL_ENV
cd ../

export FFMPEG_VER="n4.3.9"
export PKG_CONFIG_PATH="${VIRTUAL_ENV}/lib/pkgconfig:${PKG_CONFIG_PATH}"

curl -LO https://github.com/FFmpeg/FFmpeg/archive/refs/tags/${FFMPEG_VER}.tar.gz \
  && tar -xvzf ${FFMPEG_VER}.tar.gz \
  && cd FFmpeg-${FFMPEG_VER} \
  && ./configure \
  --prefix=$VIRTUAL_ENV \
  --enable-shared \
  --enable-gpl \
  --disable-libx264 \
  --disable-libaom \
  --enable-nonfree \
  --enable-libopenh264 \
  --enable-optimizations \
  --enable-static \
  --enable-version3 \
  --disable-logging \
  --disable-doc \
  --disable-avdevice \
  --disable-postproc \
  --disable-bzlib \
  --disable-iconv \
  --disable-cuda \
  --disable-cuvid \
  --disable-debug \
  && make -j4 \
  && make install \
  && ldconfig 
cd ../

export OPENCV_CONTRIB_VER="4.11.0"
curl -LO https://github.com/opencv/opencv_contrib/archive/refs/tags/${OPENCV_CONTRIB_VER}.tar.gz \
  && tar -xvzf ${OPENCV_CONTRIB_VER}.tar.gz

export OPENCV_CONTRIB_MODULE=$(pwd)/opencv_contrib-${OPENCV_CONTRIB_VER}/modules

export PYTHON_EXEC="${VIRTUAL_ENV}/bin/python3"
export PYTHON_INCLUDE="/usr/include/python3.11"
export PYTHON_SITE="${VIRTUAL_ENV}/lib/python3.11/site-packages"
export PYTHON_LIB="/usr/lib/x86_64-linux-gnu"

export OPENCV_VER="4.11.0"
export INSTALL_DIR=${VIRTUAL_ENV}
export LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib:${LD_LIBRARY_PATH}"

curl -LO https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VER}.tar.gz \
  && tar -xvzf ${OPENCV_VER}.tar.gz \
  && cd opencv-${OPENCV_VER} \
  && mkdir -p build && cd build \
  && cmake \
    -DCMAKE_BUILD_TYPE=RELEASE \
    -DOPENCV_EXTRA_MODULES_PATH=${OPENCV_CONTRIB_MODULE} \
    -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
    -DOPENCV_PYTHON_INSTALL_PATH=${PYTHON_SITE} \
    -DPYTHON3_EXECUTABLE=${PYTHON_EXEC} \
    -DPYTHON_DEFAULT_EXECUTABLE=${PYTHON_EXEC} \
    -DPYTHON3_INCLUDE_DIR=${PYTHON_INCLUDE} \
    -DPYTHON3_LIBRARY=${PYTHON_LIB}/libpython3.11.so \
    -DPYTHON3_PACKAGES_PATH=${PYTHON_SITE} \
    -DOPENCV_FFMPEG_USE_FIND_PACKAGE=ON \
    -DFFMPEG_INCLUDE_DIR=${VIRTUAL_ENV}/include \
    -DFFMPEG_LIB_DIR=${VIRTUAL_ENV}/lib \
    -DHAVE_opencv_python3=ON \
    -DBUILD_opencv_python3=ON \
    -DWITH_FFMPEG=ON \
    -DWITH_OPENH264=ON \
    -DBUILD_DOCS=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_JAVA=OFF \
    -DWITH_CUDA=OFF \
    -DWITH_JPEG=ON \
    -DWITH_PNG=ON \
    -DWITH_TIFF=ON \
    .. \
  && make -j4 \
  && make install \
  && ldconfig

cd ../../
rm *.gz
rm -rf FFmpeg-n4.3.9 opencv_contrib-4.11.0 opencv-4.11.0 openh264-2.5.1