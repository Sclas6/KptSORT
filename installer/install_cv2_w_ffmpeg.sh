python3.11 -m venv ../.cv2
source ../.cv2/bin/activate
echo ${VIRTUAL_ENV}

pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
pip install -r requirements.txt
pip uninstall opencv-python

apt install -y gcc g++ make cmake git nasm python3.11-venv python3.11-dev pkg-config

export OPENH_264_VER="2.5.1"
curl -LO https://github.com/cisco/openh264/archive/refs/tags/${OPENH_264_VER}.tar.gz \
  && tar -xvzf ${OPENH_264_VER}.tar.gz \
  && cd openh264-${OPENH_264_VER} \
  && make -j4 \
  && make install PREFIX=$VIRTUAL_ENV

cd ../

export FFMPEG_VER="n4.3.9"
curl -LO https://github.com/FFmpeg/FFmpeg/archive/refs/tags/${FFMPEG_VER}.tar.gz \
  && tar -xvzf ${FFMPEG_VER}.tar.gz \
  && cd FFmpeg-${FFMPEG_VER} \
  && ./configure \
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
  --disable-htmlpages \
  --disable-manpages \
  --disable-podpages \
  --disable-txtpages \
  --disable-avdevice \
  --disable-postproc \
  --disable-bzlib \
  --disable-iconv \
  --disable-cuda \
  --disable-cuvid \
  --disable-debug \
  && make -j4 \
  && make install PREFIX=$VIRTUAL_ENV \
  && ldconfig 

cd ../

export OPENCV_CONTRIB_VER="4.11.0"
curl -LO https://github.com/opencv/opencv_contrib/archive/refs/tags/${OPENCV_CONTRIB_VER}.tar.gz \
  && tar -xvzf ${OPENCV_CONTRIB_VER}.tar.gz

export OPENCV_CONTRIB_MODULE=/kpsort/installer/opencv_contrib-${OPENCV_CONTRIB_VER}/modules

export PYTHON_EXEC="${VIRTUAL_ENV}/bin/python3"
# python3 -c "from sysconfig import get_paths as gp; print(gp()['include'])"
export PYTHON_INCLUDE="/usr/include/python3.11"
# python3 -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
export PYTHON_SITE="${VIRTUAL_ENV}/lib/python3.11/site-packages"
# python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"
export PYTHON_LIB="/usr/lib/x86_64-linux-gnu"

export OPENCV_VER="4.11.0"
export INSTALL_DIR=${VIRTUAL_ENV}
curl -LO https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VER}.tar.gz \
  && tar -xvzf ${OPENCV_VER}.tar.gz \
  && cd opencv-${OPENCV_VER} \
  && mkdir build \
  && cd build \
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
    -DFFMPEG_INCLUDE_DIR=/usr/local/include/ \
    -DFFMPEG_LIB_DIR=/usr/local/lib/ \
    -DOPENCV_FFMPEG_USE_FIND_PACKAGE=OFF \
    -DHAVE_opencv_python2=OFF \
    -DHAVE_opencv_python3=ON \
    -DBUILD_opencv_python2=OFF \
    -DBUILD_opencv_python3=ON \
    -DWITH_OPENH264=ON \
    -DBUILD_DOCS=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_JAVA=OFF \
    -DWITH_1394=OFF \
    -DWITH_CUDA=OFF \
    -DWITH_CUFFT=OFF \
    -DWITH_FFMPEG=ON \
    -DWITH_GSTREAMERE=OFF \
    -DWITH_GTK=OFF \
    -DWITH_IPP=OFF \
    -DWITH_JASPERE=OFF \
    -DWITH_JPEG=ON \
    -DWITH_OPENEXR=OFF \
    -DWITH_PNG=ON \
    -DWITH_TIFF=ON \
    -DWITH_V4L=OFF \
    -DWITH_GPHOTO2=OFF \
    -DWITH_CUBLAS=OFF \
    -DWITH_VTK=OFF \
    -DWITH_NVCUVID=OFF \
    .. \
  && make -j4 \
  && make install PREFIX=$VIRTUAL_ENV \
  && ldconfig

cd ../../
rm *.gz
rm -r FFmpeg-n4.3.9
rm -r opencv_contrib-4.11.0
rm -r opencv-4.11.0
rm -r openh264-2.5.1
