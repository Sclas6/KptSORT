python3.11 -m venv ../.cv2
source ../.cv2/bin/activate

python -c "from sysconfig import get_paths as gp; print(gp()['include'])"
python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"

deactivate