rm -r /etc/apt/sources.list.d/ubuntu.sources
cat <<EOF > /etc/apt/sources.list.d/ubuntu.sources
Types: deb
URIs: https://ftp.udx.icscoe.jp/Linux/ubuntu/
Suites: noble noble-updates noble-backports
Components: main restricted universe multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg

Types: deb
URIs: http://security.ubuntu.com/ubuntu/
Suites: noble-security
Components: main restricted universe multiverse
Signed-By: /usr/share/keyrings/ubuntu-archive-keyring.gpg
EOF

apt update

apt install software-properties-common
add-apt-repository ppa:apt-fast/stable
add-apt-repository ppa:deadsnakes/ppa
sudo apt update
apt -y install apt-fast
apt-fast update

apt-fast upgrade -y
apt-fast update && apt-fast install -y gcc g++ make cmake git nasm python3.11-venv python3.11-dev pkg-config curl

python3.11 -m venv ../.cv2
source ../.cv2/bin/activate

python -c "from sysconfig import get_paths as gp; print(gp()['include'])"
python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())"
python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"

deactivate