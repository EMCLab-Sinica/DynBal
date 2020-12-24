set -e
set -x

pushd ARM-CMSIS
./create_symlinks.sh "$PWD/../ARM-CMSIS_5/CMSIS"
popd

cat >> /etc/pacman.conf <<EOF
[archlinuxcn]
Server = https://repo.archlinuxcn.org/\$arch
SigLevel = Never
EOF

pacman -Syu --noconfirm
pacman -S --noconfirm --needed base-devel cmake python-numpy python-onnx python-tensorflow wget
pacman -U --noconfirm https://build.archlinuxcn.org/~yan12125/python-torchaudio-git-r628.9c484027-1-x86_64.pkg.tar.zst

# preparation
cmake_args=""
MY_DEBUG="1"
run_args=""

if [[ $USE_ARM_CMSIS = 1 ]]; then
    cmake_args="$cmake_args -D USE_ARM_CMSIS=ON"
fi

if [[ $DEBUG_BUILD = 1 ]]; then
    MY_DEBUG="2"
    run_args="$run_args 1"
fi
cmake_args="$cmake_args -D MY_DEBUG=$MY_DEBUG"

if [[ $CONFIG = *mnist* ]]; then
    ./data/download-mnist.sh
fi
if [[ $CONFIG = *cifar10* ]]; then
    ./data/download-cifar10.sh
fi
if [[ $CONFIG = *kws* ]]; then
    git submodule init
    git submodule update data/ML-KWS-for-MCU
fi

python transform.py $CONFIG
cmake -B build $cmake_args
make -C build
./build/intermittent-cnn $run_args
