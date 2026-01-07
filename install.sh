path=$(dirname "$(readlink -f "$0")")

mkdir -p $path/opt
git clone https://github.com/krasow/upmem-vector.git $path/opt/vectordpu_src
git clone https://github.com/krasow/simple-pim-clone.git $path/opt/SimplePIM

DESTDIR=$path/opt/vectordpu
cd $path/opt/vectordpu_src
DESTDIR=$DESTDIR make install