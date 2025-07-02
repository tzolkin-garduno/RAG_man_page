#!/bin/bash

DESTDIR=/home/gwolf/man-mirror
MIRROR=https://deb.debian.org/debian/
PKG_AMD64=dists/unstable/main/binary-amd64/Packages.xz
PKG_ALL=dists/unstable/main/binary-all/Packages.xz
PKGLIST=$DESTDIR/packages
DONE_PATH=${DESTDIR}/done_pkgs

function report() {
    now=$(date +%H:%M)
    echo $1
    echo "[$now] $1" >> ${DESTDIR}/man-mirror.log
}

report "🔰 Starting process"
[ -d $DESTDIR ] || mkdir $DESTDIR

report "🔽 Starting download: Package lists"
wget $MIRROR$PKG_AMD64 -q -O - | xzcat | grep ^Filename: | cut -f 2 -d : > $PKGLIST
wget $MIRROR$PKG_ALL -q -O - | xzcat | grep ^Filename: | cut -f 2 -d : >> $PKGLIST

# Iterate over all downloaded packages
for PKG in $(cat $PKGLIST)
do
    report "⚞ Downloading package: $PKG"

    # Set up the environment
    TMPDIR=$(mktemp -d)
    PKGFILE=$(basename $PKG)
    PKGNAME=$(echo $PKGFILE|sed s/_.*//)

    # Have we already processed this package? If so, jump to the next one
    if egrep -q "^${PKGNAME}$" $DONE_PATH
    then
	report "✅ ${PKGNAME}: Already processed. Skipping."
	continue
    fi

    pushd $TMPDIR > /dev/null
    wget -q $MIRROR$PKG

    # Uncompress the package and extract only the manpges; decompress and
    # convert from groff to plain text
    report "📦 Unpackaging $PKG and moving manfiles to destination"
    ar x $PKGFILE
    tar xf data.tar.* ./usr/share/man 2>/dev/null
    if [ ! -d ./usr/share/man ]
    then
	report "👀 $PKGNAME contains no manual pages."
    fi
    for MANFILE in $(find usr/share/man -type f)
    do
	UNGZ_MANFILE=${MANFILE/.gz/}
	TXTFILE=${UNGZ_MANFILE}.txt
	report "  $MANFILE ⇒ $TXTFILE"
	gunzip $MANFILE
	man $UNGZ_MANFILE > $TXTFILE 2>/dev/null
	rm $UNGZ_MANFILE

	mkdir -p $DESTDIR/$(dirname $TXTFILE)
	mv $TXTFILE $DESTDIR/$TXTFILE
    done

    # ...Next one... :-Þ
    echo $PKGNAME >> $DONE_PATH
    popd > /dev/null
    rm -rf $TMPDIR
done
