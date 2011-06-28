from fabric.api import run, sudo
from fabric.context_managers import cd
import time


def install_picarus():
    run('uname -s')
    work_dir = 'picarus-%f' % time.time()
    run('mkdir %s' % work_dir)
    with cd(work_dir):
        # Apt Get
        sudo('apt-get -y install gfortran ffmpeg fftw3-dev python-dev build-essential git-core python-setuptools cmake libjpeg62-dev libpng12-dev')
        run('wget http://downloads.sourceforge.net/project/opencvlibrary/opencv-unix/2.2/OpenCV-2.2.0.tar.bz2')
        # OpenCV
        sudo('easy_install numpy scipy pil')
        run('tar -xjf OpenCV-2.2.0.tar.bz2')
        run('mkdir OpenCV-2.2.0/build')
        with cd('OpenCV-2.2.0/build'):
            run('cmake ..')
            run('make -j8')
            sudo('make install')


def install_dickarus():  # Andrew's fucntion
    pass
