from fabric.api import run, sudo
from fabric.context_managers import cd
import time


def install_picarus():
    run('uname -s')
    work_dir = 'picarus-%f' % time.time()
    run('mkdir %s' % work_dir)
    with cd(work_dir):
        # Apt Get
        sudo('apt-get install ffmpeg fftw3-dev python-dev build-essential python-imaging python-numpy python-scipy git-core python-setuptools')
        run('wget http://downloads.sourceforge.net/project/opencvlibrary/opencv-unix/2.2/OpenCV-2.2.0.tar.bz2')
        # OpenCV
        run('tar -xjf OpenCV-2.2.0.tar.bz2')


def install_dickarus():  # Andrew's fucntion
    pass
