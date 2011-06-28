from fabric.api import run, sudo
from fabric.context_managers import cd
import time
import os


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


def install_git(repo):
    run('git clone %s' % repo)
    with cd(os.path.basename(repo)):
        run('python setup.py build')
        sudo('python setup.py install')


def install_dickarus():  # Andrew's fucntion
    run('uname -s')
    work_dir = 'picarus-%f' % time.time()
    run('mkdir %s' % work_dir)
    with cd(work_dir):

        # Apt Get
        sudo('apt-get install -y ffmpeg libavcodec-dev libswscale-dev libavformat-dev fftw3-dev python-dev build-essential python-imaging python-numpy python-scipy git-core python-setuptools')

        # Setuptools
        sudo('easy_install cython numpy PIL scons')

        # Pyffmpeg, keyframe, vidfeat
        install_git('https://github.com/amiller/pyffmpeg')
        install_git('https://github.com/bwhite/vidfeat')
        install_git('https://github.com/bwhite/imfeat')
        install_git('https://github.com/bwhite/keyframe')
        install_git('https://github.com/bwhite/classipy')
        install_git('https://github.com/bwhite/hadoopy')
        install_git('https://github.com/bwhite/impoint')
        install_git('https://github.com/bwhite/hadoopy_flow')
        install_git('https://github.com/bwhite/picarus')
