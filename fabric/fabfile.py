from fabric.api import run, sudo
from fabric.context_managers import cd, settings
import time
import os


def install_picarus():
    run('uname -s')
    work_dir = 'picarus-%f' % time.time()
    run('mkdir %s' % work_dir)
    sudo('apt-get -y install libavcodec-dev libswscale-dev libavformat-dev gfortran ffmpeg fftw3-dev python-dev build-essential git-core python-setuptools cmake libjpeg62-dev libpng12-dev libblas-dev liblapack-dev libevent-dev python-scipy python-numpy')
    with cd(work_dir):
        # Apt Get
        sudo('easy_install scons cython gevent bottle pil')
        install_git('https://github.com/amiller/pyffmpeg')
        install_git('https://github.com/bwhite/vidfeat')
        install_git('https://github.com/bwhite/imfeat')
        install_git('https://github.com/bwhite/keyframe')
        install_git('https://github.com/bwhite/classipy')
        install_git('https://github.com/bwhite/impoint')
        install_git('https://github.com/bwhite/pyram')
        install_git('https://github.com/bwhite/hadoopy')
        install_git('https://github.com/bwhite/hadoopy_flow')
        install_git('https://github.com/bwhite/picarus')
        run('git clone https://github.com/bwhite/texas_pete')
    #install_opencv()


def install_opencv():
    run('wget http://downloads.sourceforge.net/project/opencvlibrary/opencv-unix/2.2/OpenCV-2.2.0.tar.bz2')
    # OpenCV
    run('tar -xjf OpenCV-2.2.0.tar.bz2')
    run('mkdir OpenCV-2.2.0/build')
    with cd('OpenCV-2.2.0/build'):
        run('cmake ..')
        run('make -j8')
        sudo('make install')
        sudo('cp  lib/cv.so `python -c "from distutils.sysconfig import get_python_lib; print get_python_lib()"`')


def install_data(root=''):
    work_dir = '%s/data-%f' % (root, time.time())
    run('mkdir -p %s/classifier_data' % work_dir)
    with cd(work_dir):
        run('wget http://picarus-data.s3.amazonaws.com/classifier_data.tar')
        run('tar -xf classifier_data.tar')
        with settings(warn_only=True):
            run('hadoop fs -mkdir .')
        run('hadoop fs -put classifier_data .')


def install_git(repo):
    run('git clone %s' % repo)
    with cd(os.path.basename(repo)):
        sudo('python setup.py install')


def run_tp():
    work_dir = 'tp-%f' % time.time()
    run('mkdir %s' % work_dir)
    with cd(work_dir):
        run('git clone https://github.com/bwhite/texas_pete')
        with cd('texas_pete'):
            run('python tp_workflow.py')
