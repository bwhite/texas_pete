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
        sudo('easy_install scons cython gevent bottle pil argparse')
        install_git('https://github.com/amiller/pyffmpeg')
        install_git('https://github.com/amiller/static_server')
        install_git('https://github.com/bwhite/image_server')
        install_git('https://github.com/bwhite/vidfeat')
        install_git('https://github.com/bwhite/imfeat')
        install_git('https://github.com/bwhite/keyframe')
        install_git('https://github.com/bwhite/classipy')
        install_git('https://github.com/bwhite/impoint')
        install_git('https://github.com/bwhite/pyram')
        install_git('https://github.com/bwhite/distpy')
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


def make_output(path='/mnt/out'):
    print('Making [%s] world writable' % path)
    sudo('mkdir %s' % path)
    sudo('chmod 777 %s' % path)


def install_data(root='.'):
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
    with settings(warn_only=True):
        sudo('hadoop fs -mkdir /texaspete', user='hdfs')
        sudo('hadoop fs -chmod 777 /texaspete', user='hdfs')
    work_dir = 'tp-%f' % time.time()
    run('mkdir %s' % work_dir)
    with cd(work_dir):
        run('git clone https://github.com/bwhite/texas_pete')
        with cd('texas_pete'):
            run('python tp_workflow.py --video_path classifier_data/video_record_youtube_action_dataset/ --graphic_path classifier_data/unlabeled_record_flickr drivehashhere --training_data classifier_data')


def install_pretrained():
    run('wget http://picarus-data.s3.amazonaws.com/run-1309370997.467325-texaspete-graphics_config-train.tar')
    run('tar -xf run-1309370997.467325-texaspete-graphics_config-train.tar')

    with settings(warn_only=True):
        sudo('hadoop fs -mkdir /texaspete', user='hdfs')
        sudo('hadoop fs -mkdir /texaspete/graphics_config', user='hdfs')
        sudo('hadoop fs -mkdir /texaspete/graphics_config/train', user='hdfs')
        sudo('hadoop fs -put run-1309370997.467325 /texaspete/graphics_config/train/', user='hdfs')


def run_pretrained():
    with settings(warn_only=True):
        sudo('hadoop fs -mkdir /texaspete', user='hdfs')
        sudo('hadoop fs -chmod 777 /texaspete', user='hdfs')
    work_dir = 'tp-%f' % time.time()
    run('mkdir %s' % work_dir)
    with cd(work_dir):
        run('git clone https://github.com/bwhite/texas_pete')
        with cd('texas_pete'):
            run('python tp_workflow.py --video_path classifier_data/video_record_youtube_action_dataset/ --graphic_path classifier_data/unlabeled_record_flickr drivehashhere --training_data classifier_data --train_start_time 1309370997.467325')


def use_report(tp_start_time, report_start_time, drive_hash):
    with settings(warn_only=True):
        sudo('hadoop fs -mkdir /texaspete', user='hdfs')
        sudo('hadoop fs -mkdir /texaspete/data', user='hdfs')
        sudo('hadoop fs -mkdir /texaspete/data/%s' % drive_hash, user='hdfs')
        sudo('hadoop fs -mkdir /texaspete/data/%s/reports' % drive_hash, user='hdfs')
        sudo('hadoop fs -mkdir /texaspete/data/%s/reports/data/' % drive_hash, user='hdfs')
        sudo('hadoop fs -chmod 777 /texaspete/data/%s/reports/data/' % drive_hash, user='hdfs')
    sudo('hadoop fs -put tp-%s/texas_pete/out/run-%s/sample_report.js /texaspete/data/%s/reports/data/' % (tp_start_time, report_start_time,
                                                                                                           drive_hash), user='hdfs')
    sudo('hadoop fs -put tp-%s/texas_pete/out/run-%s/t /texaspete/data/%s/reports/data/t/' % (tp_start_time, report_start_time,
                                                                                              drive_hash), user='hdfs')
