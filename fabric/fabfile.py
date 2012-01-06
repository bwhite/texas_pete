from fabric.api import run, sudo
from fabric.context_managers import cd, settings
import time
import os


def make_drive_root(drive_md5, start_time, name):
    return '/texaspete/data/%s/dvtp/%s/run-%s/' % (drive_md5, name, start_time)


def make_output(path='/mnt/out'):
    print('Making [%s] world writable' % path)
    sudo('mkdir %s' % path)
    sudo('chmod 777 %s' % path)


def install_data(root='/mnt/out/'):
    with settings(warn_only=True):
        make_output()
    work_dir = '%s/data-%f' % (root, time.time())
    run('mkdir -p %s/classifier_data' % work_dir)
    with cd(work_dir):
        run('wget http://picarus-data.s3.amazonaws.com/classifier_data.tar')
        run('tar -xf classifier_data.tar')
        with settings(warn_only=True):
            run('hadoop fs -mkdir .')
        run('hadoop fs -put classifier_data .')


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


def run_all():
    with settings(warn_only=True):
        sudo('hadoop fs -mkdir /texaspete', user='hdfs')
        sudo('hadoop fs -chmod 777 /texaspete', user='hdfs')
    work_dir = 'tp-%f' % time.time()
    run('mkdir %s' % work_dir)
    with cd(work_dir):
        run('git clone https://github.com/bwhite/texas_pete')
        with cd('texas_pete'):
            run('python tp_workflow.py --video_path classifier_data/video_record_youtube_action_dataset/ --graphic_path classifier_data/unlabeled_record_flickr drivehashhere --training_data classifier_data')


def use_report(report_start_time, drive_hash):
    with settings(warn_only=True):
        sudo('hadoop fs -mkdir /texaspete', user='hdfs')
        sudo('hadoop fs -mkdir /texaspete/data', user='hdfs')
        sudo('hadoop fs -mkdir /texaspete/data/%s' % drive_hash, user='hdfs')
        sudo('hadoop fs -mkdir /texaspete/data/%s/reports' % drive_hash, user='hdfs')
        sudo('hadoop fs -mkdir /texaspete/data/%s/reports/data/' % drive_hash, user='hdfs')
        sudo('hadoop fs -chmod 777 /texaspete/data/%s/reports/data/' % drive_hash, user='hdfs')
    root = make_drive_root(drive_hash, report_start_time, 'report') + '/report/report/'
    sudo('hadoop fs -cp %s/sample_report.js /texaspete/data/%s/reports/data/' % (root, drive_hash), user='hdfs')
    sudo('hadoop fs -cp %s/t /texaspete/data/%s/reports/data/t/' % (root, drive_hash), user='hdfs')
