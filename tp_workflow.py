"""Command line interface to the Texas Pete graphics pipeline
"""
USE_FLOW = True  # Set this to True to use Hadoopy Flow, else False

if USE_FLOW:
    import hadoopy_flow
    #hadoopy_flow.USE_EXISTING = True  # Auto resume existing tasks
import hadoopy
import picarus
import time
import json
import shutil
import os
import argparse
import sys
picarus.GLOBAL_JOBCONFS += ['mapred.map.output.compression.codec=org.apache.hadoop.io.compress.SnappyCodec',
                            'mapred.compress.map.output=true',
                            'mapred.output.compress=true',
                            'mapred.output.compression.codec=org.apache.hadoop.io.compress.SnappyCodec']
# 'mapred.child.java.opts=-Xmx512M'

# HDFS Paths with data of the form (sha1_hash, record) (see picarus IO docs)
data_root = None
DRIVE_MD5 = None
FEATURE = 'meta_gist_spatial_hist'  # 'hist_joint'
IMAGE_LENGTH = 64  # 128
DATA = {'photos': {'pos': 'photos',
                   'neg': 'nonphotos',
                   'feature': FEATURE,
                   'image_length': IMAGE_LENGTH,
                   'classifier': 'svmlinear_autotune'},
        'indoors': {'pos': 'indoors',
                    'neg': 'outdoors',
                    'feature': 'meta_gist_spatial_hist_autocorrelogram',
                    'image_length': IMAGE_LENGTH,
                    'classifier': 'svmlinear_autotune'},
        'objects': {'pos': 'objects',
                    'neg': 'nonobjects',
                    'feature': FEATURE,
                    'image_length': IMAGE_LENGTH,
                    'classifier': 'svmlinear_autotune'},
        'detected_faces': {'pos': 'detected_faces',
                           'neg': 'detected_nonfaces',
                           'feature': FEATURE,  # 'meta_hog_gist_hist',
                           'image_length': IMAGE_LENGTH,
                           'classifier': 'svmlinear_autotune'},
        'faces': {'pos': 'faces',
                  'neg': 'nonfaces',
                  'feature': 'eigenface',
                  'image_length': 64},
        'pr0n': {'pos': 'pr0n',
                 'neg': 'nonpr0n',
                 'feature': FEATURE,
                 'image_length': IMAGE_LENGTH,
                 'classifier': 'svmlinear_autotune'}}


CLASSIFIERS = ['photos', 'indoors', 'objects', 'detected_faces', 'pr0n']   # A classifier is learned for each of those
CLUSTERS = [('photos', ['pos', 'neg']), ('indoors', ['pos', 'neg']),
            ('objects', ['pos']), ('faces', ['pos']), ('pr0n', ['pos'])]   # Clustering is performed on each of those
PHOTOS_SUBCLASSES = ['indoors', 'objects', 'pr0n']  # Each of these are derived from predicted photos

# Clustering parameters
NUM_LOCAL_SAMPLES = 100000
NUM_CLUSTERS = 20
NUM_ITERS = 0
NUM_OUTPUT_SAMPLES = 10

# Start time overrides: If they are non-empty, then use them instead of the current time.
# This is useful if you are adding features near the end of the pipeline and you want to resuse
# existing output.
SKIP_OVERRIDE = True
OVERRIDE_TRAIN_START_TIME = ''
OVERRIDE_TRAIN_PREDICT_START_TIME = ''
OVERRIDE_PREDICT_START_TIME = ''
OVERRIDE_VIDEOS_START_TIME = ''
OVERRIDE_REPORT_START_TIME = ''


def make_drive_root(start_time, name):
    return '/texaspete/data/%s/dvtp/%s/run-%s/' % (DRIVE_MD5, name, start_time)


def make_config_root(start_time, name):
    return '/texaspete/graphics_config/%s/run-%s/' % (name, start_time)


def make_local_root(start_time):
    out = 'out/run-%s/' % start_time
    try:
        os.makedirs(out)
    except OSError:
        pass
    return out


def dump_settings(**kw):
    with open('%sreport/config.js' % (make_local_root(kw['report_start_time'])), 'w') as fp:
        g = globals()
        out = dict((x, g[x]) for x in ('CLASSIFIERS', 'CLUSTERS', 'PHOTOS_SUBCLASSES', 'DATA'))
        out.update(kw)
        json.dump(out, fp, -1)


def train():
    # HDFS Paths for Output
    if SKIP_OVERRIDE and OVERRIDE_TRAIN_START_TIME:
        return OVERRIDE_TRAIN_START_TIME
    start_time = OVERRIDE_TRAIN_START_TIME if OVERRIDE_TRAIN_START_TIME else '%f' % time.time()
    root = make_config_root(start_time, 'train')

    # Compute features for classifier train
    for dk in CLASSIFIERS:
        d = DATA[dk]
        ipathp = '%strain_%s' % (data_root, d['pos'])
        ipathn = '%strain_%s' % (data_root, d['neg'])
        rpathp = lambda x: '%s%s/%s' % (root, x, d['pos'])
        rpathn = lambda x: '%s%s/%s' % (root, x, d['neg'])
        picarus.vision.run_image_feature(ipathp, rpathp('train_feat'), d['feature'], d['image_length'])
        picarus.vision.run_image_feature(ipathn, rpathn('train_feat'), d['feature'], d['image_length'])

    # Label images
    for dk in CLASSIFIERS:
        d = DATA[dk]
        rpathp = lambda x: '%s%s/%s' % (root, x, d['pos'])
        rpathn = lambda x: '%s%s/%s' % (root, x, d['neg'])
        lpathp = 'tp_%s_labels.js' % d['pos']
        # NOTE(brandyn): rpathp('labels') was set to None below to force local execution, see docstring
        picarus.classify.run_classifier_labels(rpathp('train_feat'), rpathn('train_feat'),
                                               rpathp('labels'), d['pos'], '', lpathp, d['classifier'])
    # Train classifiers
    for dk in CLASSIFIERS:
        d = DATA[dk]
        rpathp = lambda x: '%s%s/%s' % (root, x, d['pos'])
        rpathn = lambda x: '%s%s/%s' % (root, x, d['neg'])
        lpathp = 'tp_%s_labels.js' % d['pos']
        picarus.classify.run_train_classifier([rpathp('train_feat'), rpathn('train_feat')], rpathp('classifiers'), lpathp)
    return start_time


def train_predict(train_start_time):
    if SKIP_OVERRIDE and OVERRIDE_TRAIN_PREDICT_START_TIME:
        return OVERRIDE_TRAIN_PREDICT_START_TIME
    start_time = OVERRIDE_TRAIN_PREDICT_START_TIME if OVERRIDE_TRAIN_PREDICT_START_TIME else '%f' % time.time()
    train_root = make_config_root(train_start_time, 'train')
    root = make_config_root(start_time, 'train_predict')

    for dk in CLASSIFIERS:
        d = DATA[dk]
        ipathp = '%stest_%s' % (data_root, d['pos'])
        ipathn = '%stest_%s' % (data_root, d['neg'])
        rpathp = lambda x: '%s%s/%s' % (root, x, d['pos'])
        rpathn = lambda x: '%s%s/%s' % (root, x, d['neg'])
        picarus.vision.run_image_feature(ipathp, rpathp('test_feat'), d['feature'], d['image_length'])
        picarus.vision.run_image_feature(ipathn, rpathn('test_feat'), d['feature'], d['image_length'])

    for dk in CLASSIFIERS:
        d = DATA[dk]
        ipathp = '%stest_%s' % (data_root, d['pos'])
        ipathn = '%stest_%s' % (data_root, d['neg'])
        tpathp = lambda x: '%s%s/%s' % (train_root, x, d['pos'])
        rpathp = lambda x: '%s%s/%s' % (root, x, d['pos'])
        rpathn = lambda x: '%s%s/%s' % (root, x, d['neg'])
        picarus.classify.run_predict_classifier(rpathp('test_feat'), tpathp('classifiers'), rpathp('test_predict'))
        picarus.classify.run_predict_classifier(rpathn('test_feat'), tpathp('classifiers'), rpathn('test_predict'))
    return start_time


def _score_train_prediction(pos_pred_path, neg_pred_path, classifier_name):
    tp, fp, tn, fn = 0, 0, 0, 0
    # Pos
    for image_hash, preds in hadoopy.readtb(pos_pred_path):
        for cur_classifier_name, ((cur_conf, cur_pol),) in preds.items():
            if classifier_name == cur_classifier_name:
                conf = cur_conf * cur_pol
                if conf >= 0:
                    tp += 1
                else:
                    fn += 1
    # Neg
    for image_hash, preds in hadoopy.readtb(neg_pred_path):
        for cur_classifier_name, ((cur_conf, cur_pol),) in preds.items():
            if classifier_name == cur_classifier_name:
                conf = cur_conf * cur_pol
                if conf >= 0:
                    fp += 1
                else:
                    tn += 1
    print('%s: [%d, %d, %d, %d]' % (classifier_name, tp, fp, tn, fn))
    p, r = (tp / float(tp + fp), tp / float(tp + fn))
    print('p: %.3f r: %.3f' % (p, r))
    return tp, fp, tn, fn, p, r


def score_train_predictions(test_start_time):
    root = make_config_root(test_start_time, 'train_predict')
    results = {}
    for dk in CLASSIFIERS:
        d = DATA[dk]
        rpathp = lambda x: '%s%s/%s' % (root, x, d['pos'])
        rpathn = lambda x: '%s%s/%s' % (root, x, d['neg'])
        out = _score_train_prediction(rpathp('test_predict'), rpathn('test_predict'), d['pos'])
        results[dk] = out
    return results


def predict(train_start_time, hdfs_record_input_path):
    if SKIP_OVERRIDE and OVERRIDE_PREDICT_START_TIME:
        return OVERRIDE_PREDICT_START_TIME
    class_image_hashes = {}  # [class_name] = (neg_set, pos_set)
    class_image_paths = {}  # [class_name] = (data_path, feat_path)
    start_time = OVERRIDE_PREDICT_START_TIME if OVERRIDE_PREDICT_START_TIME else '%f' % time.time()
    train_root = make_config_root(train_start_time, 'train')
    root = make_drive_root(start_time, 'predict')
    # Convert
    hdfs_input_path = '%sdata/input' % root
    picarus.io.run_record_to_kv(hdfs_record_input_path, hdfs_input_path)
    # Predict photos
    d = DATA['photos']
    tpathp = lambda x: '%s%s/%s' % (train_root, x, d['pos'])
    rpathp = lambda x: '%s%s/%s' % (root, x, d['pos'])
    feat_input_path = '%sfeat/input' % root
    picarus.vision.run_image_feature(hdfs_input_path, feat_input_path, d['feature'], d['image_length'])
    picarus.classify.run_predict_classifier(feat_input_path, tpathp('classifiers'), rpathp('predict'))
    class_image_hashes[d['pos']] = picarus.classify.thresh_predictions(rpathp('predict'), d['pos'], 0.)
    class_image_paths[d['pos']] = hdfs_input_path, feat_input_path
    # Predict photo and split images/features for photos subclasses
    for photo_subclass in PHOTOS_SUBCLASSES:
        d = DATA[photo_subclass]
        tpathp = lambda x: '%s%s/%s' % (train_root, x, d['pos'])
        rpathp = lambda x: '%s%s/%s' % (root, x, d['pos'])
        picarus.classify.run_predict_classifier(feat_input_path, tpathp('classifiers'),
                                                rpathp('predict'), image_hashes=class_image_hashes['photos'][1])
        class_image_hashes[d['pos']] = picarus.classify.thresh_predictions(rpathp('predict'), d['pos'], 0.)
        class_image_paths[d['pos']] = class_image_paths['photos']
    # Find faces (another round of classification after the intial detection)
    d = DATA['detected_faces']
    tpathp = lambda x: '%s%s/%s' % (train_root, x, d['pos'])
    rpathp = lambda x: '%s%s/%s' % (root, x, d['pos'])
    data_faces_path = rpathp('data')
    # NOTE(brandyn): The face images are storted at data_faces_path which are cropped portions of the originals,
    # thus the image hashes that they provide are necessarily disjoint from the originals.
    picarus.vision.run_face_finder(hdfs_input_path, data_faces_path, image_length=d['image_length'], boxes=False,
                                   image_hashes=class_image_hashes['photos'][1])  # Reject more faces first
    picarus.vision.run_image_feature(data_faces_path, rpathp('feat'), d['feature'], d['image_length'])
    picarus.classify.run_predict_classifier(rpathp('feat'), tpathp('classifiers'), rpathp('predict'))
    class_image_hashes[d['pos']] = picarus.classify.thresh_predictions(rpathp('predict'), d['pos'], 0.)
    class_image_paths[d['pos']] = data_faces_path, rpathp('feat')
    # Compute the eigenface feature
    d = DATA['faces']
    tpathp = lambda x: '%s%s/%s' % (train_root, x, d['pos'])
    rpathp = lambda x: '%s%s/%s' % (root, x, d['pos'])
    picarus.vision.run_image_feature(data_faces_path, rpathp('feat'), d['feature'], d['image_length'],
                                     image_hashes=class_image_hashes['detected_faces'][1])
    # NOTE(brandyn): The hashes stay the say but we compute a new feature
    class_image_hashes[d['pos']] = class_image_hashes['detected_faces']
    class_image_paths[d['pos']] = data_faces_path, rpathp('feat')
    # Sample for initial clusters
    cluster(root, class_image_hashes, class_image_paths)
    if USE_FLOW:
        hadoopy_flow.joinall()  # We join here as later jobs use the result of these greenlets
    return start_time


def cluster(root, class_image_hashes, class_image_paths):
    for (dk, pol) in CLUSTERS:
        d = DATA[dk]
        rpathp = lambda x: '%s%s/%s' % (root, x, d['pos'])
        rpathn = lambda x: '%s%s/%s' % (root, x, d['neg'])
        pols = {'pos': rpathp, 'neg': rpathn}
        for p in pol:
            print('Start Clustering[%s]' % d[p])
            picarus.cluster.run_whiten(class_image_paths[dk][1], pols[p]('whiten'),
                                       image_hashes=class_image_hashes[dk][int(p == 'pos')])  # TODO Filter
            picarus.cluster.run_sample(pols[p]('whiten'), pols[p]('cluster') + '/local_sample', NUM_LOCAL_SAMPLES)
            picarus.cluster.run_local_kmeans(pols[p]('cluster') + '/local_sample', pols[p]('cluster') + '/clust0', NUM_CLUSTERS)
            if USE_FLOW:
                hadoopy_flow.Greenlet(picarus.cluster.run_kmeans, pols[p]('whiten'), pols[p]('cluster') + '/clust0', class_image_paths[dk][0],
                                      pols[p]('cluster'), NUM_CLUSTERS, NUM_ITERS, NUM_OUTPUT_SAMPLES, 'l2sqr',
                                      image_hashes=class_image_hashes[dk][int(p == 'pos')]).start()
            else:
                picarus.cluster.run_kmeans(pols[p]('whiten'), pols[p]('cluster') + '/clust0', class_image_paths[dk][0],
                                           pols[p]('cluster'), NUM_CLUSTERS, NUM_ITERS, NUM_OUTPUT_SAMPLES, 'l2sqr',
                                           image_hashes=class_image_hashes[dk][int(p == 'pos')])
            print('Done Clustering[%s]' % d[p])


def run_videos(video_input):
    if SKIP_OVERRIDE and OVERRIDE_VIDEOS_START_TIME:
        return OVERRIDE_VIDEOS_START_TIME
    start_time = OVERRIDE_VIDEOS_START_TIME if OVERRIDE_VIDEOS_START_TIME else '%f' % time.time()
    root = make_drive_root(start_time, 'video')
    picarus.vision.run_video_keyframe(video_input, root + 'video_keyframe/', min_interval=3.0, resolution=1.0, ffmpeg=True)
    return start_time


def report_clusters_faces_videos(predict_start_time, video_start_time):
    """
    """
    if SKIP_OVERRIDE and OVERRIDE_REPORT_START_TIME:
        return OVERRIDE_REPORT_START_TIME
    root = make_drive_root(predict_start_time, 'predict')
    start_time = OVERRIDE_REPORT_START_TIME if OVERRIDE_REPORT_START_TIME else '%f' % time.time()
    video_root = make_drive_root(video_start_time, 'video')
    out_root = make_drive_root(start_time, 'report')
    local = make_local_root(start_time)
    clusters = ['indoors', 'nonphotos', 'outdoors', 'objects', 'pr0n']
    clusters += ['faces']

    # Process all the thumbnails in parallel
    thumb_input = [root + '/cluster/' + c + '/partition' for c in clusters]
    picarus.report.make_thumbnails(thumb_input, out_root + '/report/thumb', 100, 'cluster')
    if video_root is not None:
        picarus.report.make_thumbnails(video_root + '/video_keyframe/allframes', out_root + '/report/vidthumb', 100, 'frame')

    # Prepare json report
    report = {}
    for c in clusters:
        make_faces = 'faces' in c
        r = picarus.report.report_clusters(root + '/cluster/' + c, c, make_faces)
        report.update(r)

    # Copy all the thumbnails locally
    picarus.report.report_thumbnails(out_root + '/report/thumb', local + '/report/t/')
    if video_root is not None:
        r = picarus.report.report_video_keyframe(video_root + '/video_keyframe/keyframe')
        report.update(r)
        picarus.report.report_thumbnails(out_root + '/report/vidthumb', local + '/report/t/')

    with open(local + '/report/sample_report.js', 'w') as f:
        f.write('var report = ')
        f.write(json.dumps(report))

    shutil.copy(picarus.report.__path__[0] + '/data/static_sample_report.html', local + '/report')
    hadoopy.put(local + '/report', out_root + '/report/')
    print('Report output ------------------> [%s/%s]' % (out_root, '/report'))
    return start_time


def main():
    global DRIVE_MD5, data_root
    args = _parser()
    DRIVE_MD5 = args.drive_md5
    video_input_paths = args.video_path
    graphic_input_paths = args.graphic_path
    data_root = args.training_data
    if data_root and data_root[-1] != '/':
        data_root += '/'
    if not args.video_path or not args.graphic_path:
        raise ValueError('At least one video and one graphic path is required')
    dump_out = {}
    if not args.train_start_time:
        if not data_root:
            raise ValueError('Training data path needs to be set!')
        train_start_time = train()
    else:
        train_start_time = args.train_start_time
    print('Ran: TRAIN_START_TIME[%s]' % train_start_time)
    if args.train_predict:
        if not data_root:
            raise ValueError('Training data path needs to be set!')
        train_predict_start_time = train_predict(train_start_time)
        print('Ran: TRAIN_PREDICT_START_TIME[%s]' % train_predict_start_time)
        test_results = score_train_predictions(train_predict_start_time)
        dump_out['train_predict_start_time'] = train_predict_start_time
        dump_out['test_results'] = test_results
    video_start_time = run_videos(video_input_paths)
    #print('Ran: VIDEO_START_TIME[%s]' % video_start_time)
    predict_start_time = predict(train_start_time, graphic_input_paths)
    print('Ran: PREDICT_START_TIME[%s]' % predict_start_time)
    report_start_time = report_clusters_faces_videos(predict_start_time, video_start_time)
    print('Ran: REPORT_START_TIME[%s]' % report_start_time)
    dump_settings(train_start_time=train_start_time,
                  video_start_time=video_start_time,
                  predict_start_time=predict_start_time,
                  report_start_time=report_start_time, **dump_out)


def _parser():
    parser = argparse.ArgumentParser(description='Texas Pete Graphics Pipeline')
    parser.add_argument('drive_md5', help='MD5 of the drive (used for placing outputs)')
    parser.add_argument('--graphic_path', help='HDFS Path to a file or directory of sequence files in the "record" form. (can use multiple)', action='append')
    parser.add_argument('--video_path', help='HDFS Path to a file or directory of sequence files in the "record" form. (can use multiple)', action='append')
    parser.add_argument('--train_predict', help='Run the classifiers on testing data', action='store_true')
    parser.add_argument('--train_start_time', help='If set then use this instead of training a new model')
    parser.add_argument('--training_data', help='Path to training data')
    return parser.parse_args()


if __name__ == '__main__':
    main()
