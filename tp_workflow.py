import hadoopy_flow
import hadoopy
import picarus
import time
import json
import shutil


# HDFS Paths with data of the form (unique_string, binary_image_data
data_root = '/user/brandyn/classifier_data/'
FEATURE = 'hist_joint'  # meta_gist_spatial_hist
IMAGE_LENGTH = 128
DATA = {'photos': {'pos': 'photos',
                   'neg': 'nonphotos',
                   'feature': FEATURE,
                   'image_length': IMAGE_LENGTH,
                   'classifier': 'svmlinear',
                   'labels_name': 'tp_indoor_labels.js'},
        'indoors': {'pos': 'indoors',
                    'neg': 'outdoors',
                    'feature': FEATURE,
                    'image_length': IMAGE_LENGTH,
                    'classifier': 'svmlinear'},
        'objects': {'pos': 'objects',
                    'neg': 'nonobjects',
                    'feature': FEATURE,
                    'image_length': IMAGE_LENGTH,
                    'classifier': 'svmlinear'},
        'detected_faces': {'pos': 'detected_faces',
                           'neg': 'detected_nonfaces',
                           'feature': FEATURE,
                           'image_length': IMAGE_LENGTH,
                           'classifier': 'svmlinear'},
        'faces': {'pos': 'faces',
                  'neg': 'nonfaces',
                  'feature': 'eigenface',
                  'image_length': 64,
                  'classifier': 'svmlinear'},
        'pr0n': {'pos': 'pr0n',
                 'neg': 'nonpr0n',
                 'feature': FEATURE,
                 'image_length': IMAGE_LENGTH,
                 'classifier': 'svmlinear'}}


CLASSIFIERS = ['photos', 'indoors', 'objects', 'detected_faces', 'pr0n']   # A classifier is learned for each of those
CLUSTERS = [('photos', ['pos', 'neg']), ('indoors', ['pos', 'neg']),
            ('objects', ['pos']), ('faces', ['pos']), ('pr0n', ['pos'])]   # Clustering is performed on each of those
PHOTOS_SUBCLASSES = ['indoors', 'objects', 'pr0n']  # Each of these are derived from predicted photos


def make_root(start_time):
    return 'tp/image_cluster/run-%s/' % start_time


def make_local_root(start_time):
    return 'out/run-%s/' % start_time


def dump_settings(train_start_time):
    with open('%s.js' % train_start_time, 'w') as fp:
        g = globals()
        out = dict((x, g[x]) for x in ('CLASSIFIERS', 'CLUSTERS', 'PHOTOS_SUBCLASSES', 'DATA'))
        json.dump(out, fp, -1)


def train():
    # HDFS Paths for Output
    start_time = '%f' % time.time()
    root = make_root(start_time)

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


def train_predict(train_start_time='1308626598.185418'):
    start_time = time.time()
    train_root = make_root(train_start_time)
    root = make_root(start_time)

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
    return '%f' % start_time


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
    print('%.3f %.3f' % (tp / float(tp + fn), tp / float(tp + fp)))
    return tp, fp, tn, fn


def score_train_predictions(test_start_time):
    root = make_root(test_start_time)
    for dk in CLASSIFIERS:
        d = DATA[dk]
        rpathp = lambda x: '%s%s/%s' % (root, x, d['pos'])
        rpathn = lambda x: '%s%s/%s' % (root, x, d['neg'])
        _score_train_prediction(rpathp('test_predict'), rpathn('test_predict'), d['pos'])


def predict(train_start_time, hdfs_input_path):
    # NOTE(brandyn): This assumes that they all use the same feature
    train_root = make_root(train_start_time)
    start_time = '%f' % time.time()
    root = make_root(start_time)
    # Predict photos
    d = DATA['photos']
    tpathp = lambda x: '%s%s/%s' % (train_root, x, d['pos'])
    rpathp = lambda x: '%s%s/%s' % (root, x, d['pos'])
    rpathn = lambda x: '%s%s/%s' % (root, x, d['neg'])
    feat_input_path = '%sfeat/input' % root
    picarus.vision.run_image_feature(hdfs_input_path, feat_input_path, d['feature'], d['image_length'])
    picarus.classify.run_predict_classifier(feat_input_path, tpathp('classifiers'), rpathp('predict'))
    # Split images for photos/nonphotos
    picarus.classify.run_thresh_predictions(rpathp('predict'), hdfs_input_path, rpathp('data'), d['pos'], 0., 1)
    picarus.classify.run_thresh_predictions(rpathp('predict'), hdfs_input_path, rpathn('data'), d['pos'], 0., -1)
    data_photos_path = rpathp('data')
    # Split features for photos
    feat_photos_path = rpathp('feat')
    picarus.classify.run_thresh_predictions(rpathp('predict'), feat_input_path, rpathp('feat'), d['pos'], 0., 1)
    picarus.classify.run_thresh_predictions(rpathp('predict'), feat_input_path, rpathn('feat'), d['pos'], 0., -1)
    # Predict photo and split images/features for photos subclasses
    for photo_subclass in PHOTOS_SUBCLASSES:
        d = DATA[photo_subclass]
        tpathp = lambda x: '%s%s/%s' % (train_root, x, d['pos'])
        rpathp = lambda x: '%s%s/%s' % (root, x, d['pos'])
        picarus.classify.run_predict_classifier(feat_photos_path, tpathp('classifiers'), rpathp('predict'))
        picarus.classify.run_thresh_predictions(rpathp('predict'), data_photos_path, rpathp('data'), d['pos'], 0., 1)
        picarus.classify.run_thresh_predictions(rpathp('predict'), data_photos_path, rpathn('data'), d['pos'], 0., -1)
        picarus.classify.run_thresh_predictions(rpathp('predict'), feat_photos_path, rpathp('feat'), d['pos'], 0., 1)
        picarus.classify.run_thresh_predictions(rpathp('predict'), feat_photos_path, rpathn('feat'), d['pos'], 0., -1)
    # Find faces (another round of classification after the intial detection)
    d = DATA['detected_faces']
    tpathp = lambda x: '%s%s/%s' % (train_root, x, d['pos'])
    rpathp = lambda x: '%s%s/%s' % (root, x, d['pos'])
    data_faces_path = '%sdata/faces' % root
    picarus.vision.run_face_finder(data_photos_path, rpathp('data'), image_length=d['image_length'], boxes=False)  # Reject more faces first
    picarus.vision.run_image_feature(rpathp('data'), rpathp('feat'), d['feature'], d['image_length'])
    picarus.classify.run_predict_classifier(rpathp('feat'), tpathp('classifiers'), rpathp('predict'))
    picarus.classify.run_thresh_predictions(rpathp('predict'), rpathp('data'), data_faces_path, d['pos'], 0., 1)
    # Compute the eigenface feature
    d = DATA['faces']
    tpathp = lambda x: '%s%s/%s' % (train_root, x, d['pos'])
    rpathp = lambda x: '%s%s/%s' % (root, x, d['pos'])
    picarus.vision.run_image_feature(data_faces_path, rpathp('feat'), d['feature'], d['image_length'])
    # Sample for initial clusters
    cluster(root)
    return start_time


def cluster(root):
    num_clusters = 20
    num_iters = 5
    num_output_samples = 10
    for (dk, pol) in CLUSTERS:
        d = DATA[dk]
        rpathp = lambda x: '%s%s/%s' % (root, x, d['pos'])
        rpathn = lambda x: '%s%s/%s' % (root, x, d['neg'])
        pols = {'pos': rpathp, 'neg': rpathn}
        for p in pol:
            print('Start Clustering[%s]' % d[p])
            picarus.cluster.run_whiten(pols[p]('feat'), pols[p]('whiten'))
            picarus.cluster.run_sample(pols[p]('whiten'), pols[p]('cluster') + '/clust0', num_clusters)
            hadoopy_flow.Greenlet(picarus.cluster.run_kmeans, pols[p]('whiten'), pols[p]('cluster') + '/clust0', pols[p]('data'),
                                  pols[p]('cluster'), num_clusters, num_iters, num_output_samples, 'l2sqr').start()
            print('Done Clustering[%s]' % d[p])


def run_videos(video_input):
    start_time = '%f' % time.time()
    root = make_root(start_time)
    picarus.vision.run_video_keyframe(video_input, root + 'video_keyframe/', 30, 3.0, ffmpeg=True)

    # Make the thumbnails (this parallelizes)
    #for tag in ['photos', 'nonphotos']:
    #    picarus.report.make_thumbnails(data_root + 'test_' + tag, root + '/thumbs/' + tag, 100)
    #picarus.report.make_thumbnails(root + 'video_keyframe/keyframes', root + 'video_keyframe/thumbs', 100)
    return start_time


def report_clusters_faces_videos(predict_start_time, video_start_time):
    """
    """
    root = make_root(predict_start_time)
    start_time = '%f' % time.time()
    video_root = make_root(video_start_time)
    out_root = make_root(start_time)
    local = make_local_root(start_time)
    clusters = ['indoors', 'nonphotos', 'outdoors', 'objects', 'pr0n']
    clusters += ['faces']

    # Process all the thumbnails in parallel
    thumb_input = [root + 'cluster/' + c + '/samples' for c in clusters]
    picarus.report.make_thumbnails(thumb_input, out_root + 'report/thumb', 100, is_cluster=True)
    if video_root is not None:
        picarus.report.make_thumbnails(video_root + 'video_keyframe/samples', out_root + 'report/vidthumb', 100)

    # Prepare json report
    report = {}
    for c in clusters:
        make_faces = 'faces' in c
        r = picarus.report.report_clusters(root + 'cluster/' + c + '/samples', 100, c, make_faces)
        report.update(r)

    # Copy all the thumbnails locally
    picarus.report.report_thumbnails(out_root + 'report/thumb', local + 'report/t/')
    if video_root is not None:
        r = picarus.report.report_video_keyframe(video_root + 'video_keyframe/keyframe')
        report.update(r)
        picarus.report.report_thumbnails(out_root + 'report/vidthumb', local + 'report/t/')

    with open(local + 'report/sample_report.js', 'w') as f:
        f.write('var report = ')
        f.write(json.dumps(report))

    shutil.copy(picarus.report.__path__[0] + '/data/static_sample_report.html', local + 'report')

if __name__ == '__main__':
    #train_start_time = train()
    #test_start_time = train_predict(train_start_time)
    #score_train_predictions(test_start_time)
    train_start_time = '1308973654.680628'  # '1308644265.354146'
    dump_settings(train_start_time)
    #test_start_time = '1308650330.016147'
    #print('TrainStart[%s] TestStart[%s]' % (train_start_time, test_start_time))
    video_start_time = run_videos('/user/brandyn/classifier_data/video_youtube_action_dataset')
    #predict_start_time = predict(train_start_time, '/user/brandyn/classifier_data/unlabeled_flickr')
    predict_start_time = '1308988799.817190'
    report_clusters_faces_videos(predict_start_time, video_start_time)
        
