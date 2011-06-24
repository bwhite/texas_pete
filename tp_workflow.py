import hadoopy_flow
import hadoopy
import picarus
import time


def run_videos():
    start_time = time.time()
    #start_time = 1308792496.427986
    root = '/user/amiller/tp/video_keyframe/run-%f/' % start_time
    picarus.vision.run_video_keyframe('/user/brandyn/videos_small', root + 'video_keyframe/', 30, 3.0, ffmpeg=False)

    # Make the thumbnails (this parallelizes)
    for tag in ['photos', 'nonphotos']:
        picarus.report.make_thumbnails(data_root + 'test_' + tag, root + '/thumbs/' + tag, 100)
    picarus.report.make_thumbnails(root + 'video_keyframe/keyframes', root + 'video_keyframe/thumbs', 100)


# HDFS Paths with data of the form (unique_string, binary_image_data
data_root = '/user/brandyn/classifier_data/'


def make_reports():
    pass

FEATURE = 'meta_gist_spatial_hist'
DATA = {'photos': {'pos': 'photos',
                   'neg': 'nonphotos',
                   'feature': FEATURE,
                   'image_length': 256,
                   'classifier': 'svmlinear',
                   'labels_name': 'tp_indoor_labels.js'},
        'indoors': {'pos': 'indoors',
                    'neg': 'outdoors',
                    'feature': FEATURE,
                    'image_length': 256,
                    'classifier': 'svmlinear'},
        'objects': {'pos': 'objects',
                    'neg': 'nonobjects',
                    'feature': FEATURE,
                    'image_length': 256,
                    'classifier': 'svmlinear'},
        'detected_faces': {'pos': 'detected_faces',
                           'neg': 'detected_nonfaces',
                           'feature': FEATURE,
                           'image_length': 256,
                           'classifier': 'svmlinear'},
        'faces': {'pos': 'faces',
                  'neg': 'nonfaces',
                  'feature': 'eigenface',
                  'image_length': 64,
                  'classifier': 'svmlinear'},
        'pr0n': {'pos': 'pr0n',
                 'neg': 'nonpr0n',
                 'feature': FEATURE,
                 'image_length': 256,
                 'classifier': 'svmlinear'}}


CLASSIFIERS = ['photos', 'indoors', 'objects', 'detected_faces', 'pr0n']   # A classifier is learned for each of those
CLUSTERS = [('photos', ['pos', 'neg']), ('indoors', ['pos', 'neg']),
            ('objects', ['pos']), ('faces', ['pos']), ('pr0n', ['pos'])]   # Clustering is performed on each of those
PHOTOS_SUBCLASSES = ['indoors', 'objects', 'pr0n']  # Each of these are derived from predicted photos


def train():
    # HDFS Paths for Output
    start_time = '%f' % time.time()
    root = '/user/brandyn/tp/image_cluster/run-%s/' % start_time

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
    train_root = '/user/brandyn/tp/image_cluster/run-%s/' % train_start_time
    root = '/user/brandyn/tp/image_cluster/run-%f/' % start_time

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


def score_train_predictions(test_start_time='1308630752.962982'):
    root = '/user/brandyn/tp/image_cluster/run-%s/' % test_start_time
    for dk in CLASSIFIERS:
        d = DATA[dk]
        rpathp = lambda x: '%s%s/%s' % (root, x, d['pos'])
        rpathn = lambda x: '%s%s/%s' % (root, x, d['neg'])
        _score_train_prediction(rpathp('test_predict'), rpathn('test_predict'), d['pos'])


def predict(train_start_time, hdfs_input_path):
    # NOTE(brandyn): This assumes that they all use the same feature
    train_root = '/user/brandyn/tp/image_cluster/run-%s/' % train_start_time
    start_time = time.time()
    root = '/user/brandyn/tp/image_cluster/run-%s/' % start_time
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
    picarus.classify.run_thresh_predictions(rpathp('predict'), feat_input_path, feat_photos_path, d['pos'], 0., 1)
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
    picarus.vision.run_image_feature(rpathp('data'), rpathp('feat'), d['feature'], d['image_lenth'])
    picarus.classify.run_predict_classifier(rpathp('feat'), tpathp('classifiers'), rpathp('predict'))
    picarus.classify.run_thresh_predictions(rpathp('predict'), rpathp('data'), data_faces_path, d['pos'], 0., 1)
    # Compute the eigenface feature
    d = DATA['faces']
    tpathp = lambda x: '%s%s/%s' % (train_root, x, d['pos'])
    rpathp = lambda x: '%s%s/%s' % (root, x, d['pos'])
    picarus.vision.run_image_feature(data_faces_path, rpathp('feat'), d['feature'], d['image_length'])
    # Sample for initial clusters
    num_clusters = 10
    num_iters = 5
    num_output_samples = 10
    for (dk, pol) in CLUSTERS:
        d = DATA[dk]
        rpathp = lambda x: '%s%s/%s' % (root, x, d['pos'])
        rpathn = lambda x: '%s%s/%s' % (root, x, d['neg'])
        pols = {'pos': rpathp, 'neg': rpathn}
        for p in pol:
            picarus.cluster.run_whiten(pols[p]('feat'), pols[p]('whiten'))
            picarus.cluster.run_sample(pols[p]('whiten'), pols[p]('cluster') + '/clust0', num_clusters)
            hadoopy_flow.Greenlet(picarus.cluster.run_kmeans, pols[p]('whiten'), pols[p]('cluster') + '/clust0', pols[p]('data'),
                                  pols[p]('cluster'), num_clusters, num_iters, num_output_samples, 'l2sqr').start()

if __name__ == '__main__':
    train_start_time = train()
    test_start_time = train_predict(train_start_time)
    score_train_predictions(test_start_time)
    #train_start_time = '1308644265.354146'
    #test_start_time = '1308650330.016147'
    print('TrainStart[%s] TestStart[%s]' % (train_start_time, test_start_time))
    #run_videos()
    predict(train_start_time, '/user/brandyn/classifier_data/unlabeled_flickr_small')
