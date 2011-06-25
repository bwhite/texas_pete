import hadoopy
import glob
import os
import hashlib
import random


hdfs_root = 'classifier_data'
local_root = '/home/tp/data/classifier_data'
pct_train = .9


def read_files(fns, prev_hashes):
    for fn in fns:
        data = open(fn).read()
        data_hash = hashlib.md5(data).hexdigest()
        if data_hash not in prev_hashes:
            prev_hashes.add(data_hash)
            yield data_hash, data

# Write vidoes
videos = ['aladdin_videos', 'youtube_action_dataset']
for video_name in videos:
    fns = glob.glob('%s/%s/*' % (local_root, video_name))
    random.shuffle(fns)
    prev_hashes = set()
    hadoopy.writetb('%s/video_%s' % (hdfs_root, video_name), read_files(fns, prev_hashes))
    print('Unlabeled:[%s] Num[%d]' % (video_name, len(prev_hashes)))

# Write unabled data (used for evaluation)
unlabeled = ['flickr', 'flickr_small']
for unlabeled_name in unlabeled:
    fns = glob.glob('%s/%s/*' % (local_root, unlabeled_name))
    random.shuffle(fns)
    prev_hashes = set()
    hadoopy.writetb('%s/unlabeled_%s' % (hdfs_root, unlabeled_name), read_files(fns, prev_hashes))
    print('Unlabeled:[%s] Num[%d]' % (unlabeled_name, len(prev_hashes)))


# Write train/test
data_pairs = [('detected_faces', 'detected_nonfaces'), ('photos', 'nonphotos'), ('indoors', 'outdoors'), ('pr0n', 'nonpr0n'), ('objects', 'nonobjects')]
for pos_name, neg_name in data_pairs:
    pos_fns = glob.glob('%s/%s/*' % (local_root, pos_name))
    neg_fns = glob.glob('%s/%s/*' % (local_root, neg_name))
    random.shuffle(pos_fns)
    random.shuffle(neg_fns)
    num_train = int(min(len(neg_fns), len(pos_fns)) * pct_train)
    prev_hashes = set()
    # Pos
    hadoopy.writetb('%s/test_%s' % (hdfs_root, pos_name), read_files(pos_fns[num_train:], prev_hashes))
    print(len(prev_hashes))
    num_pos_test = len(prev_hashes)
    hadoopy.writetb('%s/train_%s' % (hdfs_root, pos_name), read_files(pos_fns[:num_train], prev_hashes))
    print(len(prev_hashes))
    num_pos_train = len(prev_hashes) - num_pos_test
    # Neg
    hadoopy.writetb('%s/test_%s' % (hdfs_root, neg_name), read_files(neg_fns[num_train:], prev_hashes))
    print(len(prev_hashes))
    num_neg_test = len(prev_hashes) - (num_pos_train + num_pos_test)
    hadoopy.writetb('%s/train_%s' % (hdfs_root, neg_name), read_files(neg_fns[:num_train], prev_hashes))
    print(len(prev_hashes))
    num_neg_train = len(prev_hashes) - (num_pos_train + num_pos_test + num_neg_test)
    print('+:[%s] Train[%d] Test[%d]' % (pos_name, num_pos_train, num_pos_test))
    print('-:[%s] Train[%d] Test[%d]' % (neg_name, num_neg_train, num_neg_test))


