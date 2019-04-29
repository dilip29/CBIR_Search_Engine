import cv2
import csv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageOps
from scipy.spatial import cKDTree
from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform
import tensorflow as tf
import tensorflow_hub as hub
import argparse
import pickle
import numpy as np
import glob
import os
from itertools import accumulate

def image_input_fn(image_files):
    filename_queue = tf.train.string_input_producer(
        image_files, shuffle=False)
    reader = tf.WholeFileReader()
    _, value = reader.read(filename_queue)
    image_tf = tf.image.decode_jpeg(value, channels=3)
    return tf.image.convert_image_dtype(image_tf, tf.float32)

def resize_image(srcfile, destfile, new_width=256, new_height=256):
    pil_image = Image.open(srcfile)
    pil_image = ImageOps.fit(pil_image, (new_width, new_height), Image.ANTIALIAS)
    pil_image_rgb = pil_image.convert('RGB')
    pil_image_rgb.save(destfile, format='JPEG', quality=90)
    return destfile

def resize_images_folder(srcfolder, new_width=256, new_height=256):
    destfolder = os.path.relpath(srcfolder) + '_resized'
    os.makedirs(destfolder,exist_ok=True)
    for srcfile in glob.iglob(os.path.join(srcfolder, '*.[Jj][Pp][Gg]')):
        src_basename = os.path.basename(srcfile)
        destfile=os.path.join(destfolder,src_basename)
        resize_image(srcfile, destfile, new_width, new_height)
    return destfolder

def get_resized_db_image_paths(destfolder='./images/resized'):
    return sorted(list(glob.iglob(os.path.join(destfolder, '*.[Jj][Pp][Gg]'))))

def compute_locations_and_descriptors_dir(image_path):
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.FATAL)

    m = hub.Module('https://tfhub.dev/google/delf/1')

    # The module operates on a single image at a time, so define a placeholder to
    # feed an arbitrary image in.
    image_placeholder = tf.placeholder(
        tf.float32, shape=(None, None, 3), name='input_image')

    module_inputs = {
        'image': image_placeholder,
        'score_threshold': 100.0,
        'image_scales': [0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0],
        'max_feature_num': 1000,
    }

    module_outputs = m(module_inputs, as_dict=True)

    image_tf = image_input_fn(image_path)

    with tf.train.MonitoredSession() as sess:
      results_dict = {}  # Stores the locations and their descriptors for each image
      for img_path in image_path:
        image = sess.run(image_tf)
        print('Extracting locations and descriptors from %s' % img_path)
        results_dict[img_path] = sess.run(
            [module_outputs['locations'], module_outputs['descriptors']],
            feed_dict={image_placeholder: image})
    return results_dict

# functios to extract feature of a single image
def compute_locations_and_descriptors(image_path):
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.FATAL)

    m = hub.Module('https://tfhub.dev/google/delf/1')

    # The module operates on a single image at a time, so define a placeholder to
    # feed an arbitrary image in.
    image_placeholder = tf.placeholder(
        tf.float32, shape=(None, None, 3), name='input_image')

    module_inputs = {
        'image': image_placeholder,
        'score_threshold': 100.0,
        'image_scales': [0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0],
        'max_feature_num': 1000,
    }

    module_outputs = m(module_inputs, as_dict=True)

    if type(image_path) == type([]):
        image_tf = image_input_fn(image_path)
    else:
        image_tf = image_input_fn([image_path])
    with tf.train.MonitoredSession() as sess:
        image = sess.run(image_tf)
        print('Extracting locations and descriptors from %s' % image_path)
        return sess.run(
            [module_outputs['locations'], module_outputs['descriptors']],
            feed_dict={image_placeholder: image})

def show_images(images, cols = 1):
    n_images = len(images)
    titles=[]
    titles.append("Query Image")
    for i in range(1,n_images + 1):
        titles.append('Result Image (%d)' % i)

    fig = plt.figure()
    for n,(image,title) in enumerate(zip(images,titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        a.axis('off')
        plt.imshow(image)

        a.set_title(title)
    print(np.array(fig.get_size_inches()) * (n_images))
    fig.set_size_inches(np.array(fig.get_size_inches()) * (n_images))
    plt.show()

def image_index_2_accumulated_indexes(index, accumulated_indexes_boundaries):
    '''
    Image index to accumulated/aggregated locations/descriptors pair indexes.
    '''
    if index > len(accumulated_indexes_boundaries) - 1:
        return None
    accumulated_index_start = None
    accumulated_index_end = None
    if index == 0:
        accumulated_index_start = 0
        accumulated_index_end = accumulated_indexes_boundaries[index]
    else:
        accumulated_index_start = accumulated_indexes_boundaries[index-1]
        accumulated_index_end = accumulated_indexes_boundaries[index]
    return np.arange(accumulated_index_start,accumulated_index_end)

def get_locations_2_use(image_db_index, k_nearest_indices, accumulated_indexes_boundaries, loc, loc_agg):
    '''
    Get a pair of locations to use, the query image to the database image with given index.
    Return: a tuple of 2 numpy arrays, the locations pair.
    '''
    image_accumulated_indexes = image_index_2_accumulated_indexes(image_db_index, accumulated_indexes_boundaries)
    locations_2_use_query = []
    locations_2_use_db = []
    for i, row in enumerate(k_nearest_indices):
        for acc_index in row:
            if acc_index in image_accumulated_indexes:
                locations_2_use_query.append(loc[i])
                locations_2_use_db.append(loc_agg[acc_index])
                break
    return np.array(locations_2_use_query), np.array(locations_2_use_db)

def preprocess_query_image(imagepath):
    '''
    Resize the query image and return the resized image path.
    '''
    query_temp_folder_name = '_resized'
    query_temp_folder = os.path.dirname(cmd_args.query)+ query_temp_folder_name
    os.makedirs(query_temp_folder,exist_ok=True)
    query_basename = os.path.basename(cmd_args.query)
    destfile=os.path.join(query_temp_folder,query_basename)
    resized_image = resize_image(cmd_args.query, destfile)
    return resized_image

def main():
    if not os.path.exists(cmd_args.saved):
        print('Resizing the database images to 256x256')
        destfolder = resize_images_folder(cmd_args.db)
        db_images = get_resized_db_image_paths(destfolder)
        print('Extracting features for all the images in database')
        results_dict = compute_locations_and_descriptors_dir(db_images)
        with open(cmd_args.saved, 'wb') as f:
            pickle.dump(results_dict, f)
    else:
        with open(cmd_args.saved, 'rb') as f:
            results_dict = pickle.load(f)
        db_images = list(results_dict.keys())

    locations_agg = np.concatenate([results_dict[img][0] for img in db_images])
    descriptors_agg = np.concatenate([results_dict[img][1] for img in db_images])
    accumulated_indexes_boundaries = list(accumulate([results_dict[img][0].shape[0] for img in db_images]))
    # build the KDTree with database image descriptors
    print('Building the tree from database images')
    d_tree = cKDTree(descriptors_agg)
    # preprocess the query image
    resized_image = preprocess_query_image(cmd_args.query)
    query_image_locations, query_image_descriptors = compute_locations_and_descriptors(resized_image)
    print('Querying the tree with query image')
    distance_threshold = 0.8
    # K nearest neighbors
    K = 10
    distances, indices = d_tree.query(
        query_image_descriptors, distance_upper_bound=distance_threshold, k = K, n_jobs=-1)
    # Find the list of unique accumulated/aggregated indexes
    unique_indices = np.array(list(set(indices.flatten())))
    unique_image_indexes = np.array(
    list(set([np.argmax([np.array(accumulated_indexes_boundaries)>index])
              for index in unique_indices])))
    # Array to keep track of all candidates in database.
    inliers_counts = []
    # Read the resized query image for plotting.
    img_1 = mpimg.imread(resized_image)
    for index in unique_image_indexes:
        locations_2_use_query, locations_2_use_db = get_locations_2_use(index,
                 indices, accumulated_indexes_boundaries, query_image_locations,
                 locations_agg)
        # Perform geometric verification using RANSAC.
        try:
            _, inliers = ransac(
                (locations_2_use_db, locations_2_use_query), # source and destination coordinates
                AffineTransform,
                min_samples=3,
                residual_threshold=20,
                max_trials=1000)
        except:
            continue
        # If no inlier is found for a database candidate image, we continue on to the next one.
        if inliers is None or len(inliers) == 0:
            continue
        # the number of inliers as the score for retrieved images.
        inliers_counts.append({"index": index, "inliers": sum(inliers)})
        print('Found inliers for image {} -> {}'.format(index, sum(inliers)))
    top_match = sorted(inliers_counts, key=lambda k: k['inliers'], reverse=True)[:cmd_args.top]
    images_set = []
    images_set.append(mpimg.imread(resized_image))
    results_path = [db_images[k['index']] for k in top_match]
    for result in results_path:
        # result_image=cv2.imread("Dataset/oxford5k_images/"+result[1])
        result_image = mpimg.imread(result)
        images_set.append(result_image)
    show_images(images_set, 4)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
      '--db',
      type=str,
      required=True,
      help="""
      Path to list of images of database whose features need to be extracted
      """)
    parser.add_argument(
      '--top',
      type=int,
      required=True,
      help="""
      No. of top results to get
      """)
    parser.add_argument(
      '--query',
      type=str,
      required=True,
      help="""
      Path to the query image
      """)
    parser.add_argument(
      '--saved',
      type=str,
      default='',
      help="""
      saved pickle file for the features of database images
      """)
    cmd_args, unparsed = parser.parse_known_args()
    main()
