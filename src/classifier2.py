from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import LabelEncoder

def main(args):
    # Create a new TensorFlow computation graph
    with tf.Graph().as_default():
        # Initialize a TensorFlow session
        with tf.compat.v1.Session() as sess:
            # Set random seed for reproducibility
            np.random.seed(seed=args.seed)

            # Load dataset
            dataset, labels = load_images_from_folder(args.data_dir)
            
            # Check that there is at least one training image for each class
            assert len(dataset) > 0, 'There must be at least one image in the dataset'

            # Get image paths and labels
            paths = dataset

            print('Number of images: %d' % len(paths))

            # Load the pre-trained model
            print('Loading feature extraction model')
            facenet.load_model(args.model)

            # Get input and output tensors
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Perform forward pass to calculate embeddings
            print('Calculating features for images')
            nrof_images = len(paths)
            nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / args.batch_size))
            emb_array = np.zeros((nrof_images, embedding_size))
            for i in range(nrof_batches_per_epoch):
                start_index = i * args.batch_size
                end_index = min((i + 1) * args.batch_size, nrof_images)
                paths_batch = paths[start_index:end_index]
                images = facenet.load_data(paths_batch, False, False, args.image_size)
                feed_dict = { images_placeholder: images, phase_train_placeholder: False }
                emb_array[start_index:end_index, :] = sess.run(embeddings, feed_dict=feed_dict)

            classifier_filename_exp = os.path.expanduser(args.classifier_filename)

            if args.mode == 'TRAIN':
                # Check if the classifier model already exists
                if os.path.exists(classifier_filename_exp):
                    print('Loading existing classifier model')
                    with open(classifier_filename_exp, 'rb') as infile:
                        model = pickle.load(infile)
                else:
                    model = OneClassSVM(kernel='linear', gamma='auto')
                
                # Train classifier using One-Class SVM
                print('Training classifier')
                model.fit(emb_array)

                # Save classifier model
                with open(classifier_filename_exp, 'wb') as outfile:
                    pickle.dump(model, outfile)
                print('Saved classifier model to file "%s"' % classifier_filename_exp)

            elif args.mode == 'CLASSIFY':
                # Classify images
                print('Testing classifier')
                with open(classifier_filename_exp, 'rb') as infile:
                    model = pickle.load(infile)

                print('Loaded classifier model from file "%s"' % classifier_filename_exp)

                predictions = model.predict(emb_array)
                for i, pred in enumerate(predictions):
                    if pred == 1:
                        print('%4d  %s: %.3f' % (i, 'In Class', 1.0))
                    else:
                        print('%4d  %s: %.3f' % (i, 'Out of Class', 0.0))

                accuracy = np.mean(predictions == 1)
                print('Accuracy: %.3f' % accuracy)

def load_images_from_folder(folder):
    dataset = []
    labels = []
    # Create a list of image paths
    image_paths = [os.path.join(folder, img) for img in os.listdir(folder) if img.endswith('.jpg') or img.endswith('.png')]
    # Assign a single class label for all images
    labels = [0] * len(image_paths)
    return image_paths, labels

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, choices=['TRAIN', 'CLASSIFY'],
                        help='Indicates if a new classifier should be trained or an existing classifier should be used to classify images', default='CLASSIFY')
    parser.add_argument('data_dir', type=str,
                        help='Path to the data directory containing face images.')
    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta file and the ckpt file or a model protobuf (.pb) file')
    parser.add_argument('classifier_filename',
                        help='Classifier model file name as a pickle (.pkl) file. For training, this is the output and for classification, this is an input.')
    parser.add_argument('--batch_size', type=int,
                        help='Number of images to process in a batch.', default=90)
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--seed', type=int,
                        help='Random seed.', default=666)

    return parser.parse_args(argv)

# Run the main function if the script is executed directly
if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
