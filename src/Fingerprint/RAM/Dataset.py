import numpy as np
import cv2
from RAM.config import Config


class DataSet(object):

    def __init__(self,
              images,
              labels):

        assert images.shape[0] == labels.shape[0], (
         'images.shape: %s labels.shape: %s' % (images.shape, labels.shape))
        self._num_examples = images.shape[0]

        #images = self.convert_images(images)

        if len(labels.shape)==2:
           labels = labels.reshape((labels.shape[0],))
        images = images.reshape((images.shape[0],-1))


        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def convert_images(self,images):
        result = []
        config = Config()
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img,(config.original_size,config.original_size))
            result.append(img)
        return np.array(result)

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
             # Finished epoch
             self._epochs_completed += 1
             # Shuffle the data
             perm = np.arange(self._num_examples)
             np.random.shuffle(perm)
             self._images = self._images[perm]
             self._labels = self._labels[perm]
             # Start next epoch
             start = 0
             self._index_in_epoch = batch_size
             assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end], self._labels[start:end]