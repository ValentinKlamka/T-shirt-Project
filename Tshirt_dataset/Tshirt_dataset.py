"""my_dataset dataset."""
from typing import Dict, Iterator, Tuple
import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
import re
from pathlib import Path

# TODO(Tshirt_dataset): Markdown description  that will appear on the catalog page.
_DESCRIPTION = """
Description is **formatted** as markdown.

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

# TODO(Tshirt_dataset): BibTeX citation
_CITATION = """
"""


class TshirtDataset(tfds.core.GeneratorBasedBuilder):
  """DatasetBuilder for Tshirt_dataset dataset."""

  VERSION = tfds.core.Version('1.0.2')
  RELEASE_NOTES = {
    '1.0.2':'added testset',
        '1.0.1': 'smaller coords',
      '1.0.0': 'Initial release.',
  }
  coordsdict={}

  def _info(self) -> tfds.core.DatasetInfo:
    """Returns the dataset metadata."""
    # TODO(my_dataset): Specifies the tfds.core.DatasetInfo object
    return tfds.core.DatasetInfo(
        builder=self,
        description=_DESCRIPTION,
        features=tfds.features.FeaturesDict({
            # These are the features of your dataset like images, labels ...
            'image': tfds.features.Image(shape=(None, None, 3)),
            'label':  tfds.features.Tensor(shape=(None,3),dtype=tf.float32),
        }),
        # If there's a common (input, target) tuple from the
        # features, specify them here. They'll be used if
        # `as_supervised=True` in `builder.as_dataset`.
        supervised_keys=('image', 'label'),  # Set to `None` to disable
        homepage=None,
        citation=_CITATION,
    )  

  def _split_generators(self,blub):
    """Returns SplitGenerators."""

    path = Path('../img_train')
    path2= Path('../coords_train')
    path3= Path('../img_test')
    path4=Path('../coords_test')

    return {
        'train': self._generate_examples(path, path2),
        'test': self._generate_examples(path3, path4)
    }

  def _generate_examples(self, path, path2) -> Iterator[Tuple[str, Dict[np.ndarray, tf.Tensor]]]:
        """Generator of examples for each split."""
        for img_path in path.glob('*.png'):
            print(img_path)
            print(img_path.name)

            id=re.split('\.+',img_path.name)[0]
            #find id in path2
            lines=None
            for coords in path2.glob('vertexcoords_'+id +'.txt'):
                if coords.name not in TshirtDataset.coordsdict:
                    with tf.io.gfile.GFile(coords) as f:
                        lines = f.readlines()
                        lines = [line.strip() for line in lines]
                        #remove brackets
                        lines = [line.replace('(','') for line in lines]
                        lines = [line.replace(')','') for line in lines]
                        lines = [line.split(',') for line in lines]
                        lines = [[float(x) for x in line] for line in lines]
                        lines = np.array(lines)
                        lines = np.float32(lines)
                        TshirtDataset.coordsdict[coords.name]=lines
                    break  #we need only one iteration, even if there are more (which are not)


            # Yields (key, example)
            yield img_path.name, {
                'image': img_path,
                'label': TshirtDataset.coordsdict[coords.name],
            }

