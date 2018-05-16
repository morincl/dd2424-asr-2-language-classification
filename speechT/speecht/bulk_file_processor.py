# Copyright 2016 Louis Kirsch. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Dict

import editdistance
import numpy as np
import tensorflow as tf
import speecht.vocabulary
from speecht.execution import DatasetExecutor

from speecht.speech_model import SpeechModel
import itertools

class Evaluation(DatasetExecutor):

  def create_sample_generator(self, limit_count: int):
    return self.reader.load_samples(self.flags.dataset,
                                    loop_infinitely=False,
                                    limit_count=limit_count,
                                    feature_type=self.flags.feature_type)

  def get_loader_limit_count(self):
    return self.flags.step_count * self.flags.batch_size

  def get_max_steps(self):
    if self.flags.step_count:
      return self.flags.step_count
    return None

  def run(self):

    with tf.Session() as sess:

      model = self.create_model(sess)

      print('Starting input pipeline')
      coord = self.start_pipeline(sess)

      try:
        print('Begin evaluation')
        self.run_step(model, sess, output_file=self.flags.output_file)

      except tf.errors.OutOfRangeError:
        print('Done evaluating -- step limit reached')
      finally:
        coord.request_stop()
      coord.join()

  def run_step(self, model: SpeechModel, sess: tf.Session, output_file="nofile", feed_dict: Dict=None):
    global_step = model.global_step.eval()

    # Validate on data set and write summary
    output = model.step(sess, loss=False, update=False, decode=False, return_label=False, summary=False, identity=True, feed_dict=feed_dict)


    if output_file != "nofile":
      np.save(output_file, output)
