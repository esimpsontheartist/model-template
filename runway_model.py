# MIT License

# Copyright (c) 2019 Runway AI, Inc

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# =========================================================================

# This example contains the minimum specifications and requirements
# to port a machine learning model to Runway.

# For more instructions on how to port a model to Runway, see the Runway Model
# SDK docs at https://sdk.runwayml.com

# RUNWAY
# www.runwayml.com
# hello@runwayml.com

# =========================================================================

# Import the Runway SDK. Please install it first with
# `pip install runway-python`.
import runway
from runway.data_types import number, text, image
from example_model import ExampleModel
import pickle
import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib
import os

fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

# Setup the model, initialize weights, set the configs of the model, etc.
# Every model will have a different set of configurations and requirements.
# Check https://docs.runwayapp.ai/#/python-sdk to see a complete list of
# supported configs. The setup function should return the model ready to be
# used.
setup_options = {
    'truncation': number(min=1, max=10, step=1, default=5, description='Example input.'),
    'seed': number(min=0, max=1000000, description='A seed used to initialize the model.')
}
# runway_model.py
@runway.setup(options={'checkpoint': runway.file(extension='.pkl')})
def setup(opts):
   checkpoint_path = opts['checkpoint']
   model = load_model_from_checkpoint(checkpoint_path)
    global Gs
    tflib.init_tf()
    with open(opts['checkpoint'], 'rb') as file:
        G, D, Gs = pickle.load(file, encoding='latin1')
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    rnd = np.random.RandomState()
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})
   return Gs
   return model

generate_inputs = {
    'z': runway.vector(512, sampling_std=0.5),
    'label': runway.number(min=0, max=1000000, default=22, step=1), # generate random labels
    'scale': runway.number(min=-1, max=1, default=0.25, step=0.05),  # magnitude of labels - 0 = no labels
    'truncation': runway.number(min=-1.5, max=1.5, default=1, step=0.1)
}

# Every model needs to have at least one command. Every command allows to send
# inputs and process outputs. To see a complete list of supported inputs and
# outputs data types: https://sdk.runwayml.com/en/latest/data_types.html
@runway.command(name='generate',
                inputs={ 'caption': text() },
                outputs={ 'image': image(width=1024, height=1024) },
                description='Generates.')
def generate(model, args):
    print('[GENERATE] Ran with caption value "{}"'.format(args['caption']))
    # Generate a PIL or Numpy image based on the input caption, and return it
    output_image = model.run_on_input(args['caption'])
    return {
        'image': output_image}
        
def convert(model, inputs):
    z = inputs['z']
    label = int(inputs['label'])
    scale = inputs['scale']
    truncation = inputs['truncation']
    latents = z.reshape((1, 512))
    labels = scale * np.random.RandomState(label).randn(167)
    labels = labels.reshape((1,167)).astype(np.float32)
    images = model.run(latents, labels, truncation_psi=truncation, randomize_noise=False, output_transform=fmt)
    output = np.clip(images[0], 0, 255).astype(np.uint8)
    return {'image': output}
    }

if __name__ == '__main__':
    # run the model server using the default network interface and ports,
    # displayed here for convenience
    runway.run(host='0.0.0.0', port=8000)

## Now that the model is running, open a new terminal and give it a command to
## generate an image. It will respond with a base64 encoded URI
# curl \
#   -H "content-type: application/json" \
#   -d '{ "caption": "red" }' \
#   localhost:8000/generate
