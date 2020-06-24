'''This script require TF2.0 and a custom branch (ml-module) of open3D to run'''
'''It will create a keras model and generate a test set for continuous convolution to be tested by Barracuda'''
'''Model won't be exported as there is not way currently to import custom layer to barracuda'''

import os 
import argparse
import tensorflow as tf
import open3d.ml.tf as o3dml
import numpy as np
import json

parser = argparse.ArgumentParser(description='Serialize test set for continuous conv test from TF/Open3D-ml')

parser.add_argument('-i', '--inpParticleCount', type=int, required=False, default=100)
parser.add_argument('-o', '--outParticleCount', type=int,  required=False, default=50)

args = parser.parse_args()

target_file = 'continuous_conv_1Layer_testset.json'
shouldPrintModelOutput = True;


class MyTestNetwork(tf.keras.Model):

    def __init__(
            self,
            kernel_size=[4, 4, 4],
            radius_scale=1.5,
            coordinate_mapping='ball_to_cube_volume_preserving',
            interpolation='linear',
            use_window=True,
            particle_radius=0.025,
            timestep=1 / 50,
    ):
        super().__init__(name=type(self).__name__)
        self.layer_channels = [32, 64, 64, 3]
        self.kernel_size = kernel_size
        self.radius_scale = radius_scale
        self.coordinate_mapping = coordinate_mapping
        self.interpolation = interpolation
        self.use_window = use_window
        self.particle_radius = particle_radius
        self.filter_extent = np.float32(self.radius_scale * 6 *
                                        self.particle_radius)
        self._all_convs = []

        def window_poly6(r_sqr):
            return tf.clip_by_value((1 - r_sqr)**3, 0, 1)

        def Conv(name, activation=None, **kwargs):
            conv_fn = o3dml.layers.ContinuousConv


            window_fn = None
            if self.use_window == True:
                window_fn = window_poly6

            conv = conv_fn(name=name,
                           kernel_size=self.kernel_size,
                           activation=activation,
                           align_corners=True,
                           interpolation=self.interpolation,
                           coordinate_mapping=self.coordinate_mapping,
                           normalize=False,
                           window_function=window_fn,
                           radius_search_ignore_query_points=True,
                           **kwargs)

            self._all_convs.append((name, conv))
            return conv

        self.conv0_fluid = Conv(name="conv0_fluid",
                                filters=self.layer_channels[0],
                                activation=None)
                                
    def run(self, vel, pos0, pos1):
        # compute the extent of the filters (the diameter)
        filter_extent = tf.constant(self.filter_extent)
        self.ans_conv0_fluid = self.conv0_fluid(vel, pos0, pos1, filter_extent)

        print(filter_extent)
        return self.ans_conv0_fluid.numpy()

#create inputs
numFluidParticles = args.inpParticleCount
numStaticParticles = args.outParticleCount
np.random.seed(0);
vel  = np.random.rand(numFluidParticles, 3).astype(np.float32)
pos0 = np.random.rand(numFluidParticles, 3).astype(np.float32)
pos1 = np.random.rand(numStaticParticles, 3).astype(np.float32)
print('vel shape:', vel.shape);
print('pos0 shape:', pos0.shape);
print('pos1 shape:', pos1.shape);

#create weights
weights  = np.random.rand(4,4,4,3,32).astype(np.float32)
bias  = np.random.rand(32,).astype(np.float32)

print('weights shape:', weights.shape);
print('bias shape:', bias.shape);

#create model
model = MyTestNetwork()
print('Num layer: ', len(model.layers))
print('layer0: ', model.layers[0])
#run the networks a first time to initialize the shapes then load weights
resultBeforeWeight = model.run(vel, pos0, pos1)
model.layers[0].set_weights([weights,bias])

#run the networks with the weights
result = model.run(vel, pos0, pos1)

print (result)
print (result.shape)
print(np.sum(result))


input_dict0 = {}
input_dict0['shape'] = {}
input_dict0['name'] = 'inputFeature'
input_dict0['shape']['batch'] = vel.shape[0]
input_dict0['shape']['height'] = 1
input_dict0['shape']['width'] = 1
input_dict0['shape']['channels'] = vel.shape[1]
input_dict0['data'] = vel.flatten().tolist()

input_dict1 = {}
input_dict1['shape'] = {}
input_dict1['name'] = 'inputPos'
input_dict1['shape']['batch'] = pos0.shape[0]
input_dict1['shape']['height'] = 1
input_dict1['shape']['width'] = 1
input_dict1['shape']['channels'] = pos0.shape[1]
input_dict1['data'] = pos0.flatten().tolist()

input_dict2 = {}
input_dict2['shape'] = {}
input_dict2['name'] = 'outputPos'
input_dict2['shape']['batch'] = pos1.shape[0]
input_dict2['shape']['height'] = 1
input_dict2['shape']['width'] = 1
input_dict2['shape']['channels'] = pos1.shape[1]
input_dict2['data'] = pos1.flatten().tolist()

input_dict3 = {}
input_dict3['shape'] = {}
input_dict3['name'] = 'weights'
input_dict3['shape']['batch'] = 4*4*4
input_dict3['shape']['height'] = 1
input_dict3['shape']['width'] = weights.shape[3]
input_dict3['shape']['channels'] = weights.shape[4]
input_dict3['data'] = weights.flatten().tolist()

input_dict4 = {}
input_dict4['shape'] = {}
input_dict4['name'] = 'bias'
input_dict4['shape']['batch'] = 1
input_dict4['shape']['height'] = 1
input_dict4['shape']['width'] = 1
input_dict4['shape']['channels'] = bias.shape[0]
input_dict4['data'] = bias.flatten().tolist()

output_dict = {}
output_dict['name'] = 'output'
output_dict['shape'] = {}
output_dict['shape']['batch'] = result.shape[0]
output_dict['shape']['height'] = 1
output_dict['shape']['width'] = 1
output_dict['shape']['channels'] = result.shape[1]
output_dict['data'] = result.flatten().tolist()

json_data = {}
json_data['inputs'] = []
json_data['outputs'] = []

json_data['inputs'].append(input_dict0)
json_data['inputs'].append(input_dict1)
json_data['inputs'].append(input_dict2)
json_data['inputs'].append(input_dict3)
json_data['inputs'].append(input_dict4)
json_data['outputs'].append(output_dict)

file = open(target_file, 'w+')
json.dump(json_data, file)
file.close()