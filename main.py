import argparse
import utils

# parse arguments 
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input_dir', help='Directory for input rgb and optical flow images')
parser.add_argument('-o', '--output_dir', help='Directory for exported outputs')
parser.add_argument('-p', '--model_path', type=str, help='Path to the model to use')

parser.add_argument('-f', '--flow_size', default=64, type=int, help='Size of input optical flow (default=64)')
parser.add_argument('-r', '--rgb_size', default=256, type=int, help='Size of rgb output (default=256)')
parser.add_argument('-m', '--m', default=2, type=int, help='Number of optical flow pairs per input (default=2)')
parser.add_argument('-t', '--top', default=.05, type=float, help='Top % heatmaps remain (default=0.05)')

options = parser.parse_args()

# load model 
cls = utils.load_model(options.model_path)

# collect layer names to later compute gradients between the last conv layer and the output
last_conv_layer_name = cls.get_layer('svdd').layers[-2].name
classifier_layer_names = []
classifier_layer_names.append('svdd/' + cls.get_layer('svdd').layers[-1].name)

flag = False
for l in cls.layers:
    
    if flag:
        classifier_layer_names.append(l.name)
        
    if l.name == 'svdd':
        flag = True

# load data
test_rgb_inst, test_flow_inst = utils.load_example_images(options.input_dir, 
                                n_chosen=options.m, img_size=options.flow_size,
                                rgb_img_size=options.rgb_size)

# run grad-cam 
utils.run_grad_cam(cls, export_dir=options.output_dir, test_flow_inst=test_flow_inst, 
                    test_rgb_inst=test_rgb_inst, last_conv_layer_name=last_conv_layer_name,
                    classifier_layer_names=classifier_layer_names, top_p=options.top, export_gif=True)