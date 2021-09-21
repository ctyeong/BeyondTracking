from glob import glob
import tensorflow as tf 
import tensorflow.keras as keras
from PIL import Image
import numpy as np 
import os
import matplotlib.cm as cm


def load_example_images(dir_path, n_chosen=2, img_size=64, rgb_img_size=256):
    test_flow_inst = []
    test_rgb_inst = []
    rgb_img_size = (rgb_img_size, rgb_img_size)
    img_size = (img_size, img_size)

    rgb_paths = sorted(glob(os.path.join(dir_path, 'rgb*.jpg')))
    flow_x_paths = sorted(glob(os.path.join(dir_path, 'flow_x*.jpg')))
    flow_y_paths = sorted(glob(os.path.join(dir_path, 'flow_y*.jpg')))

    assert len(flow_x_paths) == len(flow_y_paths), 'flow_x and flow_y have different numbers of paths'
    assert len(rgb_paths) - 1 == len(flow_y_paths)//n_chosen, 'rgb_paths must have one more path than flow_x or flow_y'

    for path in rgb_paths:
        rgb = np.asarray(Image.open(path).resize(rgb_img_size), dtype=np.uint8)
        test_rgb_inst.append(rgb)
            
    # load opt flows and calculate weights
    instant_flows = []
    for i, (x_path, y_path) in enumerate(zip(flow_x_paths, flow_y_paths)):
        flow_x = (np.asarray(Image.open(x_path).convert('L').resize(img_size),
                            dtype=np.float32) - 127)/128. 
        flow_y = (np.asarray(Image.open(y_path).convert('L').resize(img_size),
                            dtype=np.float32) - 127)/128.

        instant_flows.append(flow_x)
        instant_flows.append(flow_y)
        
        if (i + 1) % n_chosen != 0:
            continue 
        instant_flows = np.stack(instant_flows, axis=-1)
        test_flow_inst.append(instant_flows)
        instant_flows = []

    test_flow_inst = np.asarray(test_flow_inst)
    test_rgb_inst = np.asarray(test_rgb_inst)
    print('{} flows loaded'.format(len(test_flow_inst)))
    print()

    return test_rgb_inst, test_flow_inst


def load_model(model_path):

    def smooth_accuracy(y_true, y_pred):
        y_true = keras.backend.round(y_true)
        y_pred = keras.backend.round(y_pred)
        correct = keras.backend.cast(keras.backend.equal(y_true, y_pred), dtype='float32')
        return keras.backend.mean(correct)

    cls = keras.models.load_model(model_path, 
            custom_objects={'smooth_accuracy': smooth_accuracy, 'keras': keras})
    cls = keras.Model(inputs=cls.input, outputs=cls.layers[-2].output)

    last_conv_layer_name = cls.get_layer('svdd').layers[-2].name
    classifier_layer_names = []
    classifier_layer_names.append('svdd/' + cls.get_layer('svdd').layers[-1].name)

    flag = False
    for l in cls.layers:
        
        if flag:
            classifier_layer_names.append(l.name)
            
        if l.name == 'svdd':
            flag = True

    print(cls.summary())
    print()
    return cls


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names, 
                         top_p=.05, svdd=False):
    def euclidean_distance_square_loss(c_vec, v_vec):
        return keras.backend.sum(keras.backend.square(v_vec - c_vec), axis=-1)
    # First, we create a model that maps the input image to the activations
    # of the last conv layer
    
    if svdd:
        last_conv_layer = model.get_layer(last_conv_layer_name)
        last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)
        
    else:
        last_conv_layer = model.get_layer('svdd').get_layer(last_conv_layer_name)
        last_conv_layer_model = keras.Model(model.get_layer('svdd').inputs, last_conv_layer.output)

    # Second, we create a model that maps the activations of the last conv
    # layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        if layer_name[:5] == 'svdd/':
            x = model.get_layer('svdd').get_layer(layer_name[5:])(x)
            
        else:
            x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        # Compute class predictions
        preds = classifier_model(last_conv_layer_output)
        
        ################ TChoi ################
        # use distance as the top_class_channel 
        if svdd:
            top_class_channel = euclidean_distance_square_loss(target_feat, preds)
            
        else:
            top_class_channel = preds 

    # This is the gradient of the top predicted class with regard to
    # the output feature map of the last conv layer
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    
    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    last_conv_layer_output = last_conv_layer_output.numpy()#[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        for j in range(pooled_grads.shape[0]):
            last_conv_layer_output[j, :, :, i] *= pooled_grads[j, i]

    # The channel-wise mean of the resulting feature map
    # is our heatmap of class activation
    heatmap = np.mean(last_conv_layer_output, axis=-1)

   # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap - np.min(heatmap, 0) # subtract regional mins 
    
    for i in range(len(heatmap)):
        h = heatmap[i]
        if np.max(h) > 0:
            heatmap[i] /= np.max(h)
            
    #         h += np.min(h)# min = 0
            threshold = np.quantile(h, (1-top_p))
            heatmap[i][h < threshold] = 0. # filter out lows
    
    return heatmap


def run_grad_cam(cls, export_dir='grad_cam_outputs', output_name='out', 
                test_flow_inst=[], test_rgb_inst=[], last_conv_layer_name='',
                classifier_layer_names=[], top_p=.05, export_gif=True):
    import copy
    
    assert len(test_flow_inst) == len(test_rgb_inst)-1
    assert top_p > 0 and top_p < 1

    export_frames = True
    heatmap_intensity = .4

    if not os.path.isdir(export_dir):
        os.makedirs(export_dir)

    # Generate class activation heatmap
    heatmaps = make_gradcam_heatmap(test_flow_inst, cls, last_conv_layer_name, classifier_layer_names,
                                   top_p=top_p, svdd=False)

    # Get each superimposed image
    imposed_imgs = []
    for i, (heatmap, rgb) in enumerate(zip(heatmaps, test_rgb_inst)):
        heatmap = copy.deepcopy(heatmap)

        if not np.max(heatmap) == 0:
            heatmap /= np.max(heatmap)

            # We rescale heatmap to a range 0-255
            heatmap = np.uint8(255 * heatmap)

            # We use jet colormap to colorize heatmap
            jet = cm.get_cmap("jet")

            # We use RGB values of the colormap
            jet_colors = jet(np.arange(256))[:, :3]
            jet_heatmap = jet_colors[heatmap]

            # We create an image with RGB colorized heatmap
            jet_heatmap /= np.max(jet_heatmap)
            jet_heatmap = Image.fromarray(np.uint8(jet_heatmap*255))
            jet_heatmap = jet_heatmap.resize((rgb.shape[1], rgb.shape[0]))
            jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

            # Superimpose the heatmap on original image
            superimposed_img = jet_heatmap * heatmap_intensity + rgb #* (1 - heatmap_intensity)
            superimposed_img /= np.max(superimposed_img)
            superimposed_img = Image.fromarray(np.uint8(superimposed_img*255))

        else:
            superimposed_img = Image.fromarray(np.uint8(rgb*255))

        imposed_imgs.append(superimposed_img)

    # export .jpg and .gif
    for j, imposed_img in enumerate(imposed_imgs):
        imposed_img.save(os.path.join(export_dir, '{}-{:02d}.jpg'.format(output_name, j)), 'JPEG')

    if export_gif:
        imposed_imgs[0].save(os.path.join(export_dir, '{}.gif'.format(output_name)), save_all=True, 
                             duration=750, append_images=imposed_imgs[1:])