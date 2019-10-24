from keras.utils import plot_model

def PlotKerasModelArchitecture(model, to_file):
    show_shapes = True
    show_layer_names = True

    plot_model(model, to_file=to_file, show_shapes=show_shapes,\
               show_layer_names=show_layer_names)
    return
