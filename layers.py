# A Wavenet For Speech Denoising - Dario Rethage - 19.05.2017
# Layers.py

import tensorflow.keras as keras


class AddSingletonDepth(keras.layers.Layer):

    def call(self, x, mask=None):
        x = keras.backend.expand_dims(x, -1)  # add a dimension of the right

        if keras.backend.ndim(x) == 4:
            return keras.backend.permute_dimensions(x, (0, 3, 1, 2))
        else:
            return x

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 3:
            return input_shape[0], 1, input_shape[1], input_shape[2]
        else:
            return input_shape[0], input_shape[1], 1


class Subtract(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(Subtract, self).__init__(**kwargs)

    def call(self, x, mask=None):
        return x[0] - x[1]

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class Slice(keras.layers.Layer):

    def __init__(self, selector, output_shape, **kwargs):
        self.selector = selector
        
        self._selector_to_config()
        #print("Slice selector:", self.selector, "Type:", type(self.selector))
        
        self.desired_output_shape = output_shape
        super(Slice, self).__init__(**kwargs)

    def _selector_to_config(self):
        selector_config = []
        for v in self.selector:
            if isinstance(v, type(Ellipsis)):
                selector_config.append("...")
            elif isinstance(v, int):
                selector_config.append(v)
            elif isinstance(v, slice):
                selector_config.append([v.start, v.stop, v.step])
            else:
                raise Exception(f"Unknown type in selector: {type(v)}. Expects: Ellipsis, int or slice")

        return selector_config

    @classmethod
    def _selector_from_config(cls, config):
        new_selector = []
        for v in config:
            if isinstance(v, str) and v == '...':
                new_selector.append(Ellipsis)
            elif isinstance(v, int):
                new_selector.append(v)
            elif isinstance(v, list):
                new_selector.append( slice(v[0], v[1], v[2]) )
            else:
                raise Exception(f"Unknown type in config: {type(v)}. Expects: string (...), int or list")

        return tuple(new_selector)

    def get_config(self):
        config = super(Slice, self).get_config()
        config.update({'selector': self._selector_to_config(), 'desired_output_shape': self.desired_output_shape})
        return config

    @classmethod
    def from_config(cls, config):
        selector = Slice._selector_from_config(config.pop('selector'))
        return cls(selector, **config)
        

    def call(self, x, mask=None):

        selector = self.selector
        if len(self.selector) == 2 and not type(self.selector[1]) is slice and not type(self.selector[1]) is int:
            x = keras.backend.permute_dimensions(x, [0, 2, 1])
            selector = (self.selector[1], self.selector[0])

        y = x[selector]

        if len(self.selector) == 2 and not type(self.selector[1]) is slice and not type(self.selector[1]) is int:
            y = keras.backend.permute_dimensions(y, [0, 2, 1])

        return y

    def compute_output_shape(self, input_shape):

        output_shape = (None,)
        for i, dim_length in enumerate(self.desired_output_shape):
            if dim_length == Ellipsis:
                output_shape = output_shape + (input_shape[i+1],)
            else:
                output_shape = output_shape + (dim_length,)
        return output_shape
