import tensorflow as tf
import tensorflow_probability as tfp

def substitute_random_variable(rv, value_dict, scope=None):
    """Create a new random variable with substituted values and scoped."""
    with tf.name_scope(scope):
        value = value_dict.get(rv)
        return type(rv)(value=value, name=rv.name)

def create_copy(q, dict_swap, scope=None):
    """Create a copy of the distribution q with scoped name."""
    return substitute_random_variable(q, dict_swap, scope=scope)