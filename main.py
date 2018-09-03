import tensorflow as tf
import tflowtools as TFT

LOG_DIR = "probeview"


class GANN():

    def __init__():
        pass


def main():
    sess = TFT.gen_initialized_session(dir=LOG_DIR)

    x = tf.placeholder()  # Scalar constant variables
    y = tf.placholder()
    z = x * y           # As the product of variables, z is automatically declared as a multiplication OPERATOR
    result = sess.run(z)  # Run the operator z and return the value of x * y
    sess.close()  # Explicitly close the session to release memory resources.  No danger if omitted.
    print(result)

    TFT.fireup_tensorboard(LOG_DIR)



main()
