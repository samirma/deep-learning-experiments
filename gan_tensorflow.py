import tensorflow as tf
import numpy as np
import helper
from matplotlib import pyplot

def leaky_relu(x, alpha=0.1, name='leaky_relu'):
    return tf.maximum(x, alpha * x, name=name)

def conv2d_transpose(inputs, num_outputs, kernel, strides, is_train):


    x = tf.layers.conv2d_transpose(inputs, num_outputs, kernel, strides, padding='same',
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())
    
    x = tf.layers.batch_normalization(x, training=is_train)
    
    x = leaky_relu(x)
    
    return x


def conv2d(inputs, num_outputs, kernel, strides):

    x = tf.layers.conv2d(inputs, num_outputs, kernel, strides, padding='same',
                          kernel_initializer=tf.contrib.layers.xavier_initializer())
    
    x = tf.layers.batch_normalization(x, training=True)
    
    x = leaky_relu(x)
        
    return x

class NeuralNetwork:
    
    def model_inputs(self, image_width, image_height, image_channels, z_dim):
        inputs_real   = tf.placeholder(tf.float32, (None, image_width, image_height, image_channels), name='input_real')
        inputs_z      = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        return inputs_real, inputs_z, learning_rate

    def discriminator(self, x, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):

            x = conv2d(x, 64, 5, 2)
            
            x = conv2d(x, 128, 5, 2)
            
            x = conv2d(x, 256, 5, 2)
            
            x = conv2d(x, 512, 5, 2)

            # Flatten it
            x = tf.contrib.layers.flatten(x)
            
            x = tf.nn.dropout(x, 0.5)
            
            x = tf.layers.dense(x, 512, kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            x = leaky_relu(x)
            
            x = tf.nn.dropout(x, 0.5)
            
            x = tf.layers.dense(x, 256, kernel_initializer=tf.contrib.layers.xavier_initializer())

            x = leaky_relu(x)
            
            x = tf.layers.dense(x, 128, kernel_initializer=tf.contrib.layers.xavier_initializer())

            x = leaky_relu(x)
            
            x = tf.nn.dropout(x, 0.5)

            logits = tf.layers.dense(x, 1, kernel_initializer=tf.contrib.layers.xavier_initializer())

            out = tf.sigmoid(logits)

            return out, logits



    def generator(self, z, out_channel_dim, is_train=True):
        with tf.variable_scope('generator', reuse=not is_train):
            # First fully connected layer
            h = 72 
            w = 54
            
            kernel = 5
            
            strides = 3
            
            x = tf.layers.dense(z, 12*1024)
            
            x = tf.reshape(x, (-1, 6, 2, 1024))
            
            #x = conv2d_transpose(x, 1024, kernel, strides, is_train)
            
            #x = conv2d_transpose(x, 512, kernel, strides, is_train)
            
            #x = conv2d_transpose(x, 256, kernel, strides, is_train)
            
            x = conv2d_transpose(x, 128, kernel, strides, is_train)
            
            x = conv2d_transpose(x, 64, kernel, strides, is_train)
            
            x = conv2d_transpose(x, 32, kernel, strides, is_train)
            
            x = tf.image.resize_images(x, [h, w])

            # Output Layer
            x = tf.layers.conv2d_transpose(x, out_channel_dim, kernel, strides=1, padding='same')
            out = tf.tanh(x)
            
            return out   

    def model_loss(self, input_real, input_z, out_channel_dim):
        g_model = self.generator(input_z, out_channel_dim)
        d_model_real, d_logits_real = self.discriminator(input_real)
        d_model_fake, d_logits_fake = self.discriminator(g_model, reuse=True)

        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_model_real)))
        d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.zeros_like(d_model_fake)))
        g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake, labels=tf.ones_like(d_model_fake)))

        d_loss = d_loss_real + d_loss_fake

        return d_loss, g_loss


    def model_opt(self, d_loss, g_loss, learning_rate, beta1):

        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if var.name.startswith('discriminator')]
        g_vars = [var for var in t_vars if var.name.startswith('generator')]

        g_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='generator')
        d_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='discriminator')

        # Optimize
        with tf.control_dependencies(d_update_ops):
            d_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(d_loss, var_list=d_vars)

        with tf.control_dependencies(g_update_ops):
            g_train_opt = tf.train.AdamOptimizer(learning_rate, beta1=beta1).minimize(g_loss, var_list=g_vars)

        return d_train_opt, g_train_opt   



    def show_generator_output(self, sess, n_images, input_z, out_channel_dim, image_mode):

        cmap = None if image_mode == 'RGB' else 'gray'
        z_dim = input_z.get_shape().as_list()[-1]
        example_z = np.random.uniform(-1, 1, size=[n_images, z_dim])

        samples = sess.run(
            self.generator(input_z, out_channel_dim, False),
            feed_dict={input_z: example_z})

        images_grid = helper.images_square_grid(samples, image_mode)
        pyplot.imshow(images_grid, cmap=cmap)
        pyplot.show()

        
    def train(self, epoch_count, batch_size, z_dim, learning_rate, beta1, get_batches, data_shape, data_image_mode):
        
        self.batch_size = batch_size
        
        _, image_width, image_height, image_channels = data_shape
        # Set model inputs
        input_real, input_z, learn_rate = self.model_inputs(image_height, image_width, image_channels, z_dim)
        # Set model loss
        d_loss, g_loss = self.model_loss(input_real, input_z, image_channels)
        # Set model optimization
        d_opt, g_opt = self.model_opt(d_loss, g_loss, learning_rate, beta1)
        
        samples, losses = [], []
        steps    = 0    
        print_at = 500
        show_at  = 10000
        
        images_to_show = 4
        
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch_i in range(epoch_count):
                for batch_images in get_batches(batch_size):
                    
                    d_iters = 3
                    g_iters = 2
                    
                    steps += 1
                    
                    batch_images = batch_images * 2
                    
                    # Sample random noise for G
                    sample_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))

                    # Run optimizers
                    for k in range(d_iters):
                        _ = sess.run(d_opt, feed_dict={input_real: batch_images, input_z: sample_z, learn_rate: learning_rate})
                    
                    for k in range(g_iters):
                        _ = sess.run(g_opt, feed_dict={input_z: sample_z, learn_rate: learning_rate}) 
                        
                    if steps == 1:
                        print("Training started")
                    
                    if steps % print_at == 0:
                        # At the end of each epoch, get the losses and print them out
                        train_loss_d = d_loss.eval({input_z: sample_z, input_real: batch_images})
                        train_loss_g = g_loss.eval({input_z: sample_z})
                        
                        print("Epoch {}/{}...".format(epoch_i + 1, epoch_count),
                              "Discriminator Loss: {:.4f}...".format(train_loss_d),
                              "Generator Loss: {:.4f}".format(train_loss_g))
                        
                    if steps % show_at == 0:
                        self.show_generator_output(sess, images_to_show, input_z, image_channels, data_image_mode)
                self.show_generator_output(sess, images_to_show, input_z, image_channels, data_image_mode)
