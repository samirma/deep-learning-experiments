import tensorflow as tf
import numpy as np
import helper
from matplotlib import pyplot

def leaky_relu(x, alpha=0.1, name='leaky_relu'):
    return tf.maximum(x, alpha * x, name=name)

class NeuralNetwork:
    

    def model_inputs(self, image_width, image_height, image_channels, z_dim):
        inputs_real   = tf.placeholder(tf.float32, (None, image_width, image_height, image_channels), name='input_real')
        inputs_z      = tf.placeholder(tf.float32, (None, z_dim), name='input_z')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')
        
        print (inputs_real.shape)

        return inputs_real, inputs_z, learning_rate

    def discriminator(self, images, reuse=False):
        with tf.variable_scope('discriminator', reuse=reuse):
            # Input layer is 28x28x3
            x1 = tf.layers.conv2d(images, 64, 5, strides=2, padding='same',
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
            relu1 = leaky_relu(x1)

            x2 = tf.layers.conv2d(relu1, 128, 5, strides=2, padding='same',
                                  kernel_initializer=tf.contrib.layers.xavier_initializer())
            bn2 = tf.layers.batch_normalization(x2, training=True)
            relu2 = leaky_relu(bn2)

            x3 = tf.layers.conv2d(relu2, 256, 5, strides=2, padding='same',
                                 kernel_initializer=tf.contrib.layers.xavier_initializer())
            bn3 = tf.layers.batch_normalization(x3, training=True)
            relu3 = leaky_relu(bn3)

            # Flatten it
            flat = tf.reshape(relu3, (-1, 4*4*256))

            # Apply some dropout
            layer_dropout = tf.nn.dropout(flat, 0.7)

            logits = tf.layers.dense(layer_dropout, 1,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer())


            out = tf.sigmoid(logits)

            return out, logits



    def generator(self, z, out_channel_dim, is_train=True):
        with tf.variable_scope('generator', reuse=not is_train):
            # First fully connected layer
            x1 = tf.layers.dense(z, 7*7*512)
            # Reshape it to start the convolutional stack
            x1 = tf.reshape(x1, (-1, 7, 7, 512))
            x1 = tf.layers.batch_normalization(x1, training=is_train)
            x1 = leaky_relu(x1)
            # 7x7x512

            # Apply some dropout
            layer_dropout = tf.nn.dropout(x1, 0.5)        

            x2 = tf.layers.conv2d_transpose(layer_dropout, 256, 5, strides=2, padding='same',
                                           kernel_initializer=tf.contrib.layers.xavier_initializer())
            x2 = tf.layers.batch_normalization(x2, training=is_train)
            x2 = leaky_relu(x2)
            # 14x14x256

            # Apply some dropout
            layer_dropout1 = tf.nn.dropout(x2, 0.5)        


            x3 = tf.layers.conv2d_transpose(layer_dropout1, 128, 5, strides=2, padding='same',
                                           kernel_initializer=tf.contrib.layers.xavier_initializer())
            x3 = tf.layers.batch_normalization(x3, training=is_train)
            x3 = leaky_relu(x3)
            # 28x28x128

            # Apply some dropout
            layer_dropout2 = tf.nn.dropout(x3, 0.5)         

            # Output Layer
            logits = tf.layers.conv2d_transpose(layer_dropout2, out_channel_dim, 5, strides=1, padding='same',
                                               kernel_initializer=tf.contrib.layers.xavier_initializer())
            # 28x28x3

            out = tf.tanh(logits)
            
            return out  

    def model_loss(self, input_real, input_z, out_channel_dim):
        g_model = self.generator(input_z, out_channel_dim)
        d_model_real, d_logits_real = self.discriminator(input_real)
        d_model_fake, d_logits_fake = self.discriminator(g_model, reuse=True)

        d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_real, labels=tf.ones_like(d_model_real) * (1 - 0.01)))
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
        input_real, input_z, learn_rate = self.model_inputs(image_width, image_height, image_channels, z_dim)
        # Set model loss
        d_loss, g_loss = self.model_loss(input_real, input_z, image_channels)
        # Set model optimization
        d_opt, g_opt = self.model_opt(d_loss, g_loss, learning_rate, beta1)
        
        samples, losses = [], []
        steps    = 0    
        print_at = 50
        show_at  = 100
        
        images_to_show = 10
        
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch_i in range(epoch_count):
                for batch_images in get_batches(batch_size):
                    # TODO: Train Model
                    steps += 1
                    
                    batch_images = batch_images * 2
                    
                    # Sample random noise for G
                    sample_z = np.random.uniform(-1, 1, size=(batch_size, z_dim))

                    # Run optimizers
                    _ = sess.run(d_opt, feed_dict={input_real: batch_images, input_z: sample_z, learn_rate: learning_rate})
                    _ = sess.run(g_opt, feed_dict={input_z: sample_z, learn_rate: learning_rate}) 
                
                                    
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
