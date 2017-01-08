# used code from https://github.com/machrisaa/tensorflow-vgg/

def VGG16(x):

    def avg_pool(bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
    
    def conv_layer(bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)
            
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            
            return relu
        
        def fc_layer(bottom, in_size, out_size, name):
            with tf.variable_scope(name):
                weights, biases = self.get_fc_var(in_size, out_size, name)
                
                x = tf.reshape(bottom, [-1, in_size])
                fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
                
                return fc
            
            def get_conv_var(filter_size, in_channels, out_channels, name):
                initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
                filters = self.get_var(initial_value, name, 0, name + "_filters")
                
                initial_value = tf.truncated_normal([out_channels], .0, .001)
                biases = self.get_var(initial_value, name, 1, name + "_biases")
                
                return filters, biases
            
            #Layer 1: Convolutional. Input = 32x32x3. Output = 64x64x3.
            #resize images to
            #resize to vgg16 224x224x3
            VGG_MEAN = [103.939, 116.779, 123.68]
            x1 = tf.image.resize_images(x, [224,224])
            
            print(x1.get_shape().as_list()[1:])
            print("build model started")
            rgb_scaled = x1 * 255.0
            #Cinvert RGB to BGR
            red, green, blue = tf.split(3, 3, rgb_scaled)
            assert red.get_shape().as_list()[1:] == [224, 224, 1]
            assert green.get_shape().as_list()[1:] == [224, 224, 1]
            assert blue.get_shape().as_list()[1:] == [224, 224, 1]

            bgr = tf.concat(3, [
                        blue - VGG_MEAN[0],
                        green - VGG_MEAN[1],
                        red - VGG_MEAN[2],
                    ])
                assert bgr.get_shape().as_list()[1:] == [224, 224, 3]



                conv1_1 = conv_layer(bgr, 3, 64, "conv1_1")
                conv1_2 = conv_layer(conv1_1, 64, 64, "conv1_2")
                pool1 = max_pool(conv1_2, 'pool1')

                conv2_1 = conv_layer(pool1, 64, 128, "conv2_1")
                conv2_2 = conv_layer(conv2_1, 128, 128, "conv2_2")
                pool2 = max_pool(conv2_2, 'pool2')
                
                conv3_1 = conv_layer(pool2, 128, 256, "conv3_1")
                conv3_2 = conv_layer(conv3_1, 256, 256, "conv3_2")
                conv3_3 = conv_layer(conv3_2, 256, 256, "conv3_3")
                conv3_4 = conv_layer(conv3_3, 256, 256, "conv3_4")
                pool3 = max_pool(conv3_4, 'pool3')
                
                conv4_1 = conv_layer(pool3, 256, 512, "conv4_1")
                conv4_2 = conv_layer(conv4_1, 512, 512, "conv4_2")
                conv4_3 = conv_layer(conv4_2, 512, 512, "conv4_3")
                conv4_4 = conv_layer(conv4_3, 512, 512, "conv4_4")
                pool4 = max_pool(conv4_4, 'pool4')
                
                conv5_1 = conv_layer(pool4, 512, 512, "conv5_1")
                
                
                conv5_2 = conv_layer(conv5_1, 512, 512, "conv5_2")
                conv5_3 = conv_layer(conv5_2, 512, 512, "conv5_3")
                conv5_4 = conv_layer(conv5_3, 512, 512, "conv5_4")
                pool5 = max_pool(conv5_4, 'pool5')

                fc6 = fc_layer(pool5, 25088, 4096, "fc6")  # 25088 = ((224 / (2 ** 5)) ** 2) * 512
                relu6 = tf.nn.relu(fc6)
                if train_mode is not None:
                    relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(relu6, 0.5), lambda: relu6)
                elif trainable:
                    relu6 = tf.nn.dropout(relu6, 0.5)
                    
                fc7 = fc_layer(relu6, 4096, 4096, "fc7")
                relu7 = tf.nn.relu(fc7)
                if train_mode is not None:
                    relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(relu7, 0.5), lambda: relu7)
                elif trainable:
                    relu7 = tf.nn.dropout(relu7, 0.5)
                        
                fc8 = fc_layer(relu7, 4096, 1000, "fc8")
        
                prob = tf.nn.softmax(fc8, name="prob")
                return logits

