import json
import os

import tensorflow as tf
import numpy as np
from layers_object import conv_layer, up_sampling, max_pool, initialization
from evaluation_object import cal_loss, normal_loss, train_op, MAX_VOTE, var_calculate
from inputs_object import get_filename_list, dataset_inputs, get_all_test_data
from drawings_object import draw_plots_bayes

class SegNet:
    def __init__(self, conf_file="config.json"):
        with open(conf_file) as f:
            self.config = json.load(f)

        self.num_classes = self.config["NUM_CLASSES"]
        self.use_vgg = self.config["USE_VGG"]

        if self.use_vgg is False:
            self.vgg_param_dict = None
            print("No VGG path in config, so learning from scratch")
        else:
            self.vgg16_npy_path = self.config["VGG_FILE"]
            self.vgg_param_dict = np.load(self.vgg16_npy_path, encoding='latin1').item()
            print("VGG parameter loaded")

        self.train_file = self.config["TRAIN_FILE"]
        self.val_file = self.config["VAL_FILE"]
        self.test_file = self.config["TEST_FILE"]
        self.img_prefix = self.config["IMG_PREFIX"]
        self.label_prefix = self.config["LABEL_PREFIX"]
        self.opt = self.config["OPT"]
        self.input_w = self.config["INPUT_WIDTH"]
        self.input_h = self.config["INPUT_HEIGHT"]
        self.input_c = self.config["INPUT_CHANNELS"]
        self.tb_logs = self.config["TB_LOGS"]
        self.saved_dir = self.config["SAVE_MODEL_DIR"]

        self.images_tr, self.labels_tr = None, None
        self.images_val, self.labels_val = None, None
        self.train_loss, self.train_accuracy = [], []
        self.val_loss, self.val_acc = [], []

        self.model_version = 0  # used for saving the model
        self.saver = None

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session()
            self.batch_size_pl = tf.placeholder(tf.int64, shape=[], name="batch_size")
            #self.is_training_pl = tf.placeholder(tf.bool, name="is_training")
            self.inputs_pl = tf.placeholder(tf.float32, [None, self.input_h, self.input_w, self.input_c])
            self.labels_pl = tf.placeholder(tf.int64, [None, self.input_h, self.input_w, 1])

            # perform local response normalization
            self.norm1 = tf.nn.lrn(self.inputs_pl, depth_radius=5, bias=1.0, alpha=0.0001, beta=0.75, name='norm1')
            # first box of convolution layer,each part we do convolution two times, so we have conv1_1, and conv1_2
            self.conv1_1 = conv_layer(self.norm1, "conv1_1", [3, 3, 3, 64], self.use_vgg, self.vgg_param_dict)
            self.conv1_2 = conv_layer(self.conv1_1, "conv1_2", [3, 3, 64, 64], self.use_vgg, self.vgg_param_dict)
            self.pool1, self.pool1_index, self.shape_1 = max_pool(self.conv1_2, 'pool1')

            # Second box of convolution layer(4)
            self.conv2_1 = conv_layer(self.pool1, "conv2_1", [3, 3, 64, 128], self.use_vgg, self.vgg_param_dict)
            self.conv2_2 = conv_layer(self.conv2_1, "conv2_2", [3, 3, 128, 128], self.use_vgg, self.vgg_param_dict)
            self.pool2, self.pool2_index, self.shape_2 = max_pool(self.conv2_2, 'pool2')

            # Third box of convolution layer(7)
            self.conv3_1 = conv_layer(self.pool2, "conv3_1", [3, 3, 128, 256], self.use_vgg, self.vgg_param_dict)
            self.conv3_2 = conv_layer(self.conv3_1, "conv3_2", [3, 3, 256, 256], self.use_vgg, self.vgg_param_dict)
            self.conv3_3 = conv_layer(self.conv3_2, "conv3_3", [3, 3, 256, 256], self.use_vgg, self.vgg_param_dict)
            self.pool3, self.pool3_index, self.shape_3 = max_pool(self.conv3_3, 'pool3')

            # Fourth box of convolution layer(10)
            self.conv4_1 = conv_layer(self.pool3, "conv4_1", [3, 3, 256, 512], self.use_vgg, self.vgg_param_dict)
            self.conv4_2 = conv_layer(self.conv4_1, "conv4_2", [3, 3, 512, 512], self.use_vgg, self.vgg_param_dict)
            self.conv4_3 = conv_layer(self.conv4_2, "conv4_3", [3, 3, 512, 512], self.use_vgg, self.vgg_param_dict)
            self.pool4, self.pool4_index, self.shape_4 = max_pool(self.conv4_3, 'pool4')

            # Fifth box of convolution layers(13)
            self.conv5_1 = conv_layer(self.pool4, "conv5_1", [3, 3, 512, 512], self.use_vgg, self.vgg_param_dict)
            self.conv5_2 = conv_layer(self.conv5_1, "conv5_2", [3, 3, 512, 512], self.use_vgg, self.vgg_param_dict)
            self.conv5_3 = conv_layer(self.conv5_2, "conv5_3", [3, 3, 512, 512], self.use_vgg, self.vgg_param_dict)
            self.pool5, self.pool5_index, self.shape_5 = max_pool(self.conv5_3, 'pool5')

            # ---------------------So Now the encoder process has been Finished--------------------------------------#
            # ------------------Then Let's start Decoder Process-----------------------------------------------------#

            # First box of deconvolution layers(3)
            self.deconv5_1 = up_sampling(self.pool5, self.pool5_index, self.shape_5, self.batch_size_pl, name="unpool_5")
            self.deconv5_2 = conv_layer(self.deconv5_1, "deconv5_2", [3, 3, 512, 512])
            self.deconv5_3 = conv_layer(self.deconv5_2, "deconv5_3", [3, 3, 512, 512])
            self.deconv5_4 = conv_layer(self.deconv5_3, "deconv5_4", [3, 3, 512, 512])
            
            # Second box of deconvolution layers(6)
            self.deconv4_1 = up_sampling(self.deconv5_4, self.pool4_index, self.shape_4, self.batch_size_pl, name="unpool_4")
            self.deconv4_2 = conv_layer(self.deconv4_1, "deconv4_2", [3, 3, 512, 512])
            self.deconv4_3 = conv_layer(self.deconv4_2, "deconv4_3", [3, 3, 512, 512])
            self.deconv4_4 = conv_layer(self.deconv4_3, "deconv4_4", [3, 3, 512, 256])
            
            # Third box of deconvolution layers(9)
            self.deconv3_1 = up_sampling(self.deconv4_4, self.pool3_index, self.shape_3, self.batch_size_pl, name="unpool_3")
            self.deconv3_2 = conv_layer(self.deconv3_1, "deconv3_2", [3, 3, 256, 256])
            self.deconv3_3 = conv_layer(self.deconv3_2, "deconv3_3", [3, 3, 256, 256])
            self.deconv3_4 = conv_layer(self.deconv3_3, "deconv3_4", [3, 3, 256, 128])
            
            # Fourth box of deconvolution layers(11)
            self.deconv2_1 = up_sampling(self.deconv3_4, self.pool2_index, self.shape_2, self.batch_size_pl, name="unpool_2")
            self.deconv2_2 = conv_layer(self.deconv2_1, "deconv2_2", [3, 3, 128, 128])
            self.deconv2_3 = conv_layer(self.deconv2_2, "deconv2_3", [3, 3, 128, 64])
            
            # Fifth box of deconvolution layers(13)
            self.deconv1_1 = up_sampling(self.deconv2_3, self.pool1_index, self.shape_1, self.batch_size_pl, name="unpool_1")
            self.deconv1_2 = conv_layer(self.deconv1_1, "deconv1_2", [3, 3, 64, 64])
            self.deconv1_3 = conv_layer(self.deconv1_2, "deconv1_3", [3, 3, 64, 64])

            with tf.variable_scope('conv_classifier') as scope:
                self.kernel = tf.get_variable('weights', shape=[1, 1, 64, self.num_classes], initializer=initialization(1, 64))
                self.conv = tf.nn.conv2d(self.deconv1_3, self.kernel, [1, 1, 1, 1], padding='SAME')
                self.biases = tf.get_variable('biases', shape=[self.num_classes], initializer=tf.constant_initializer(0.0))
                self.logits = tf.nn.bias_add(self.conv, self.biases, name=scope.name)

    def train(self, max_steps=30000, batch_size=3):
        image_filename, label_filename = get_filename_list(self.train_file, self.config)
        val_image_filename, val_label_filename = get_filename_list(self.val_file, self.config)

        with self.graph.as_default():
            if self.images_tr is None:
                self.images_tr, self.labels_tr = dataset_inputs(image_filename, label_filename, batch_size, self.config)
                self.images_val, self.labels_val = dataset_inputs(val_image_filename, val_label_filename, batch_size, self.config)

            loss, accuracy, prediction = cal_loss(logits=self.logits, labels=self.labels_pl)
#                                                  , number_class=self.num_classes)
            train, global_step = train_op(total_loss=loss, opt=self.opt)

            summary_op = tf.summary.merge_all()

            
            with self.sess.as_default():
                self.sess.run(tf.local_variables_initializer())
                self.sess.run(tf.global_variables_initializer())

                coord = tf.train.Coordinator()
                threads = tf.train.start_queue_runners(coord=coord)

                train_writer = tf.summary.FileWriter(self.tb_logs, self.sess.graph)
                self.saver = tf.train.Saver()
                
                for step in range(max_steps):
                    image_batch, label_batch = self.sess.run([self.images_tr, self.labels_tr])
                    feed_dict = {self.inputs_pl: image_batch,
                                 self.labels_pl: label_batch,
                                 
                                 self.batch_size_pl: batch_size}

                    _, _loss, _accuracy, summary = self.sess.run([train, loss, accuracy, summary_op],
                                                                 feed_dict=feed_dict)
                    self.train_loss.append(_loss)
                    self.train_accuracy.append(_accuracy)
                    print("Iteration {}: Train Loss{:6.3f}, Train Accu {:6.3f}".format(step, self.train_loss[-1], self.train_accuracy[-1]))

                    if step % 100 == 0:
                        train_writer.add_summary(summary, step)

                    if step % 1000 == 0:
                        print("start validating..")
                        _val_loss = []
                        _val_acc = []
                        for test_step in range(9):
                            image_batch_val, label_batch_val = self.sess.run([self.images_val, self.labels_val])
                            feed_dict_valid = {self.inputs_pl: image_batch_val,
                                               self.labels_pl: label_batch_val,
                                               
                                               self.batch_size_pl: batch_size}
                            # since we still using mini-batch, so in the batch norm we set phase_train to be
                            # true, and because we didin't run the trainop process, so it will not update
                            # the weight!
                            _loss, _acc, _val_pred = self.sess.run([loss, accuracy, self.logits], feed_dict_valid)
                            _val_loss.append(_loss)
                            _val_acc.append(_acc)

                        self.val_loss.append(np.mean(_val_loss))
                        self.val_acc.append(np.mean(_val_acc))

                        print("Val Loss {:6.3f}, Val Acc {:6.3f}".format(self.val_loss[-1], self.val_acc[-1]))
                        
                        self.saver.save(self.sess, os.path.join(self.saved_dir, 'model.ckpt'), global_step=self.model_version)
                        self.model_version = self.model_version + 1

                coord.request_stop()
                coord.join(threads)
           
    def test(self):
        image_filename, label_filename = get_filename_list(self.test_file, self.config)

        with self.graph.as_default():
            with self.sess as sess:
                loss, accuracy, prediction = normal_loss(self.logits, self.labels_pl, self.num_classes)

                images, labels = get_all_test_data(image_filename, label_filename)

                #acc_final = []
                #iu_final = []
                #iu_mean_final = []

                loss_tot = []
                acc_tot = []
                pred_tot = []
                #hist = np.zeros((self.num_classes, self.num_classes))
                step = 0
                for image_batch, label_batch in zip(images, labels):
                    image_batch = np.reshape(image_batch, [1, self.input_h, self.input_w, self.input_c])
                    label_batch = np.reshape(label_batch, [1, self.input_h, self.input_w, 1])
                    feed_dict = {self.inputs_pl: image_batch,
                                 self.labels_pl: label_batch,
                                 
                                 self.batch_size_pl: 1}
                    fetches = [loss, accuracy, self.logits, prediction]
                    loss_per, acc_per, logit, pred = sess.run(fetches=fetches, feed_dict=feed_dict)
                    loss_tot.append(loss_per)
                    acc_tot.append(acc_per)
                    pred_tot.append(pred)
                    print("Image Index {}: TEST Loss{:6.3f}, TEST Accu {:6.3f}".format(step, loss_tot[-1], acc_tot[-1]))
                    step = step + 1
                    #per_class_acc(logit, label_batch, self.num_classes)
                    #hist += get_hist(logit, label_batch)

                #acc_tot = np.diag(hist).sum() / hist.sum()
                #iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

                #print("Total Accuracy for test image: ", acc_tot)
                #print("Total MoI for test images: ", iu)
                #print("mean MoI for test images: ", np.nanmean(iu))

                #acc_final.append(acc_tot)
                #iu_final.append(iu)
                #iu_mean_final.append(np.nanmean(iu))

            return pred_tot
    '''        
    def save(self):
        np.save(os.path.join(self.saved_dir, "Data", "trainloss"), self.train_loss)
        np.save(os.path.join(self.saved_dir, "Data", "trainacc"), self.train_accuracy)
        np.save(os.path.join(self.saved_dir, "Data", "valloss"), self.val_loss)
        np.save(os.path.join(self.saved_dir, "Data", "valacc"), self.val_acc)
        with self.sess.as_default():
            with self.graph.as_default():
                self.saver = tf.train.Saver()
                checkpoint_path = os.path.join(self.saved_dir, 'model.ckpt')
                self.saver.save(self.sess, checkpoint_path, global_step=self.model_version)
                self.model_version += 1
    '''
    
    '''
    def restore(self, res_file):
        from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
        print_tensors_in_checkpoint_file(res_file, None, False)
        with self.graph.as_default():
            for v in tf.global_variables():
                print(v)
            if self.saver is None:
                self.saver = tf.train.Saver(tf.global_variables())
            self.saver.restore(self.sess, res_file)
    '''

if __name__ == "__main__":
    segnetObj = SegNet()
    segnetObj.train()