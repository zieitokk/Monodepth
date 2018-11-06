
from keras.layers import Input, BatchNormalization, UpSampling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, AveragePooling2D, Conv2DTranspose
from keras.layers.core import Activation
from keras import backend as K
from utils import ReflectionPadding2D, GradientPadding
from bilinear_sampler import *
from config import *

class model(object):
    def __init__(self,  mode, left, right, reuse_variables=None, model_index=0):
        """

        :param mode: Either 'test' mode or 'train' mode
        :param left: Batch of left angle of view of images and its dim is (batch size, height, width, channels)
        :param right: Batch of right angle of view of images and its dim is (batch size, height, width, channels)
        :param reuse_variables: Flag for tensorflow reuse_variables function
        :param model_index: Collect model summary, it can generate different models when you train it multiple times
        """
        # super(model, self).__init__()
        self.mode = mode
        # self.left = Input(shape=(128, 416, 3))
        # self.right = Input(shape=(128, 416, 3))
        self.left = left
        self.right = right
        self.model_collection = ['model_' + str(model_index)]

        self.reuse_variables = reuse_variables

        self.build_model()
        self.build_outputs()

        if self.mode == 'test':
            return

        self.build_losses()
        self.build_summaries()


    def gradient_x(self, img):
        """

        :param img: image that need to be processed
        :return: gradient_x is the value that right pixel minus left pixel along the width dim.
        """
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx

    def gradient_y(self, img):
        """

        :param img: image that need to be processed
        :return: gradient_x is the value that upper pixel minus lower pixel along the height dim.
        """
        gy = img[:, :-1, :, :] - img[:, 1:, :, :]
        return gy

    def scale_image(self, img, num_scales):
        """

        :param img:
        :param num_scales: number that how many scales that need to processed.
                            For example, num_scales = 4, so we will have 4 different-sized images with corresponding
                            image dims are h x w, h/2 x w/2, h/4 x w/4, h/8 x w/8
        :return:
        """
        scaled_imgs = [img]
        h = scaled_imgs[0].shape[1]
        w = scaled_imgs[0].shape[2]
        for i in range(num_scales - 1):
            ratio = 2 ** (i+1)
            # nh = h // ratio
            # nw = w // ratio
            scaled_imgs.append(AveragePooling2D(pool_size=(ratio, ratio))(img))
        print('hello')
        return scaled_imgs

    def generate_image_left(self, img, disp):
        """

        :param img: right view images for projection
        :param disp: disparity that projects right view images into left view images
        :return:
        """
        return bilinear_sampler_1d_h(img, -disp)

    def generate_image_right(self, img, disp):
        """

        :param img: left view images for projection
        :param disp: disparity that projects left view images into right view images
        :return:
        """
        return bilinear_sampler_1d_h(img, disp)

    def get_disparity_smoothness(self, disp, pyramid):
        """

        :param disp: disparity of pyramid images
        :param pyramid: pyramid images with different dimensions. Shape of pyramid[0] is quadratic larger than shape of pyramid[1]
        :return:

                According to the paper, L_s = abs(gradient_x(disparity/mean(disparity)) * exp (-abs(gradient_x(image) /
                                                + abs(gradient_y(disparity/mean(disparity)) * exp (-abs(gradient_y(image)
        """
        ## 1. mean
        ## 2. element wise d/d_mean
        ## 3. padding
        ## 4. gradients
        ## 5. abs of gradients

        # disp_star = disp / mean_disp
        disp_star = [g / (K.mean(g, axis=0,  keepdims=False) + 1e-8) for g in disp]
        # padding for gradients
        tempA = [GradientPadding(padding=((0, 0), (1, 0)))(p) for p in disp_star]
        # tempA = [K.spatial_2d_padding(p, padding=((0, 0), (0, 1)), data_format='channels_last') for p in disp_star]
        disp_gradients_x = [K.abs(self.gradient_x(d)) for d in tempA]

        tempB = [GradientPadding(padding=((1, 0), (0, 0)))(p) for p in disp_star]
        # tempB = [K.spatial_2d_padding(p, padding=((0, 1), (0, 0)), data_format='channels_last') for p in disp_star]
        disp_gradients_y = [K.abs(self.gradient_y(d)) for d in tempB]

        image_tempA = [GradientPadding(padding=((0, 0), (1, 0)))(p) for p in pyramid]
        # image_tempA = [K.spatial_2d_padding(p, padding=((0, 0), (0, 1)), data_format='channels_last') for p in pyramid]
        weight_x = [K.exp(-K.abs(self.gradient_x(img))) for img in image_tempA]

        image_tempB = [GradientPadding(padding=((1, 0), (0, 0)))(p) for p in pyramid]
        # image_tempB = [K.spatial_2d_padding(p, padding=((0, 1), (0, 0)), data_format='channels_last') for p in pyramid]
        weight_y = [K.exp(-K.abs(self.gradient_y(img))) for img in image_tempB]

        smoothness_x = [disp_gradients_x[i] * weight_x[i] for i in range(4)]
        smoothness_y = [disp_gradients_y[i] * weight_y[i] for i in range(4)]

        return smoothness_x + smoothness_y

    def cal_std(self, x):
        """

        :param x:
        :return:
                Based on the standard deviation formula:
                    sigma = E(x**2) - E(x)**2
                we return the sigma of x and its mean
        """
        mu_x = AveragePooling2D((3, 3), 1, 'valid')(x)
        pow_mu_x = mu_x**2
        pow_x = x**2
        pooled_pow_x = AveragePooling2D((3, 3), strides=1, padding='valid')(pow_x)
        sigma_x = pooled_pow_x - pow_mu_x
        return sigma_x, mu_x

    def SSIM(self, x, y):
        """
        SSIM is one of the method to calculate the structural similarity between images x and y. The larger SSIM value
        the better. Its maximum value is 1.

                    SSIM formula:
                            SSIM(x, y) = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2) / ((mu_x**2 + mu_y**2 + C1) * (sigma_x**2 + sigma_y**2 + C2))
        :param x:
        :param y:
        :return:
        """
        with tf.variable_scope('SSIM'):
            C1 = 0.01 ** 2
            C2 = 0.03 ** 2
            sigma_x, mu_x = self.cal_std(x)
            sigma_y, mu_y = self.cal_std(y)
            xy = x*y
            mu_xy = mu_x*mu_y
            sigma_xy = AveragePooling2D((3, 3), 1, 'valid')(xy)
            sigma_xy = sigma_xy - mu_xy
            SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
            SSIM_d = (mu_x**2 + mu_y **2 + C1) * (sigma_x + sigma_y + C2)
            SSIM = SSIM_n/SSIM_d
            Last_SSIM = (1-SSIM)/2
        return K.clip(Last_SSIM, 0, 1)


    def res_block(self, input, output_chn, first_flag=False):
        """

        :param input: input layer with shape (batch size, height, width, channels)
        :param output_chn: output channels
        :param first_flag: flag that whether it is the first layer of entire network
        :return: y = F(x) + x
        """
        if first_flag is False:
            input = MaxPooling2D(pool_size=(2, 2), strides=2, padding='same')(input)
        else:
            input = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(input)

        x = Conv2D(output_chn, kernel_size=3, strides=(1, 1), padding='same')(input)
        x = BatchNormalization(momentum=0.99, epsilon=1e-3)(x)
        x = Activation('relu')(x)
        x = Conv2D(output_chn, kernel_size=3, strides=(1, 1), padding='same')(x)
        x = BatchNormalization(momentum=0.99, epsilon=1e-3)(x)
        residual = Activation('relu')(x)
        modified_input = self.add_input(input, output_chn)
        output = modified_input + residual

        x = Conv2D(output_chn, kernel_size=3, strides=(1, 1), padding='same')(output)
        x = BatchNormalization(momentum=0.99, epsilon=1e-3)(x)
        x = Activation('relu')(x)
        x = Conv2D(output_chn, kernel_size=3, strides=(1, 1), padding='same')(x)
        x = BatchNormalization(momentum=0.99, epsilon=1e-3)(x)
        residual = Activation('relu')(x)
        modified_input = self.add_input(input, output_chn)
        output = modified_input + residual
        return output

    def add_input(self, input, output_chn):
        """

        :param input: input to be the shortcut layer
        :param output_chn: output channels
        :return: identity mapping
        """
        output = Conv2D(output_chn, kernel_size=3, strides=(1, 1), padding='same')(input)
        return output

    def up_block(self, input, short_cut, output_chn, output_disparity_flag=False, last_layer_flag=False):
        # x = ReflectionPadding2D(padding=(1, 1))(input)
        # x = Conv2D(output_chn, kernel_size=3, strides=(1, 1), padding='valid', activation='elu')(x)
        # x = UpSampling2D(size=(2, 2), data_format='channels_last')(x)

        """

        :param input: input layer x
        :param short_cut: short_cut x
        :param output_chn: output channels
        :param output_disparity_flag: flag to determine whether this layer should output disparity
        :param last_layer_flag: flag to determine whether this is the last layer of entire network, with which does not need to combine short_cut layer
        :return:
        """
        x = Conv2DTranspose(output_chn, kernel_size=3, strides=(1, 1), padding='same', activation='elu', data_format='channels_last')(input)
        ## The reason that we use x.set_shape function below is because there is a bug in Keras.layer. It cannot output the
        ## actural shape after conv2dtranspose, and hence we need to manually set its shape.
        x.set_shape(x._keras_shape)
        x = UpSampling2D(size=(2, 2), data_format='channels_last')(x)
        if last_layer_flag is False:
            x = x + short_cut
        x = ReflectionPadding2D(padding=(1, 1))(x)
        output = Conv2D(output_chn, kernel_size=3, strides=(1, 1), padding='valid', activation='elu')(x)
        if output_disparity_flag is True:
            cur_disp = ReflectionPadding2D(padding=(1, 1))(output)
            cur_disp = 0.3 * Conv2D(2, kernel_size=3, strides=(1, 1), padding='valid', activation='sigmoid')(cur_disp)
        else:
            cur_disp = None
        return output, cur_disp

    def build_depth_model(self):
        """
        encoder
        """
        with tf.variable_scope('encoder'):
        # econv1
            x = Conv2D(32, kernel_size=7, strides=(2, 2), padding='same')(self.model_input)
            x = BatchNormalization(momentum=0.99, epsilon=1e-3)(x)
            econv1 = Activation('relu')(x)
            # econv2
            econv2 = self.res_block(econv1, 64, first_flag=True)
            # econv3
            econv3 = self.res_block(econv2, 128)
            # econv4
            econv4 = self.res_block(econv3, 256)
            # econv5
            econv5 = self.res_block(econv4, 512)

        """
        skip (shortcut)
        """
        with tf.variable_scope('skips'):
            skip1 = econv1
            skip2 = econv2
            skip3 = econv3
            skip4 = econv4
        """
        decoder
        """
        with tf.variable_scope('decoder'):
            # upconv5
            upconv5, _ = self.up_block(econv5, skip4, 256)
            # upconv4
            upconv4, self.disp4 = self.up_block(upconv5, skip3, 128, output_disparity_flag=True)
            # upconv3
            upconv3, self.disp3 = self.up_block(upconv4, skip2, 64, output_disparity_flag=True)
            # upconv2
            upconv2, self.disp2 = self.up_block(upconv3, skip1, 32, output_disparity_flag=True)
            # upconv1
            upconv1, self.disp1 = self.up_block(upconv2, skip1, 16, output_disparity_flag=True, last_layer_flag=True)

    # def build_pose_model(self):
    #     with tf.variable_scope('pose'):
    #         input1 = Input(shape=(None, 128, 416, 3), name='input1')
    #         input2 = Input(shape=(None, 128, 416, 3), name='input2')
    #         input3 = Input(shape=(None, 128, 416, 3), name='input3')
    #         input1_econv5 = self.build_depth_model(input1)
    #         input1_econv5 = Conv2D(256, kernel_size=1, strides=(1, 1))(input1_econv5)
    #         input2_econv5 = self.build_depth_model(input2)
    #         input2_econv5 = Conv2D(256, kernel_size=1, strides=(1, 1))(input2_econv5)
    #         input3_econv5 = self.build_depth_model(input3)
    #         input3_econv5 = Conv2D(256, kernel_size=1, strides=(1, 1))(input3_econv5)
    #
    #         merge_layer = concatenate([input1_econv5, input2_econv5, input3_econv5])
    #         merge_layer = GlobalAveragePooling2D()(merge_layer)
    #         # pconv1
    #         pconv1 = Conv2D(256, kernel_size=3, strides=(2, 2))(merge_layer)
    #         # pconv2
    #         pconv2 = Conv2D(256, kernel_size=3, strides=(2, 2))(pconv1)
    #         # pconv3
    #         pconv3 = Conv2D(12, kernel_size=1, strides=(1, 1))(pconv2)

    def build_model(self):
        """
        Steps for building model:

            1. Construct image pyramid
            2. Initialize the build_depth_model() function

        :return:
        """
        with tf.variable_scope('model', reuse=self.reuse_variables):
            self.left_pyramid = self.scale_image(self.left, 4)
            if self.mode == 'train':
                self.right_pyramid = self.scale_image(self.right, 4)

            if config.DO_STEREO:
                self.model_input = K.concatenate([self.left, self.right], 3)
            else:
                self.model_input = self.left

            self.build_depth_model()

    def build_losses(self):
        # IMAGE RECONSTRUCTION
        # L1
        self.l1_left = [K.abs(self.left_est[i] - self.left_pyramid[i]) for i in range(4)]
        self.l1_reconstruction_loss_left  = [K.mean(l) for l in self.l1_left]
        self.l1_right = [K.abs(self.right_est[i] - self.right_pyramid[i]) for i in range(4)]
        self.l1_reconstruction_loss_right = [K.mean(l) for l in self.l1_right]

        # SSIM
        self.ssim_left = [self.SSIM( self.left_est[i],  self.left_pyramid[i]) for i in range(4)]
        self.ssim_loss_left  = [K.mean(s) for s in self.ssim_left]
        self.ssim_right = [self.SSIM(self.right_est[i], self.right_pyramid[i]) for i in range(4)]
        self.ssim_loss_right = [K.mean(s) for s in self.ssim_right]


        # WEIGTHED SUM
        self.image_loss_right = [config.ALPHA_IMAGE_LOSS * self.ssim_loss_right[i] + (1 - config.ALPHA_IMAGE_LOSS) * self.l1_reconstruction_loss_right[i] for i in range(4)]
        self.image_loss_left  = [config.ALPHA_IMAGE_LOSS * self.ssim_loss_left[i]  + (1 - config.ALPHA_IMAGE_LOSS) * self.l1_reconstruction_loss_left[i]  for i in range(4)]
        self.image_loss = K.sum([self.image_loss_left, self.image_loss_right], axis=0)

        # DISPARITY SMOOTHNESS
        self.disp_left_loss  = [K.mean(K.abs(self.disp_left_smoothness[i]))  / 2 ** i for i in range(4)]
        self.disp_right_loss = [K.mean(K.abs(self.disp_right_smoothness[i])) / 2 ** i for i in range(4)]
        self.disp_gradient_loss = K.sum([self.disp_left_loss, self.disp_right_loss], axis=0)

        # LR CONSISTENCY
        self.lr_left_loss = [K.mean(K.abs(self.right_to_left_disp[i] - self.disp_left_est[i])) for i in
                             range(4)]
        self.lr_right_loss = [K.mean(K.abs(self.left_to_right_disp[i] - self.disp_right_est[i])) for i in
                              range(4)]
        self.lr_loss = K.sum([self.lr_left_loss, self.lr_right_loss], axis=0)

        # TOTAL LOSS
        self.total_loss = self.image_loss + config.DISP_GRADIENT_LOSS_WEIGHT * self.disp_gradient_loss + config.LR_LOSS * self.lr_loss


    def build_outputs(self):
        # STORE DISPARITIES
        with tf.variable_scope('disparities'):
            self.disp_est  = [self.disp1, self.disp2, self.disp3, self.disp4]
            self.disp_left_est  = [K.expand_dims(d[:, :, :, 0], 3) for d in self.disp_est]
            self.disp_right_est = [K.expand_dims(d[:, :, :, 1], 3) for d in self.disp_est]

        if self.mode == 'test':
            return
        # GENERATE IMAGES
        with tf.variable_scope('images'):
            self.left_est  = [self.generate_image_left(self.right_pyramid[i], self.disp_left_est[i])  for i in range(4)]
            self.right_est = [self.generate_image_right(self.left_pyramid[i], self.disp_right_est[i]) for i in range(4)]

        # DISPARITY SMOOTHNESS
        with tf.variable_scope('smoothness'):
            self.disp_left_smoothness  = self.get_disparity_smoothness(self.disp_left_est,  self.left_pyramid)
            self.disp_right_smoothness = self.get_disparity_smoothness(self.disp_right_est, self.right_pyramid)

        with tf.variable_scope('left-right'):
            self.right_to_left_disp = [self.generate_image_left(self.disp_right_est[i], self.disp_left_est[i])  for i in range(4)]
            self.left_to_right_disp = [self.generate_image_right(self.disp_left_est[i], self.disp_right_est[i]) for i in range(4)]

    def build_summaries(self):
        # SUMMARIES
        with tf.device('/cpu:0'):
            for i in range(4):
                tf.summary.scalar('ssim_loss_' + str(i), self.ssim_loss_left[i] + self.ssim_loss_right[i], collections=self.model_collection)
                tf.summary.scalar('l1_loss_' + str(i), self.l1_reconstruction_loss_left[i] + self.l1_reconstruction_loss_right[i], collections=self.model_collection)
                tf.summary.scalar('image_loss_' + str(i), self.image_loss_left[i] + self.image_loss_right[i], collections=self.model_collection)
                tf.summary.scalar('disp_gradient_loss_' + str(i), self.disp_left_loss[i] + self.disp_right_loss[i], collections=self.model_collection)
                tf.summary.image('disp_left_est_' + str(i), self.disp_left_est[i], max_outputs=4, collections=self.model_collection)
                tf.summary.image('disp_right_est_' + str(i), self.disp_right_est[i], max_outputs=4, collections=self.model_collection)
                if True:
                    tf.summary.image('left_est_' + str(i), self.left_est[i], max_outputs=4, collections=self.model_collection)
                    tf.summary.image('right_est_' + str(i), self.right_est[i], max_outputs=4, collections=self.model_collection)
                    tf.summary.image('ssim_left_'  + str(i), self.ssim_left[i],  max_outputs=4, collections=self.model_collection)
                    tf.summary.image('ssim_right_' + str(i), self.ssim_right[i], max_outputs=4, collections=self.model_collection)
                    tf.summary.image('l1_left_'  + str(i), self.l1_left[i],  max_outputs=4, collections=self.model_collection)
                    tf.summary.image('l1_right_' + str(i), self.l1_right[i], max_outputs=4, collections=self.model_collection)

            if True:
                tf.summary.image('left',  self.left,   max_outputs=4, collections=self.model_collection)
                tf.summary.image('right', self.right,  max_outputs=4, collections=self.model_collection)

