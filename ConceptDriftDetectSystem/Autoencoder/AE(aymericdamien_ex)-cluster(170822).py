# -*- coding: utf-8 -*-

""" Auto Encoder Example.
Using an auto encoder on MNIST handwritten digits.
References:
    Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner. "Gradient-based
    learning applied to document recognition." Proceedings of the IEEE,
    86(11):2278-2324, November 1998.
Links:
    [MNIST Dataset] http://yann.lecun.com/exdb/mnist/
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True, validation_size = 100)

#print(mnist.train.labels[1]) # mnist data print
#print(mnist.train.images[1]) # mnist data print
'''
for i in range(12):
    for j in range(784):
        print(mnist.train.images[i][j], end=",")
    print("") 
'''

# Parameters
display_step = 1
examples_to_vectorize = 1285

CLUSTERS = 120
examples_to_show =[[280, 525, 852, 856, 1209, 1210, 1211, 1212, 1213], [198, 216, 1010, 1170, 1261], [252, 303, 384, 411, 453, 627, 632, 748, 756, 958, 1037, 1045], [16, 120, 159, 241, 879, 1011, 1068, 1132], [742, 760, 1120], [576, 586, 587], [147, 296, 348, 408, 422, 425, 471, 492, 509, 716, 730, 920], [76, 413, 863, 1066, 1176], [641, 645, 849, 1061, 1067], [86, 158, 178, 213], [114, 176, 468, 823], [196, 197, 199, 200, 201, 204, 205, 206, 207, 279], [87, 268, 332, 360, 626, 917, 927, 1022, 1023, 1272], [47, 127, 346, 355, 383, 759, 792, 833, 1020, 1024, 1137, 1144], [256, 302, 323, 326, 330, 351, 358, 381, 442, 635, 701, 771], [52, 78, 293, 615, 672, 717, 934, 941, 962, 969, 983, 1018], [107, 221, 614, 653, 778, 820, 915, 964, 1056], [77, 210, 287, 725], [226, 387, 397, 429, 652, 698, 706, 712, 734, 743, 764, 1049], [29, 160, 254, 258, 297, 324, 349, 418, 426, 427, 433, 514], [267, 423, 438, 502, 599, 728, 733, 888, 898, 908, 916, 919], [238, 301, 420, 473, 483, 497, 639, 665, 900, 935, 1005, 1075], [99, 103, 247, 444, 470, 494, 596, 661, 769, 795, 796, 889], [85, 362, 633, 709, 723, 731, 735, 989, 995, 1000, 1053, 1270], [230, 476, 616, 886, 914], [40, 136, 143, 255, 439, 449, 500, 612, 617, 625, 649, 741], [72, 74, 286, 400, 403, 638, 845, 859, 865, 867, 868, 869], [175, 181, 219, 459, 624, 851, 872, 1172, 1186, 1191, 1194, 1196], [119, 1109], [843, 1131, 1166, 1183, 1229, 1265], [125, 148, 245, 350, 364, 385, 398, 469, 518, 588, 697, 802], [264, 269, 312, 365, 393, 414, 619, 669, 700, 705, 761, 772], [73, 166, 192, 284, 684, 1198, 1199], [112, 683, 1264], [7, 622, 640, 1103], [82, 214, 505, 537, 539, 589, 794, 818, 912, 975, 998, 1128], [570, 572, 573, 574, 575, 580, 581, 582, 583, 585], [46, 311, 353, 379, 386, 396, 516, 655, 691, 694, 960, 985], [137, 146, 510, 533, 541, 543, 681, 947, 1104, 1136, 1174, 1180], [407, 548, 552, 554, 643, 786, 814, 883], [6, 14, 31, 35, 42, 98, 104, 319, 647, 703, 1032], [1145], [224], [113, 172, 182, 685, 830, 864], [1, 2, 53, 88, 90, 150, 347, 388, 446, 603, 607, 654], [17, 1099, 1143, 1154, 1155, 1158, 1168, 1184, 1228], [108, 390], [68, 139, 282, 460, 488, 679, 892, 913, 1087, 1094, 1095, 1108], [376, 457, 478, 538], [153, 188, 211, 642, 816, 885, 1165], [51, 132, 244, 591, 945, 1177], [64, 89, 110, 316, 650, 811, 1013, 1027, 1034, 1224], [209, 563, 564, 567, 578], [248, 354, 367, 648, 667, 699, 707, 727, 931, 965, 978, 988], [165, 169, 170, 177, 184, 228, 233, 534], [129, 515, 899, 1244], [140, 428, 454, 475, 517, 800, 808, 906, 949, 991, 1003, 1019], [41, 83, 677, 720, 787, 1007, 1058, 1221], [152, 237, 431, 479, 501, 508, 605, 757, 891, 909, 944, 1055], [272, 337, 601, 675, 788, 1069, 1081, 1122, 1274], [135, 155, 191, 225, 374, 487, 549, 555, 680, 729, 799, 825], [69, 185, 558, 562, 847, 853, 855, 875, 1064, 1249], [208, 399, 465, 1001, 1107, 1173, 1197], [54, 222, 291, 298, 382, 416, 499, 557, 600, 610, 621, 763], [292, 307, 313, 320, 327, 474, 503, 511, 671, 1046], [565, 568, 569, 571, 577, 579, 584], [164, 507, 521, 528, 529, 535, 547, 604, 829, 1235], [55, 97, 300, 335, 340, 370, 371, 377, 447, 658, 704, 773], [194, 261, 451, 744, 955, 996, 1223, 1242], [8, 21, 27, 48, 105, 314, 338, 356, 391], [79, 180, 186, 193, 243, 262, 894], [56, 271, 289, 310, 394, 432, 443, 689, 710, 928, 956, 1004], [167, 168, 637, 848, 858, 866, 870, 871, 1200, 1201, 1202], [275, 668, 670, 674, 695, 752, 878, 976, 987, 990, 1276, 1284], [405, 461, 462, 553, 644, 837, 1106, 1171, 1179, 1185, 1189, 1192], [123, 162, 187, 220, 540, 593, 842, 857, 881, 1175], [96, 100, 121, 130, 142, 144, 151, 232, 234, 687, 754, 810], [4, 43, 299, 333, 334, 363, 380, 440, 609, 686, 693, 950], [60, 276, 305, 309, 630, 755], [33, 141, 236, 242, 249, 270, 435, 445, 452, 484, 512, 606], [30, 290, 325, 738, 1017, 1234], [138, 251, 266, 308, 409, 412, 490, 504, 531, 532, 590, 595], [24, 36, 38, 80, 91, 295, 318, 321, 660, 662, 968, 1021], [111, 183, 203, 218, 253, 456, 458, 467, 482, 486, 620, 804], [0, 9, 329, 721], [161, 231, 259, 343, 378, 392, 651, 907, 970, 1047, 1124, 1257], [223, 288, 489, 496, 523, 822, 836, 1040, 1161, 1167, 1169, 1182], [190, 767, 846, 893, 1156], [315, 341], [250, 421, 495, 801, 839, 901, 1101, 1153, 1260], [546, 597, 608, 831, 884, 1105, 1116, 1147], [95, 122, 491, 506, 592, 598, 708, 747, 798, 805, 812, 821], [70, 75, 116, 401, 463, 466, 559, 560, 566, 1205, 1206, 1207], [11, 32, 37, 39, 44, 58, 957, 992, 1273, 1280], [133, 195, 273, 283, 285, 322, 753, 813, 1060, 1082, 1275], [124, 131, 149, 163, 212, 227, 235, 480, 513, 542, 544, 550], [15, 18, 49, 63, 339, 711, 997, 1031], [20, 128, 173, 189, 217, 441, 450, 464, 477, 485, 602, 781], [22, 23, 50, 57, 92, 246, 737, 775], [71, 115, 117, 118, 281, 402, 526, 844, 861, 1214], [278, 493, 498, 522, 530, 536, 556, 611, 817, 834, 840, 841], [3, 93, 106, 109, 171, 263, 277, 437, 657, 803, 806, 1016], [145, 239, 417, 646, 774, 1089], [62, 94, 126, 215, 369, 372, 664, 745, 746, 1028, 1038, 1042], [45, 306, 317, 331, 352, 359, 373, 415, 1036], [156, 561, 850, 854, 860, 862, 1187, 1188, 1227], [59, 101, 366, 436, 455, 481], [26, 240, 880], [344, 688, 732, 739, 942, 984, 1012, 1026, 1054], [520, 629, 634, 673, 678, 740, 762, 768, 770, 896, 933, 961], [28, 294, 342, 345, 357, 434, 448, 718, 963], [12, 19, 65, 67, 84, 410, 659, 682, 726, 779, 783], [25, 66, 265, 336, 395, 696, 702, 951], [472, 551, 838, 873, 876, 1063, 1178, 1232, 1239], [10, 229, 361, 389, 424, 430, 713, 736, 751, 807, 929, 930], [154, 202, 404, 406, 524, 545, 623, 784, 1100, 1216], [157, 174, 179, 260, 519, 527, 777, 897, 1250, 1254], [257, 274, 419, 628, 692, 715, 719, 750, 780, 782, 790, 815], [5, 61, 81, 102, 134, 304, 368, 375, 636, 971, 1030], [13, 34, 328, 676, 714, 966, 1074]]


# Network Parameters
n_hidden_1 = 64 # 1st layer num features
n_hidden_2 = 25 # 2nd layer num features
n_input = 1440 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
    'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b2': tf.Variable(tf.random_normal([n_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    # Decoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2

# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

'''
# Define loss and optimizer, minimize the squared error
cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)
'''

# Initializing the variables
init = tf.global_variables_initializer()

# Model Saver op
saver = tf.train.Saver()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    
    ckpt = tf.train.get_checkpoint_state("./trained_AE")
    if ckpt and ckpt.model_checkpoint_path:
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        # Assuming model_checkpoint_path looks something like:
        #   /my-favorite-path/cifar10_train/model.ckpt-0,
        # extract global_step from it.
        global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
        print('No checkpoint file found')
    '''
    # Applying encode and decode over test set
    encode = sess.run(
        encoder_op, feed_dict={X: mnist.test.images[:examples_to_show]})
    #print(encode[0])
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})
    '''
    

    # Applying encode and decode over test set
    encode = sess.run(
        encoder_op, feed_dict={X: mnist.test.images[:examples_to_vectorize]})
    #print(encode[0])
    encode_decode = sess.run(
        y_pred, feed_dict={X: mnist.test.images[:examples_to_vectorize]})
        
    for y in range(0, CLUSTERS):
        # Compare original images with their reconstructions
        f, a = plt.subplots(3, 12, figsize=(12, 3))
        
        z = 0;
        for i in examples_to_show[y]:
            a[0][z].imshow(np.reshape(mnist.test.images[i], (40, 36)), clim=(0.0,1.0))
            a[1][z].imshow(np.reshape(encode[i], (5, 5)), clim=(0.0,1.0))
            '''
            # print vectors to standard output
            for j in range(n_hidden_2):
                print(encode[i][j], end=" ")
            print("")
            '''
            a[2][z].imshow(np.reshape(encode_decode[i], (40, 36)), clim=(0.0,1.0))
            z = z + 1;
            #aaa = open("./encoded_data2", "w")
        f.savefig("./" + str(y))
        f.clear()
        #plt.gcf().clear()
    
    f.show()
    plt.draw()
    plt.waitforbuttonpress()
