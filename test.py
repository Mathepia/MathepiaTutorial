import torch

flag = torch.cuda.is_available()
print(flag)

ngpu = 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
print(device)
print(torch.cuda.get_device_name(0))
print(torch.rand(3, 3).cuda())

# test
#%%
import tensorflow as tf

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)) as sess:
    a = tf.constant(1)
    b = tf.constant(3)
    c = a + b
    print('结果是：%d\n值为：%d' % (sess.run(c),sess.run(c)))