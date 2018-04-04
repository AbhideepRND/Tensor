import tensorflow as tf

a = tf.constant(3.4, name="const_a")
b = tf.constant(4.5, name="const_b")
c = tf.constant(3.4, name="const_c")
d = tf.constant(100.2, name="const_d")

square = tf.square(a, name="square_a")
pow = tf.pow(b,c, name="pow_b_c")
sqrt = tf.sqrt(d, name="sqrt_d")

final_sum= tf.add_n([square, pow, sqrt],name="final_sum")

sess=tf.Session()

print("Square of a ",sess.run(square) )
print("Power of b ^ c ",sess.run(pow) )
print("Square root of d ",sess.run(sqrt) )
print("Final sum ",sess.run(final_sum) )

writer = tf.summary.FileWriter("/home/liveyoung/django/log/Computation", sess.graph)

writer.close()

#writer = tf.train.SummaryWriter('/home/liveyoung/django/Deep Learning/TensorBoard/Computation', sess.graph)
#writer.close()
sess.close()



