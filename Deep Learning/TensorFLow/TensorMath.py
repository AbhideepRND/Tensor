import tensorflow as tf

x = tf.constant([100,200,300], name='x')
y = tf.constant([1,2,3], name='y')

sum_x = tf.reduce_sum(x, name='reduce_x')
prod_y = tf.reduce_prod(y, name='reduce_y')

final_div = tf.div(sum_x,prod_y, name="final_div")
final_mean = tf.reduce_mean([sum_x,prod_y], name="final_mean")

sess = tf.Session()

print ("X :- ", sess.run(x))
print ("Y :- ", sess.run(y))
print ("Sum Of X :- ", sess.run(sum_x))
print ("Multiplication of Y :- ", sess.run(prod_y))
print ("SUM_X/PROD_Y :- ", sess.run(final_div))
print ("Average between SUM_X and PROD_Y :- ", sess.run(final_mean))
print ("Rank of tensor X ", tf.rank(x))

writer = tf.summary.FileWriter("/home/liveyoung/django/log/TensorMath", sess.graph)
writer.close()

sess.close()
