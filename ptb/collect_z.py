import tensorflow as tf

s = tf.InteractiveSession()

# inputs = tf.constant([[1,1,1],[2,2,2],[3,3,3]])
# res = tf.reshape(tf.concat(inputs, 1), [-1, 4])
# print s.run([inputs, tf.concat(inputs, 1)])




inputsBatch = tf.constant([[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16], [17,18,19,20]])
z_s = tf.constant([[0.1,0.1,0.1,0.1], [0.2,0.2,0.2,0.2], [0.3,0.3,0.3,0.3], [0.4,0.5,0.6,0.7], [0.4,0.5,0.6,0.7]])

def getLowerDiag(inputs):
	inputs_matrix = tf.reshape(tf.tile(inputs, [tf.shape(inputs)[0]]), [-1,tf.shape(inputs)[0]])
	result = tf.matrix_band_part(inputs_matrix, -1, 0)
	return result


resultBatch = tf.map_fn(lambda x: getLowerDiag(x), tf.transpose(inputsBatch))
resultBatch = tf.transpose(resultBatch, perm=[1, 0, 2]) 

temp = tf.map_fn(lambda x: getLowerDiag(x), tf.transpose(z_s))
z_s = tf.transpose(temp, perm=[1, 0, 2]) 

weights = tf.ones_like(inputsBatch)
weights = tf.map_fn(lambda x: getLowerDiag(x), tf.transpose(weights))
weights = tf.transpose(weights, perm=[1, 0, 2])


print s.run([resultBatch, z_s, weights])


