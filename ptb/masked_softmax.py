import tensorflow as tf


# Ignore those that aren't 1's in the softmax.
def softmax_with_mask(valuesAndMask):
	values, masks = valuesAndMask[0], tf.to_int32(valuesAndMask[1])
	newData = tf.dynamic_partition(values, masks, 2)
	softMData = tf.nn.softmax(newData[1]) 
	# Append back the zeros. "zeros_like" wasn't atually necessary.
	result = tf.concat([softMData, tf.zeros_like(newData[0])],0)
	return result


def softmax_with_batch_and_mask(valuesBatch, maskBatch):
	# Prepare the data for our map-reduce that feeds into softmax.
	combined = tf.map_fn(lambda (x, y): (x, y), (valuesBatch, maskBatch))
	combined = tf.transpose(combined, perm=[1, 0, 2]) 
	resultBatch = tf.map_fn(softmax_with_mask, combined)
	return resultBatch	


s = tf.InteractiveSession()
# valuesBatch = tf.constant([[0.1,0.1,0.1,0.1, 0, 0], [0.2,0.2,0.2,0.2, 0, 0], [0.3,0.3,0.3,0.3, 0, 0], [0.4,0.5,0.6,0.7, 0, 0]], tf.float32)
# maskBatch = tf.constant([[1,1,1,1,0,0],[1,1,1,1,0,0],[1,1,1,1,0,0],[1,1,1,1,0,0]], tf.float32)
# resultBatch = softmax_with_batch_and_mask(valuesBatch, maskBatch)
# print s.run(resultBatch)



# def softmax_with_matrix_and_mask(values, masks):




v = tf.constant([[[0.1,0.1,0.1,0.1, 0, 0], [0.2,0.2,0.2,0.2, 0, 0]], [[0.3,0.3,0.3,0.3, 0, 0], [0.4,0.5,0.6,0.7, 0, 0]]], tf.float32)
m = tf.constant([[[1,1,1,1,0,0],[1,1,1,1,0,0]],[[1,1,1,1,0,0],[1,1,1,1,0,0]]], tf.float32)

m = tf.nn.softmax(tf.log(m) + v)

# results = softmax_with_batch_and_mask(tf.concat(v,0), tf.concat(m,0))
print s.run(m)

# softmax_with_batch_and_mask(results[0], results[1])
