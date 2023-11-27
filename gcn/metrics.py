import tensorflow.compat.v1 as tf


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking."""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_accuracy(preds, labels, mask):
    """Accuracy with masking."""
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)

def masked_precision(preds, labels, mask):
    TP = tf.math.multiply(tf.subtract(tf.ones_like(tf.argmax(preds,1),tf.int64),tf.argmax(preds,1)),tf.subtract(tf.ones_like(tf.argmax(labels,1),tf.int64),tf.argmax(labels,1)))
    FP = tf.math.multiply(tf.subtract(tf.ones_like(tf.argmax(preds,1),tf.int64),tf.argmax(preds,1)),tf.argmax(labels,1))
    TP_all = tf.cast(TP, tf.float32)
    FP_all = tf.cast(FP, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    TP_all *= mask
    FP_all *= mask    
    return tf.reduce_sum(TP_all)/(tf.reduce_sum(TP_all)+tf.reduce_sum(FP_all))
    
def masked_recall(preds, labels, mask):
    TP = tf.math.multiply(tf.subtract(tf.ones_like(tf.argmax(preds,1),tf.int64),tf.argmax(preds,1)),tf.subtract(tf.ones_like(tf.argmax(labels,1),tf.int64),tf.argmax(labels,1)))
    FN = tf.math.multiply(tf.argmax(preds,1),tf.subtract(tf.ones_like(tf.argmax(labels,1),tf.int64),tf.argmax(labels,1)))
    TP_all = tf.cast(TP, tf.float32)
    FN_all = tf.cast(FN, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    TP_all *= mask
    FN_all *= mask    
    return tf.reduce_sum(TP_all)/(tf.reduce_sum(TP_all)+tf.reduce_sum(FN_all))

