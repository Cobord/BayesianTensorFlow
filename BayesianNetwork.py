import tensorflow as tf

with tf.variable_scope("Rain") as scope:
    Rprob=tf.Variable([.2,.8],dtype=tf.float32)

with tf.variable_scope("Sprinkler") as scope:
    SprobRF=tf.Variable([.4,.6],dtype=tf.float32)
    SprobRT=tf.Variable([.01,.99],dtype=tf.float32)
    with tf.name_scope("Rain") as scope2:
        SprobR = Rprob
    SprobRU = tf.stack([SprobRT,SprobRF],axis=0)
    # [[STRT,SFRT],[STRF,SFRF]]
    SprobS=tf.matmul(tf.expand_dims(SprobR,0),SprobRU)

with tf.variable_scope("GrassWet") as scope:
    GprobSFRF=tf.Variable([0,1.0],dtype=tf.float32)
    GprobSFRT=tf.Variable([.8,.2],dtype=tf.float32)
    GprobSTRF=tf.Variable([.9,.1],dtype=tf.float32)
    GprobSTRT=tf.Variable([.99,.01],dtype=tf.float32)
    ## remove the knowledge of rain
    with tf.name_scope("Rain") as scope2:
        GprobR = Rprob
    GprobSFRU = tf.stack([GprobSFRT,GprobSFRF],axis=0)
    # [[GTSFRT,GFSFRT],[GTSFRF,GFSFRF]]
    GprobSF=tf.matmul(tf.reshape(GprobR,[1,2]),GprobSFRU)
    # [RT,RF]*[[GTSFRT,GFSFRT],[GTSFRF,GFSFRF]]
    GprobSTRU = tf.stack([GprobSTRT,GprobSTRF],axis=0)
    GprobST=tf.matmul(tf.reshape(GprobR,[1,2]),GprobSTRU)
    ## remove the knowledge of sprinkler
    with tf.name_scope("Sprinkler") as scope2:
        GprobSRT = SprobRT
        GprobSRF = SprobRF
    GprobSURT = tf.stack([GprobSTRT,GprobSFRT],axis=0)
    # [[GTSTRT,GFSTRT],[GTSFRT,GFSFRT]]
    GprobRT=tf.matmul(tf.reshape(GprobSRT,[1,2]),GprobSURT)
    GprobSURF = tf.stack([GprobSTRF,GprobSFRF],axis=0)
    # [[GTSTRF,GFSTRF],[GTSFRF,GFSFRF]]
    GprobRF=tf.matmul(tf.reshape(GprobSRF,[1,2]),GprobSURF)
    ## remove the knowledge of both
    GprobRU=tf.stack([GprobRT,GprobRF],axis=0)
    Gprob=tf.matmul(tf.reshape(GprobR,[1,2]),tf.reshape(GprobRU,[2,2]))

sess=tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

with tf.name_scope('Rain') as scope:
    print(sess.run(Rprob))

with tf.name_scope('GrassWet') as scope:
    print(sess.run(GprobSF))

with tf.name_scope('Rain') as scope:
    Rprob=tf.assign(Rprob,[.3,.7])
    print(sess.run(Rprob))

with tf.name_scope('GrassWet') as scope:
    print(sess.run(GprobSF))
    
with tf.name_scope('Rain') as scope:
    Rprob=tf.assign(Rprob,[.8,.2])
    print(sess.run(Rprob))

with tf.name_scope('GrassWet') as scope:
    print(sess.run(GprobSF))