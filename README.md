# VAE
VAE models for experiments and education usage.

## Repo Structure

### data

Saving data of MNIST.

### vae_tf.py

A python script implemented by tensorflow and experiments conducted.

### vae_keras.py

A python script implemented by keras and experiments conducted.

## Environment

* python2
* Tensorflow1.4.1
* Keras2.1.5

## Usage

* the keras version：

`python vae_keras.py`

* the tensorflow version：

`python vae_tf.py`

`tensorboard --logdir=tf_record/`

## Experiment Results

### keras version

epoch=5, distribution of **z**:

![](http://odjt9j2ec.bkt.clouddn.com/vae-epoch5_z.png)

epoch=5, the output of generator(decoder):

![](http://odjt9j2ec.bkt.clouddn.com/vae-epoch5_x.png)

epoch=50, distribution of **z**:

![](http://odjt9j2ec.bkt.clouddn.com/vae-epoch50_z.png)

epoch=50, the output of generator(decoder):

![](http://odjt9j2ec.bkt.clouddn.com/vae-epoch50_x.png)

### tensorflow version

training for 1000 steps, and the training loss is as follows：

![](http://odjt9j2ec.bkt.clouddn.com/vae-trainloss.png)

## More Information

Please visit http://skyhigh233.com/blog/2018/04/04/vae/ for more information.

## License

MIT LICENSE
