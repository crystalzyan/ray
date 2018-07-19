ML: Distributed Stochastic Gradient Descent
===========================================

Gradient descent is a staple training algorithm in machine learning for 
iteratively optimizing the accuracy of machine learning models. By calculating 
the gradient/direction to incrementally improve our model parameters at every 
iteration step, gradient descent can converge to find the optimal model 
parameters. 

To do so, however, gradient descent must repeatedly compute the gradient over 
the entire dataset for every iteration step. With large datasets, this can get 
computationally expensive.

So long as the samples in our dataset were obtained independently from each 
other, **stochastic gradient descent,** otherwise known as SGD, is a valid 
performance shortcut to traditional gradient descent. By calculating 
the gradient over a random mini-batch (subset) of data samples instead of 
over the batch (full dataset), SGD saves computation at each iteration. 


How Ray's Distributed SGD Works
-------------------------------

Ray offers a dataparallel distributed SGD algorithm to further speed up 
iterations, by taking advantage of how gradient computation of the mini-batch 
can be broken up and done independently per sample. 

After initializing a copy of the model and an equal-sized subset of the data 
to store in each device (CPU/GPU) per remote worker, Ray takes the following 
steps to perform a single distributed SGD iteration:

1. Each worker calculates a gradient 'shard' from their subset of samples. These
   gradient computations across the workers are calculated in parallel.
2. The driver accumulates all gradient shards back together into the gradient
   to use for the weights update.
3. The driver updates the model weights in each worker in parallel using the
   accumulated gradient. 

Through Ray's stateful workers and shared memory stores, Ray can reduce the
significant performance overhead it would otherwise take to sync the entire 
model state and to accumulate gradient shards across workers for every 
iteration. Ray also saves time from having to randomly distribute mini-batch 
samples from the driver across workers each iteration (as in typical dataparallel 
distributed SGD), by having each worker internally store a fixed subset of 
mini-batch data for all iterations.

Therefore, by parallelizing mini-batch computation with Ray, we can take 
advantage of the benefits of larger effective mini-batch sizes. This reduces the 
number of overall iterations/loops SGD needs to converge towards optimal model 
parameters and to finish.

Currently, distributed SGD in Ray is still experimental and can be found in our 
`source code here`_. 

.. _`source code here`: https://github.com/ray-project/ray/compare/master...ericl:sgd


Invoking Distributed SGD in Ray
-------------------------------

Here is the basic template for invoking distributed SGD in Ray. For this example, 
the template runs on a pre-defined convolutional neural net classifier 
``TFBenchModel`` that internally generates its own subset of mock image data:

.. code-block:: python

	from __future__ import absolute_import
	from __future__ import division
	from __future__ import print_function

	import ray

	import argparse
	import numpy as np
	import tensorflow as tf

	from test_model import TFBenchModel
	from sgd import DistributedSGD


	if __name__ == "__main__":
	    ray.init()
	    
	    model_creator = (
	        lambda worker_idx, device_idx: TFBenchModel(batch=1, use_cpus=True))

	    sgd = DistributedSGD(
	        model_creator, num_workers=2, devices_per_worker=2, use_cpus=True)

	    for _ in range(100):
	        loss = sgd.step()
	        print("Current loss", loss)

Let's break this down.

After importing and initializing Ray, we create a function ``model_creator``. 
When called, our function generates our data subset and model according to the 
TensorFlow configurations we have defined and provided in some other file as 
``TFBenchModel`` (not provided here). This function will be used for generating 
a copy of our model, along with a data subset, on each device (CPU/GPU) in each 
worker. 

For Ray's internal use, this function takes in worker and device IDs from Ray 
as parameters:

.. code-block:: python

	model_creator = (
	        lambda worker_idx, device_idx: TFBenchModel(batch=1, use_cpus=True))

In this example, ``TFBenchModel`` can customize the ``batch`` or subset number 
of dataset images to generate, and whether to use CPUs instead of GPUs. 
The default behavior of ``TFBenchModel`` is to initialize a data subset with 64 
images in a device, and to use GPUs.

With our next statement, we pass in our ``model_creator`` function into Ray,
specifying to use 2 workers with 2 CPUs per worker:

.. code-block:: python

	sgd = DistributedSGD(
	        model_creator, num_workers=2, devices_per_worker=2, use_cpus=True)

Finally, it's time to run and iterate over our distributed SGD set-up! Each call 
to ``sgd.step()`` performs an iteration of stochastic gradient descent using our 
``TFBenchModel`` mock data. ``sgd.step()`` then returns us the accumulated loss 
(model training error) over the workers after that iteration. 

Helpfully, ``sgd.step()`` defaults to providing verbose output, timing Ray as
it performs each stage in the iteration.

We run 100 iterations and print out the loss at each iteration, so that we can 
also see our model progressively improving at minimizing its training error 
(``TFBenchModel`` calculates this via softmax cross-entropy loss):

.. code-block:: python

	for _ in range(100):
	        loss = sgd.step()
	        print("Current loss", loss)

Here's how our code's output might look in its first three iterations when running 
our template with verbose ``DistributedSGD`` output:

.. code-block:: bash
	
	# First iteration
	compute grad interior time 9.909772872924805
	compute grad interior time 10.037450790405273
	compute all grads time 32.77693462371826
	grad reduce time 0.21662235260009766
	apply grad interior time 0.412905216217041
	apply grad interior time 0.422518253326416
	apply all grads time 0.621117353439331
	Current loss 7.801279306411743

	# Second iteration
	compute grad interior time 5.760771989822388
	compute grad interior time 5.785480976104736
	compute all grads time 5.906821250915527
	grad reduce time 0.19811248779296875
	apply grad interior time 0.2362966537475586
	apply grad interior time 0.23021912574768066
	apply all grads time 0.4307553768157959
	Current loss 7.583970069885254

	# Third iteration
	compute grad interior time 5.780273675918579
	compute grad interior time 5.912511825561523
	compute all grads time 6.0070977210998535
	grad reduce time 0.18091201782226562
	apply grad interior time 0.2546241283416748
	apply grad interior time 0.2835371494293213
	apply all grads time 0.47388339042663574
	Current loss 7.070155143737793

Ray's verbose output provides both the timings within each worker, and the 
timings over all workers. Because we are using 2 workers in this example, 
``compute grad interior time`` and ``apply grad interior time`` are printed
twice, once for each worker. 

Additionally, ``compute grad`` refers to the first half of the distributed 
SGD algorithm, when the gradient is computed for each sample, and 
``apply grad`` refers to the second half when the workers update the 
model parameters by applying the gradient. Accumulating the gradient shards 
into an average gradient on the driver is not timed, but takes place between 
these two stages.

Therefore, we can see that the initialization of the distributed SGD
takes a while, as the ``compute grad`` step in the first iteration takes
32.7 seconds. However, in all future iterations, computing the gradient takes
an average of 6 seconds, and updating the model weights takes only half a 
second. 

We can also see that overhead (from fetching the two workers' results back to
the driver) is kept at a minimum in Ray, because it takes only a little over 0.1 
seconds to fetch all gradients, on top of the 5.7 seconds it takes for the worker 
to be ready.

.. note:: TO-DO 

	How do we get our model/model weights back afterwards so that we can use our 
	SGD-trained model on future data? Users only have access to the DistributedSGD
	object, but not the model copies on the remote workers?


Customizing our Model and Data for Ray's Distributed SGD
------------------------------------------------------

Of course, we would preferrably wish to fill in the above SGD template with
our own model and dataset. We can replace ``TFBenchModel`` by defining our
own model and data as a class instead.

.. note:: TO-DO 

	To be continued...
