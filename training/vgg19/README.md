VGG19
===

<a name="config"></a>
## Learning configs

* input shape = (3, 64, 64)
* Don't use image normalization
* Use L2(0.01) regularization after Conv2D layers
* Use Dropout(0.5) after Dense layers
* Shift and rotate training images
* Amdam optimizer, learning rate=0.001

