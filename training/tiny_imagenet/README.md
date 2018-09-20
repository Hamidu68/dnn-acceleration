Tiny ImageNet
===

<a name="toc"></a>
## Dataset

	Tiny Imagenet has 200 classes. Each class has 500 training images, 50 validation images, and 50 test images.
	We have released the training and validation sets with images and annotations. We provide both class labels and bounding boxes as annotations.
	You can download the data at here: https://tiny-imagenet.herokuapp.com/

- directory
	```
	tiny-imagenet-200/
	¦¦	test/
		¦¦	images/
			¦¦	test_0.JPEG
				...
				test_9999.JPEG
	¦¦	train/
		¦¦	n01443537/
			¦¦	images/
				¦¦	n01443537_0.JPEG
					...
			¦¦	n01443537_boxes.txt
		¦¦	n01629819/
			¦¦	...
		¦¦	n01641577/
			...	
	¦¦	val/
		¦¦	images/
			¦¦	val_0.JPEG
				...
				val_9999.JPEG
		¦¦	val_annotations
	¦¦	wnids.txt
	¦¦	words.txt
	```
