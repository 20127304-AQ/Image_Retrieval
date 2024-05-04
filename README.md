# Image Retrieval
Retrieve top k similar images using Haar Cascade, MobileNetV2, KDTree.

# Work flow

![Slide_20127247_20127304 pptx](https://github.com/20127304-AQ/Image-Retrieval-/assets/79797296/f81729ef-4daf-4598-9679-3b2eaab68575)

- Dataset preparation:
  - Using face-detection classifier model Haar Cascade for all images in the dataset.
  - Apply pretrained model MobileNetV2 for extracting all images in the dataset into feature vectors.
  - Save all feature vectors into dictionary and load into .pkl file.
  - Load feature vectors into module.
  - Build KDTree by these feature vectors for optimize indexing step.
- Retrieve image:
  - Upload input image into the module.
  - Apply face-detection classifier model Haar Cascade for input image.
  - Apply pretrained model MobileNetV2 for extracting input image into feature vector.
  - Retrieve top k similar images from KDTree.
  - Display images.
 
# Technology
- Haar Cascade
- MobileNet V2
- KDTree
- Flask

# Reference
- [Haar Cascade](https://docs.opencv.org/4.x/db/d28/tutorial_cascade_classifier.html)
- [MobileNetV2](https://pytorch.org/vision/main/transforms.html)
- [KDTree](https://viblo.asia/p/gioi-thieu-thuat-toan-kd-trees-nearest-neighbour-search-RQqKLvjzl7z)
- [Flask](https://flask.palletsprojects.com/en/3.0.x/)
