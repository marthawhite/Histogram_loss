# Age Estimation

The [age estimation](https://paperswithcode.com/task/age-estimation) problem is about predicting the age of a person from a picture of their face.

We ran experiments with and without face alignment using points detected by [MTCNN](https://github.com/ipazc/mtcnn) ([Zhang *et al.* 2016](https://arxiv.org/ftp/arxiv/papers/1604/1604.02878.pdf)).

## Datasets
 - [FG-NET](https://yanweifu.github.io/FG_NET_data/) - Contains 1002 images of 82 people from 0 to 69 years old ([Lanitis *et al.* 2002](https://doi.org/10.1109/34.993553))
 - [UTKFace](https://susanqq.github.io/UTKFace/) - Consists of 20k+ aligned and cropped images from 0 - 116 years old ([Zhang *et at.* 2017](https://arxiv.org/pdf/1702.08423v2.pdf))
 - [MegaAge Asian]() - Contains 40k images of Asian faces from ages 0 - 70 ([Zhang *et al.* 2017](https://arxiv.org/pdf/1708.09687v2.pdf))

 ## Base Models
  - Xception - 22.9 M parameters ([Chollet 2017](https://arxiv.org/pdf/1610.02357.pdf))
  - VGG16 - 138.4 M paramters ([Simonyan and Zisserman 2015](https://arxiv.org/pdf/1409.1556.pdf))