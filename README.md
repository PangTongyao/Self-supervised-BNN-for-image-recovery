# Self-supervised Bayesian Deep Learning for Image Recovery with Applications to  Compressive Sensing

This repository is an Pytorch implementation of the paper ["Self-supervised Bayesian Deep Learning for
Image Recovery with Applications to  Compressive Sensing"](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123560460.pdf).

## How to use

The file *demo_image.py*  provides a demo for solving block-wise Gaussian compressive sensing problem with a test image *boats.tif*. No training dataset is required. 



```python
python3 demo_image.py --gpu 0 ----CS_ratio 40 
```



## How to cite

```
@article{pang2020self,
  title={Self-supervised Bayesian Deep Learning for Image Recovery with Applications to Compressive Sensing},
  author={Pang, Tongyao and Quan, Yuhui and Ji, Hui},
  booktitle={ECCV},
  year={2020}
}
```

