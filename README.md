# marathi-neural-machine-translation
Code for building a Marathi-English machine translation neural network model. 

# Results
models/metric | BLEU Score
------------ | -------------
Google Translate _(benchmark)_ | 63.8
Seq2Seq with Attention (100 epochs) | 52.8


# References

Links of references which we refered to when making this project: 

## Code:

### PyTorch:
*   [PyTorch's 60 min Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
*   [Ben Trevett's Seq2Seq tutorials in Pytorch](https://github.com/bentrevett/pytorch-seq2seq)
*   [PyTorch's Official TorchText tutorial](https://pytorch.org/tutorials/beginner/torchtext_translation_tutorial.html)
*   [Pytorch's official Seq2Seq tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
*   [PyTorch's nn.Transformer tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html)

### TensorFlow:
*   [Tensorflow Officical NMT with Attention Tutorial](https://www.tensorflow.org/tutorials/text/nmt_with_attention)
*   [Tensorflow Official NMT with Transformer Tutorial](https://www.tensorflow.org/tutorials/text/transformer)
*   [Tensorflow saving checkpoints tutorial](https://www.tensorflow.org/guide/checkpoint)
*   [Effectively use TF 2.0](https://blog.tensorflow.org/2019/02/effective-tensorflow-20-best-practices.html)

### Azure:
*   [Blob storage tutorial](https://docs.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-python)

## Papers:
*   [Seq2Seq with Attention (Bahdanau et al.)](https://arxiv.org/pdf/1409.0473.pdf)
