# CharizardNet
Learning high-dimensional causal effect.

In this repository, we propose simulation of causal effect and study of deep learning models on the simulated dataset.

This work is inspired by Claudia Shi et. al. https://arxiv.org/abs/1906.02120; https://github.com/claudiashi57/dragonnet. We adapted a few parts of their code.

For the Vision Transformer class, we refer code presented here: https://keras.io/examples/vision/image_classification_with_vision_transformer/


## Simulation MNIST-based causal effect dataset + Dragonnet inspired Neural Network trainers

To run the code:

`python code/main.py --encoder '<encoder-name>' --treatment 'odd-even'`

For the argument `encoder`,  the following are options:

`encoder-name: 'resnet', 'vit', or 'fc'`

`resnet` is for ResNet50 as representation learner (encoder model)
`vit` is for Vision Transformer model as representation learner (encoder model)
`fc` is for Dragonnet based model where feed-forward layers are representation learner (encoder model).