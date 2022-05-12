# Simulation MNIST-based causal effect dataset + Dragonnet inspired Neural Network trainers

To run the code:

`python ./experiment/main.py --encoder '<encoder-name>' --treatment 'odd-even'`

For the argument `encoder`,  the following are options:

`encoder-name: 'resnet', 'vit', or 'fc'`

`resnet` is for ResNet50 as representation learner (encoder model)
`vit` is for Vision Transformer model as representation learner (encoder model)
`fc` is for Dragonnet based model where feed-forward layers are representation learner (encoder model).