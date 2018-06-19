## MovieLens recommandation model.
MovieLens is a dataset of 20M ratings of more than 27 000 movies by more than 138 000 users, more infos [here](https://grouplens.org/datasets/movielens/).

State of the art models uses:
* [matrix factorization](https://docs.treasuredata.com/articles/hivemall-movielens20m-fm) with RMSE of 0.80,
* [autoencoders](https://arxiv.org/pdf/1606.07659.pdf) with RMSE of 0.81.

We build a deep learning model using Entity Embeddings for Categorical Variables, from [this paper](https://arxiv.org/abs/1604.06737), that achieves an **RMSE of 0.81**, on par with state of the art models.
The neural network is implement in Keras with TensorFlow backend. The code is in the "movienet.py" file, and the training is in the training notebook.

A big plus of entity embedding is that during the training we compute an embedding space of movies and users. 
Hence, we have differents methods to recommand movies to an user:
1. We evaluate the network and suggest the most rated movie. However with an RMSE of 0.81, each predicted rating as an average error of 0.8 star.
2. For a movie, we look at the closest neighbourds in the embedding space. In this case we are using an KNN index with [nmslib](https://github.com/nmslib/nmslib). With enough dimensions we hope that the embeddings catch good relationhips between the movies. This approach give good results without any other infos than the movies ratings.
3. For a user, we look at the closest neighbourds in the embedding space, and for each neighbourd we recommand they most rated movie. This approach give average results, we need more infos on the user charateristic to have a more accurate embedding space.


