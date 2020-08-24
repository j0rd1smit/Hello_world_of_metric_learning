

# Hello World of Deep Metric Learning: Siamese Contrastive loss

Deep metric learning is a fascinating deep learning technique. This technique aims a learn an embedding mapping that places similar items as close as possible, while it maps dissimilar item as far away from each other as possible in the embedding space. Using this technique, we can solve tough challenges such as the Google Landmark and Humpback Whale Identification Kaggle challenge. However, getting started with this method can immensely challenging due to the lack of materials and the vast amount of new concepts. In this blog post, I will show you the Hello World version of Deep metric learning, allowing you to start your incredible journey into the world of metric learning.

![Training set](https://raw.githubusercontent.com/j0rd1smit/Hello_world_of_metric_learning/master/images/training_set.gif)


### The goal
As is the tradition in ML, we will use the MNIST dataset for this hello world example. Normally our goal would be to create a classifier that given a 28x28 handwritten digit, can predict whether it is a 0, 1, ... or 9. However, in metric learning, we want to create a function that maps the images an embedding spaces that clusters all the images with the same class label together, while keeping the clusters as separable as possible. 
At the end of this article, you should be able to map all the MNIST digits into a 2D space which should look like figure above. During this article, I will be using PyTorch Lightning, but the code will also work in native PyTorch. Feel free to follow along or just to run the example notebook in Colab. 



### Loss function
Let's get started with the loss function. We will be using the Contrastive Loss function:
$$ L = [d_{pos}]_+ + [m - d_{neg}]_+ = max(0, d_{pos}) + max(0, m - d_{neg}) $$

In this function, $d_{pos}$ is the distance between similar instance tuples, $d_{neg}$ is the distance between dissimilar instance tuples, and $m$ is the margin. The first part of the formula measures how far apart all the similar instances are, minimizing this part ensures that similar instances will be clustered to gather.
The second part measures how far apart all the dissimilar pairs are, minimizing this part ensures that the different cluster will be as separable as possible.

Now the first thing we have to do is translate this bit of math in usable PyTorch code. In this case we will be using the euclidean distance metric, which gives us the following loss function:
```
class ContrastiveLoss(nn.Module):  
    def __init__(self, margin, *, eps:float=1e-9):  
        super(ContrastiveLoss, self).__init__()  
        self.margin = margin  
        self.eps = eps  
  
    def forward(self, embedding1, embedding2, target):  
        positve_loss = target.float() * (embedding2 - embedding1).pow(2).sum(1)  
        negative_loss = (1 + -1 * target).float() * F.relu(self.margin - (embedding2 - embedding1).pow(2).sum(1) + self.eps).pow(2)  
  
        loss = positve_loss + negative_loss  
        return loss.mean()
        
```
In this function, the target label indicates whether the two embeddings should be similar (1) or dissimilar (0).


### The model
![Siamese Network](https://raw.githubusercontent.com/j0rd1smit/Hello_world_of_metric_learning/master/images/siamese.png)


As the title suggested, we will be using a Siamese Network. This part is rather straight forward. First, we create a simple MNIST encoding backbone. This encoder should map the 28x28 image to a 2D feature.  I created the model following:
```
class ConvBackbone(nn.Module):  
    def __init__(self):  
        super(ConvBackbone, self).__init__()  
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, 5), nn.PReLU(),  
                                     nn.MaxPool2d(2, stride=2),  
                                     nn.Conv2d(32, 64, 5), nn.PReLU(),  
                                     nn.MaxPool2d(2, stride=2))  
  
        self.fc = nn.Sequential(nn.Linear(64 * 4 * 4, 256),  
                                nn.PReLU(),  
                                nn.Linear(256, 256),  
                                nn.PReLU(),  
                                nn.Linear(256, 2)  
                                )  
  
      def forward(self, x):  
          output = self.convnet(x)  
          output = output.view(output.size()[0], -1)  
          output = self.fc(output)  
          return output
```
We then create the training step as follows. 
```
def training_step(self, batch, batch_idx):
  # Fetch the data  
    (input1, input2), (targets, labels) = batch  
    # Siamese embedding using any backbone architecture
    embedding1 = self(input1)  
    embedding2 = self(input2)
    # Calc the loss using the previously created loss funcion.  
    loss = self.loss_func(embedding1, embedding2, targets)  
  
    log = {"train_loss": loss}  
    return {"loss": loss, "log": log, "embeddings": embedding1, "labels": labels}
```


### Loading the data
In the training loop, we need the following batch structure:
- `X = (input1, input2)` is batches of MNIST image tuples.
- `Y = (targets, labels)` is a batch of targets (is it a tuple of similar or dissimilar images) and labels are the digit label, which we will use for visualization purposes.

We will be using the following dataset wrapper to transform MNIST into the required format:
```
class SiameseMNIST(Dataset):  
    def __init__(self, mnist_dataset):  
        self.mnist_dataset = mnist_dataset  
  
        self.train = self.mnist_dataset.train  
        self.transform = self.mnist_dataset.transform  
  
        self.labels = self.mnist_dataset.targets  
        self.data = self.mnist_dataset.data  
        self.labels_set = set(self.labels.numpy())  
        self.label_to_indices = {label: np.where(self.labels.numpy() == label)[0] for label in self.labels_set}  
  
        if not self.train:  
          # During validation, always pick the same tuple.
            np.random.seed(42)  
            self.val_data = [self._draw(i) for i, _ in enumerate(self.data)]  
  
  
    def __getitem__(self, index):  
        if self.train:
           # Randomly pick during training  
            image1, image2, target, label1 = self._draw(index)  
        else:  
          # During validation always pick the same tuple.
            image1, image2, target, label1 = self.val_data[index]  
  
        if self.transform is not None:  
            image1 = self.transform(image1)  
            image2 = self.transform(image2)  
  
        return (image1, image2), (target, label1)  
  
    def _draw(self, index):  
        image1 = self.data[index]  
        label1 = self.labels[index]  
  
        target = np.random.choice([0, 1])  
        if target == 1:  
            # Pick a random image with the same label as image1.
            siamese_index = np.random.choice(self.label_to_indices[int(label1)])  
        else:  
          # Pick a random label that is not the same as image1.
            siamese_label = np.random.choice(list(self.labels_set - {label1}))  
            # Pick a random image with the randomly chosen label.
            siamese_index = np.random.choice(self.label_to_indices[siamese_label])  
  
        image2 = self.data[siamese_index]  
    # Load the images
        image1 = Image.fromarray(image1.numpy(), mode='L')  
        image2 = Image.fromarray(image2.numpy(), mode='L')  
  
        return image1, image2, target, label1  
  
    def __len__(self):  
        return len(self.mnist_dataset)
```

### Metric 
We also want to see how well the model does during training. We can do this by measuring the accuracy of a K Nearest Neighbor (K-NN) classifier in the embedding space. We will do that as follows. First, we will gather all the embeddings during the training loop. Then, at the end of the training loop, we measure the accuracy of the K-NN classifier in the embedding space. If this accuracy is high, we know that the embeddings are separable and closely clustered. We can do this using the following code:
```
from sklearn.neighbors import KNeighborsClassifier

def knn_accuracy(embeddings, labels):
    embeddings = embeddings.detach().cpu()
    labels = labels.detach().cpu()

    return KNeighborsClassifier().fit(embeddings, labels).score(embeddings, labels)
```

### Visualization
In this toy example, we are creating a 2D embedding space. So we can directly visualize the embedding space. We have already gathered all the embedings in the previous step. So lets also plot these to visualise the embedings space using a scatter plot to see how it changes during the training process:
```
f create_embeddings_plot_image(embeddings, labels):
    colours = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

    embeddings = embeddings.detach().cpu()
    labels = labels.detach().cpu()
    for label in torch.unique(labels):
        color = colours[int(label) % len(colours)]
        idx_slice = labels == label
        plt.scatter(embeddings[idx_slice, 0], embeddings[idx_slice, 1], label=str(int(label)), c=color)

    plt.legend(loc='upper right')
    plt.grid()
    
    # Transform the plot to numpy array such that we can plot it Tensorboard
    buf = io.BytesIO()
    plt.savefig(buf, format='jpg')
    buf.seek(0)
    image = cv2.imdecode(np.frombuffer(buf.getvalue(), np.uint8), -1)
    image = image.transpose(2, 0, 1)
    
    plt.close()

    return image
```


### Results
The results look very good. The K-NN accuracy quickly approaches 100% for the training and validation set.
![K-NN accuracy](https://raw.githubusercontent.com/j0rd1smit/Hello_world_of_metric_learning/master/images/knn_accuracy.png)


The embedding space also looks very good. We get distinct clusters for each of the ten digits. We might want to train a bit longer than 20 epochs to make the more separable.
![Training set](https://raw.githubusercontent.com/j0rd1smit/Hello_world_of_metric_learning/master/images/training_set.gif)

The embedding also works for the validation set, so we are not overfitting.
![Validation set](https://raw.githubusercontent.com/j0rd1smit/Hello_world_of_metric_learning/master/images/validation.gif)

### Conclusion
In this article, you got your first practical introduction into Deep Metric Larning. You have learned how to use contrastive loss in a Siamese network setting to obtain adequate results. However, there is still room for improvement. For example, our tuple sample strategy might be a bit naive. In future articles, I will introduce you to a concept called mining which significantly improve the efficiency of this algorithm. 