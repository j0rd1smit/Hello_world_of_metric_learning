# Hello World of Deep Metric Learning: Siamese Contrastive loss

Deep metric learning is a fascinating deep learning technique. This technique aims a learn an embedding mapping that places similar items as close as possible, while it maps dissimilar item as far away from each other as possible in the embedding space, allowing us to solve tough challenges such as the Google Landmark and Humpback Whale Identification. However, getting started with technique can immensely challenging due to the lack of materials and the vast amount of new concepts. In this blog post, I will show you the Hello World version of Deep metric learning, allowing you to start your incredible journey into the world of metric learning.

### The goal
As is the tradition in ML, we will use the MNIST dataset for this hello world example. Normally our goal would be to create a classifier that given a 28x28 handwritten digit, can predict whether it is a 0, 1, ... or 9. However, in metric learning, we want to create a function that maps the images an embedding spaces that clusters all the images with the same class label together, while keeping the clusters separable as possible.



### Loss function
Let's get started with the loss function. We will be using the Contrastive Loss function:
$$ L = [d_{pos}]_+ + [m - d_{neg}]_+ = max(0, d_{pos}) + max(0, m - d_{neg}) $$

In this function, $d_{pos}$ is the distance between similar instances, $d_{neg}$ is the distance between dissimilar cases and $m$ is the margin. The first part of the formula measure how far apart all the similar examples are. The second part measures how far apart all the dissimilar pairs are.

Now the first thing we have to is translate this bit of math in usable pytorch code. We can to that as follows:
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
As the title suggested, we will be using a Siamese Network. This part is rather straight forward. First, we create a simple MNIST encoding backbone. This encoder should map the 28x28 image to a 2D feature.  I created the model following model:
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
We then create the training loop as follows. 
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
I create the training loop in PyTorch Lightning, but this also works in a standard PyTorch training loop.

### Loading the data
In the training loop, we need the following batch structure:
- `X = (input1, input2)` is a batches of MNIST image tuples.
- `Y = (targets, labels)` is a batch of targets (is it a tuple of similar or dissimilar images) and labels are the digit label, which we will use for visualization purposes.

We can create a Siamese version of the MNIST dataset using the following wrapper:
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

### Metric and visualization
To see how well the does during training we also need some metric and visualizations. So during training we gather all the embedings. Then at the of the epoch we create a scatter plot tensor board to so how the model progress during training. We can also calculate the estimate the predictive power of the embedding using a K-NN clasifier.


### Results

