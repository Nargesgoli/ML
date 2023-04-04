CIFAR-10 - revisited


The provided data contain 10K images from 5 classes ['plane','car','bird','horse','ship'] extracted from CIFAR-10 dataset. 

Three files are provided:

 - X: Contains the 128 features extracted from all the provided 10K images. (using a CNN)

 - y: Contains the true labels for each of the 10K images (one label by row): [0:'plane',1:'car',2:'bird',3:'horse',4:'ship'] 

 - images: contain the original images. Each row contains 32*32*3 = 3072 columns corresponding to the scaled pixels intensity (1./255) acros the three RGB channels.
 PS: In order to visualize the image corresponding to a given rowm you should first revert it to its original shape (32,32,3).
 Example: To visualize the first image (row 0), you can use: plt.imshow(images[0].reshape(32,32,3))



NB: the index order is respected in each of the three files: X[0] contain the the features extracted from images[0] whose label is y[0]
