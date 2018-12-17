from PIL import Image
import PIL
from PIL import ImageFilter
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import random
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
import collections
from pylab import *
from PIL import *


class ImageClassifier(object):
    def __init__(self, path_to_image):
        self.training_dir = os.path.join(os.getcwd(), 'training')
        self.prediction_dir = os.path.join(os.getcwd(), 'predictions')
        self.image_path = path_to_image
        self.original_image = Image.open(path_to_image).convert('L')
        self.random_sample_dir = os.path.join(os.getcwd(), 'random_sample')
        self.selected_directory = os.path.join(os.getcwd(), 'selected')
        self.features = None
        self.training_features = None
        self.training_labels = None
        self.logistic_model = None
        self.svm_model = None
        self.kmean_cluster = None
        self.processed_grid = None
        self.distinct_labels = None

    def show_image(self, path=None):
        if path is None:
            plt.axis('off')
            plt.imshow(self.original_image)
            plt.tight_layout()
            plt.show()

    def find_clusters(self, image=None, clusters=5):
        if image is None:
            kmeans = KMeans(n_clusters=clusters, n_jobs=-1)
            kmeans.fit(np.array(self.original_image).flatten().reshape(-1, 1))
            self.kmean_cluster = kmeans
        else:
            if isinstance(image, collections.Iterable):
                if type(image) is dict:
                    items = image.keys()
                    obs = list()
                    for item in items:
                        obs += [np.array(image[item]['image']).flatten()]
                    kmeans = KMeans(n_clusters=clusters, n_jobs=-1)
                    kmeans.fit(np.array(obs))
                    self.kmean_cluster = kmeans
                    for i in items:
                        label = kmeans.predict([np.array(image[i]['image']).flatten()])[0]
                        path = check_make_dir(os.path.join(self.prediction_dir, str(label)))
                        image[i]['image'].save(os.path.join(path, str(image[i]['ux']) + '_' + str(image[i]['uy']) + '.tiff'))
                        image[i]['label'] = label
                    self.processed_grid = image
                    self.distinct_labels = clusters - 1
            elif type(image) is PIL.Image.Image:
                kmeans = KMeans(n_clusters=clusters, n_jobs=-1)
                kmeans.fit(np.array(self.original_image).flatten().reshape(-1, 1))
                self.kmean_cluster = kmeans

    def generate_random_sample(self, size=(40, 40)):
        x, y = self.original_image.size
        x -= size[0]
        y -= size[1]
        upper_left = (random.randint(0, x), random.randint(0, y))
        area = (upper_left[0], upper_left[1], upper_left[0] + size[0], upper_left[1] + size[1])
        new_img = self.original_image.crop(area)
        new_img.save(os.path.join(self.random_sample_dir, str(random.randint(0, 1000)) + '.tiff'))

    def generate_image(self, ux, uy, lx, ly, label):
        new_img = self.original_image.crop((ux, uy, lx, ly))
        file_dir = self.selected_directory
        initial = 0
        nfile = os.path.join(file_dir, label + '_{}.tiff'.format(str(initial)))
        while os.path.exists(nfile):
            initial += 1
            nfile = os.path.join(file_dir, label + '_{}.tiff'.format(str(initial)))
        new_img.save(nfile)

    def generate_image_grid(self, input_x, input_y):
        image_grid = dict()
        x, y = self.original_image.size
        cols = x / input_x
        rows = y / input_y
        count = 0
        for i in range(cols):
            for j in range(rows):
                image_grid[count] = dict()
                image_grid[count]['row'] = i
                image_grid[count]['column'] = j
                image_grid[count]['ux'] = (i * input_x)
                image_grid[count]['uy'] = (j * input_y)
                image_grid[count]['lx'] = (i * input_x) + input_x
                image_grid[count]['ly'] = (j * input_y) + input_y
                image_grid[count]['area'] = ((i * input_x), (j * input_y),
                                             (i * input_x) + input_x, (j * input_y) + input_y)
                image_grid[count]['image'] = self.original_image.crop(image_grid[count]['area'])
                count += 1
        return image_grid

    def read_training_features(self):
        imgs = dict()
        files = os.listdir(self.selected_directory)
        for i in range(len(files)):
            imgs[i] = dict()
            imgs[i]['label'] = files[i].split('_')[0]
            imgs[i]['image'] = Image.open(os.path.join(self.selected_directory, files[i])).convert('L')
            imgs[i]['vals'] = [[np.array(imgs[i]['image']).flatten().mean(),
                                                  np.array(imgs[i]['image']).flatten().std(),
                                                  imgs[i]['image'].size[0],
                                                  imgs[i]['image'].size[1]],
                                        imgs[i]['label']]
        self.features = np.array(imgs)
        self.training_features = np.array([imgs[i]['vals'][0] for i in imgs.keys()])
        self.training_labels = np.array([imgs[i]['vals'] for i in imgs.keys()])[:, 1].reshape(-1, 1)

    def log_fit_training_data(self):
        self.read_training_features()
        model = LogisticRegressionCV()
        model.fit(self.training_features, self.training_labels)
        self.logistic_model = model
        for i in range(len(self.training_features)):
            print(self.training_labels[i], self.logistic_model.predict(
                self.training_features[i].reshape(1, -1)))

    def svc_fit_training_date(self):
        self.read_training_features()
        model = SVC()
        model.fit(self.training_features, self.training_labels)
        self.svm_model = model
        for i in range(len(self.training_features)):
            print(self.training_labels[i], self.svm_model.predict(self.training_features[i].reshape(1, -1)))

    def create_pixel_grid(self):
        grids = self.processed_grid
        pixel = self.original_image.crop((0, 0, self.original_image.size[0], self.original_image.size[1])).load()
        color_range = self.distinct_labels
        for i in grids:

            xi = grids[i]['ux']
            yi = grids[i]['uy']
            x_end = grids[i]['lx']
            y_end = grids[i]['ly']
            for x in range(xi, x_end + 1):
                for y in range(yi, y_end + 1):
                    pixel[x, y] = grids[i]['label'] * (256/color_range)

        result = Image.fromarray(pixel)
        result.save('result.png')


def check_make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
    return path


image_path = os.path.join(os.getcwd(), 'images')
images = os.listdir(image_path)


def get_image(ind):
    return os.path.join(image_path, images[ind])


def get_training_image(directory, ind):
    class_dir = os.path.join(os.path.join(os.path.dirname(__file__), 'training_images'), directory)
    files = os.listdir(class_dir)
    return os.path.join(class_dir, files[ind])


def get_classifications():
    training_dir = os.path.join(os.path.dirname(__file__), 'training_images')
    initial_classes = os.listdir(training_dir)
    return [item for item in initial_classes if len(os.listdir(os.path.join(training_dir, item))) > 0]


def get_stats_of_image(filepath):
    image = Image.open(filepath)
    im = image.convert('L')
    image_array = array(im).flatten()
    med = np.median(image_array)
    mean = np.mean(image_array)
    std = np.std(image_array)
    return dict(fp=filepath, mean=mean, median=med, std=std)


def image_to_array(filepath):
    image = Image.open(filepath)
    im = image.convert('L')
    return array(im)


def read_training_data():
    training_dir = os.path.join(os.path.dirname(__file__), 'training_images')
    classes = get_classifications()
    metrics = []
    for cls in classes:
        curpath = os.path.join(training_dir, cls)
        files = os.listdir(curpath)
        for file in files:
            file_stat = get_stats_of_image(os.path.join(curpath, file))
            file_stat['class'] = cls
            metrics.append(file_stat)
    return metrics


def test(ind):
    image = Image.open(get_image(ind))
    im = image.convert('L')
    p = image.convert("L").filter(ImageFilter.GaussianBlur(radius=1))
    p.show()
