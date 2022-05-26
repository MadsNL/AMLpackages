from sklearn.preprocessing import StandardScaler #should scale by the same
from PIL import Image
import numpy as np

cols = ['Area (ABD)', 'Area (Filled)', 'Aspect Ratio', 'Biovolume (Cylinder)',
        'Biovolume (P. Spheroid)', 'Circle Fit',
        'Circularity', 'Circularity (Hu)', 'Compactness', 'Convex Perimeter',
        'Convexity', 'Diameter (ABD)', 'Diameter (ESD)', 'Edge Gradient',
        'Elongation', 'Feret Angle Max', 'Feret Angle Min', 'Fiber Curl',
        'Fiber Straightness', 'Geodesic Aspect Ratio', 'Geodesic Length',
        'Geodesic Thickness', 'Intensity', 'Length', 'Particles Per Chain',
        'Perimeter', 'Roughness', 'Sigma Intensity', 'Sum Intensity',
        'Symmetry', 'Transparency', 'Volume (ABD)', 'Volume (ESD)', 'Width'] #go through the rest of the columns

types = ['camp', 'corylus', 'dust', 'grim', 'qrob', 'qsub', 'cont'] #which are pollen?

image_size = 128


class not_training_set():
    def __init__(self, df, location_for_folder):
        self.df = df
        self.X_features_names = cols
        self.imgpaths = df['imgpaths'].to_numpy()
        for i in range(len(self.imgpaths)):
            self.imgpaths[i] = location_for_folder + \
                self.imgpaths[i].split('/')[-1]
        scaler = StandardScaler()
        self.X_features = scaler.fit_transform(df[cols])

    def __getitem__(self, idx):
        imgpath = self.imgpaths[idx]
        image = Image.open(imgpath)
        #image = image.convert('RGB')
        image = image.resize((image_size, image_size))  # 128 as in his
        # make gray_scale?
        image = image.convert('L')
        # A norm?
        image = np.array(image).astype('float')
        # for i in range(3):
        mi = np.min(image)
        ma = np.max(image)
        image -= mi
        image *= (1.0/(ma-mi))

        xfeatures = self.X_features[idx]

        return image, xfeatures


class training_set():
    def __init__(self, df, location_for_folder):
        self.df = df
        self.X_features_names = cols
        self.imgpaths = df['imgpaths'].to_numpy()
        type_list = df[types].values
        self.labels = np.argmax(type_list, axis=1)
        for i in range(len(self.imgpaths)):
            spl = self.imgpaths[i].split('/')
            self.imgpaths[i] = location_for_folder + spl[-2] + '/' + spl[-1]
        scaler = StandardScaler()
        self.X_features = scaler.fit_transform(df[cols])

    def __getitem__(self, idx):
        imgpath = self.imgpaths[idx]
        image = Image.open(imgpath)
        #image = image.convert('RGB')
        image = image.resize((image_size, image_size))
        # make gray_scale?
        image = image.convert('L')
        # A norm?
        image = np.array(image).astype('float')
        # for i in range(3):
        mi = np.min(image)
        ma = np.max(image)
        image -= mi
        image *= (1.0/(ma-mi))

        label = self.labels[idx]

        xfeatures = self.X_features[idx]

        return image, label, xfeatures
