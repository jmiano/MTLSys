import torchvision
import torchvision
from torch import nn


class MTLClassifier(nn.Module):
    def __init__(self, num_channels=3, num_genders=2, num_ethnicities=5):
        super(MTLClassifier, self).__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        # Encoding #
        # Load a pre-trained encoder
        self.encoder = torchvision.models.resnet18(pretrained=True)

        # Task 1: Age Regression
        self.class2_age = nn.Sequential(nn.Linear(in_features=1000, out_features=600),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class1_age = nn.Sequential(nn.Linear(in_features=600, out_features=300),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class0_age = nn.Linear(in_features=300, out_features=1)
        
        # Task 2: Gender Classification
        self.class2_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=600),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class1_gender = nn.Sequential(nn.Linear(in_features=600, out_features=300),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class0_gender = nn.Linear(in_features=300, out_features=num_genders)
        
        
        # Task 3: Ethnicity Classification
        self.class2_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=600),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class1_ethnicity = nn.Sequential(nn.Linear(in_features=600, out_features=300),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class0_ethnicity = nn.Linear(in_features=300, out_features=num_ethnicities)
        
        
    def forward(self, X):
        # Encoding
        
        encoding = self.encoder(X)
        
        # Task 1: Age Regression
        age_output = self.class2_age(encoding)
        age_output = self.class1_age(age_output)
        age_output = self.class0_age(age_output)
        
        # Task 2: Gender Classification
        gender_output = self.class2_gender(encoding)
        gender_output = self.class1_gender(gender_output)
        gender_output = self.class0_gender(gender_output)
        
        # Task 3: Ethnicity Classification
        ethnicity_output = self.class2_ethnicity(encoding)
        ethnicity_output = self.class1_ethnicity(ethnicity_output)
        ethnicity_output = self.class0_ethnicity(ethnicity_output)
        
        return age_output, gender_output, ethnicity_output



class AgeRegressor(nn.Module):
    def __init__(self, num_channels=3):
        super(AgeRegressor, self).__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        # Encoding #
        # Load a pre-trained encoder
        self.encoder = torchvision.models.resnet18(pretrained=True)

        # Task 1: Age Regression
        self.class2_age = nn.Sequential(nn.Linear(in_features=1000, out_features=600),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class1_age = nn.Sequential(nn.Linear(in_features=600, out_features=300),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class0_age = nn.Linear(in_features=300, out_features=1)
        
        
    def forward(self, X):
        # Encoding
        
        encoding = self.encoder(X)
        
        # Task 1: Age Regression
        age_output = self.class2_age(encoding)
        age_output = self.class1_age(age_output)
        age_output = self.class0_age(age_output)
        
        return age_output



class GenderClassifier(nn.Module):
    def __init__(self, num_channels=3, num_genders=2,):
        super(GenderClassifier, self).__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        # Encoding #
        # Load a pre-trained encoder
        self.encoder = torchvision.models.resnet18(pretrained=True)

        # Task 2: Gender Classification
        self.class2_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=600),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class1_gender = nn.Sequential(nn.Linear(in_features=600, out_features=300),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class0_gender = nn.Linear(in_features=300, out_features=num_genders)
        
        
    def forward(self, X):
        # Encoding
        
        encoding = self.encoder(X)
        
        # Task 2: Gender Classification
        gender_output = self.class2_gender(encoding)
        gender_output = self.class1_gender(gender_output)
        gender_output = self.class0_gender(gender_output)
        
        return gender_output


class EthnicityClassifier(nn.Module):
    def __init__(self, num_channels=3, num_ethnicities=5):
        super(EthnicityClassifier, self).__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        # Encoding #
        # Load a pre-trained encoder
        self.encoder = torchvision.models.resnet18(pretrained=True)

        # Task 3: Ethnicity Classification
        self.class2_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=600),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class1_ethnicity = nn.Sequential(nn.Linear(in_features=600, out_features=300),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class0_ethnicity = nn.Linear(in_features=300, out_features=num_ethnicities)
        
        
    def forward(self, X):
        # Encoding
        
        encoding = self.encoder(X)

        # Task 3: Ethnicity Classification
        ethnicity_output = self.class2_ethnicity(encoding)
        ethnicity_output = self.class1_ethnicity(ethnicity_output)
        ethnicity_output = self.class0_ethnicity(ethnicity_output)
        
        return ethnicity_output