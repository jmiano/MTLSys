import torchvision
import torchvision
from torch import nn


class MTL1Layer(nn.Module):
    def __init__(self, num_channels=3, num_genders=2, num_ethnicities=5):
        super(MTL1Layer, self).__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        # Encoding #
        # Load a pre-trained encoder
        self.encoder = torchvision.models.resnet18(pretrained=True)

        # Task 1: Age Regression
        self.class0_age = nn.Linear(in_features=1000, out_features=1)
        
        # Task 2: Gender Classification
        self.class0_gender = nn.Linear(in_features=1000, out_features=num_genders)
        
        
        # Task 3: Ethnicity Classification
        self.class0_ethnicity = nn.Linear(in_features=1000, out_features=num_ethnicities)
        
        
    def forward(self, X):
        # Encoding
        
        encoding = self.encoder(X)
        
        # Task 1: Age Regression
        age_output = self.class0_age(encoding)
        
        # Task 2: Gender Classification
        gender_output = self.class0_gender(encoding)
        
        # Task 3: Ethnicity Classification
        ethnicity_output = self.class0_ethnicity(encoding)
        
        return age_output, gender_output, ethnicity_output



class MTL3Layer(nn.Module):
    def __init__(self, num_channels=3, num_genders=2, num_ethnicities=5):
        super(MTL3Layer, self).__init__()
        
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



class MTL5Layer(nn.Module):
    def __init__(self, num_channels=3, num_genders=2, num_ethnicities=5):
        super(MTL5Layer, self).__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        # Encoding #
        # Load a pre-trained encoder
        self.encoder = torchvision.models.resnet18(pretrained=True)

        # Task 1: Age Regression
        self.class4_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class3_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class2_age = nn.Sequential(nn.Linear(in_features=1000, out_features=600),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class1_age = nn.Sequential(nn.Linear(in_features=600, out_features=300),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class0_age = nn.Linear(in_features=300, out_features=1)
        

        # Task 2: Gender Classification
        self.class4_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class3_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class2_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=600),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class1_gender = nn.Sequential(nn.Linear(in_features=600, out_features=300),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class0_gender = nn.Linear(in_features=300, out_features=num_genders)
        
        
        # Task 3: Ethnicity Classification
        self.class4_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class3_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
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
        age_output =  self.class4_age(encoding)
        age_output =  self.class3_age(age_output)
        age_output = self.class2_age(age_output)
        age_output = self.class1_age(age_output)
        age_output = self.class0_age(age_output)
        
        # Task 2: Gender Classification
        gender_output =  self.class4_gender(encoding)
        gender_output =  self.class3_gender(gender_output)
        gender_output = self.class2_gender(gender_output)
        gender_output = self.class1_gender(gender_output)
        gender_output = self.class0_gender(gender_output)
        
        # Task 3: Ethnicity Classification
        ethnicity_output =  self.class4_ethnicity(encoding)
        ethnicity_output =  self.class3_ethnicity(ethnicity_output)
        ethnicity_output = self.class2_ethnicity(ethnicity_output)
        ethnicity_output = self.class1_ethnicity(ethnicity_output)
        ethnicity_output = self.class0_ethnicity(ethnicity_output)
        
        return age_output, gender_output, ethnicity_output



class MTL7Layer(nn.Module):
    def __init__(self, num_channels=3, num_genders=2, num_ethnicities=5):
        super(MTL7Layer, self).__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        # Encoding #
        # Load a pre-trained encoder
        self.encoder = torchvision.models.resnet18(pretrained=True)

        # Task 1: Age Regression
        self.class6_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class5_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class4_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class3_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class2_age = nn.Sequential(nn.Linear(in_features=1000, out_features=600),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class1_age = nn.Sequential(nn.Linear(in_features=600, out_features=300),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class0_age = nn.Linear(in_features=300, out_features=1)
        

        # Task 2: Gender Classification
        self.class6_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class5_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class4_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class3_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class2_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=600),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class1_gender = nn.Sequential(nn.Linear(in_features=600, out_features=300),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class0_gender = nn.Linear(in_features=300, out_features=num_genders)
        
        
        # Task 3: Ethnicity Classification
        self.class6_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class5_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class4_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class3_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
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
        age_output =  self.class6_age(encoding)
        age_output =  self.class5_age(age_output)
        age_output = self.class4_age(age_output)
        age_output =  self.class3_age(age_output)
        age_output = self.class2_age(age_output)
        age_output = self.class1_age(age_output)
        age_output = self.class0_age(age_output)
        
        # Task 2: Gender Classification
        gender_output =  self.class6_gender(encoding)
        gender_output =  self.class5_gender(gender_output)
        gender_output = self.class4_gender(gender_output)
        gender_output =  self.class3_gender(gender_output)
        gender_output = self.class2_gender(gender_output)
        gender_output = self.class1_gender(gender_output)
        gender_output = self.class0_gender(gender_output)
        
        # Task 3: Ethnicity Classification
        ethnicity_output =  self.class6_ethnicity(encoding)
        ethnicity_output =  self.class5_ethnicity(ethnicity_output)
        ethnicity_output = self.class4_ethnicity(ethnicity_output)
        ethnicity_output =  self.class3_ethnicity(ethnicity_output)
        ethnicity_output = self.class2_ethnicity(ethnicity_output)
        ethnicity_output = self.class1_ethnicity(ethnicity_output)
        ethnicity_output = self.class0_ethnicity(ethnicity_output)
        
        return age_output, gender_output, ethnicity_output



class MTL9Layer(nn.Module):
    def __init__(self, num_channels=3, num_genders=2, num_ethnicities=5):
        super(MTL9Layer, self).__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        # Encoding #
        # Load a pre-trained encoder
        self.encoder = torchvision.models.resnet18(pretrained=True)

        # Task 1: Age Regression
        self.class8_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class7_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class6_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class5_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class4_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class3_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class2_age = nn.Sequential(nn.Linear(in_features=1000, out_features=600),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class1_age = nn.Sequential(nn.Linear(in_features=600, out_features=300),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class0_age = nn.Linear(in_features=300, out_features=1)
        

        # Task 2: Gender Classification
        self.class8_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class7_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class6_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class5_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class4_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class3_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class2_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=600),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class1_gender = nn.Sequential(nn.Linear(in_features=600, out_features=300),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class0_gender = nn.Linear(in_features=300, out_features=num_genders)
        
        
        # Task 3: Ethnicity Classification
        self.class8_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class7_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class6_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class5_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class4_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class3_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
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
        age_output =  self.class8_age(encoding)
        age_output =  self.class7_age(age_output)
        age_output = self.class6_age(age_output)
        age_output =  self.class5_age(age_output)
        age_output = self.class4_age(age_output)
        age_output =  self.class3_age(age_output)
        age_output = self.class2_age(age_output)
        age_output = self.class1_age(age_output)
        age_output = self.class0_age(age_output)
        
        # Task 2: Gender Classification
        gender_output =  self.class8_gender(encoding)
        gender_output =  self.class7_gender(gender_output)
        gender_output = self.class6_gender(gender_output)
        gender_output =  self.class5_gender(gender_output)
        gender_output = self.class4_gender(gender_output)
        gender_output =  self.class3_gender(gender_output)
        gender_output = self.class2_gender(gender_output)
        gender_output = self.class1_gender(gender_output)
        gender_output = self.class0_gender(gender_output)
        
        # Task 3: Ethnicity Classification
        ethnicity_output =  self.class8_ethnicity(encoding)
        ethnicity_output =  self.class7_ethnicity(ethnicity_output)
        ethnicity_output = self.class6_ethnicity(ethnicity_output)
        ethnicity_output =  self.class5_ethnicity(ethnicity_output)
        ethnicity_output = self.class4_ethnicity(ethnicity_output)
        ethnicity_output =  self.class3_ethnicity(ethnicity_output)
        ethnicity_output = self.class2_ethnicity(ethnicity_output)
        ethnicity_output = self.class1_ethnicity(ethnicity_output)
        ethnicity_output = self.class0_ethnicity(ethnicity_output)
        
        return age_output, gender_output, ethnicity_output




class MTL11Layer(nn.Module):
    def __init__(self, num_channels=3, num_genders=2, num_ethnicities=5):
        super(MTL11Layer, self).__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        # Encoding #
        # Load a pre-trained encoder
        self.encoder = torchvision.models.resnet18(pretrained=True)

        # Task 1: Age Regression
        self.class10_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class9_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class8_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class7_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class6_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class5_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class4_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class3_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class2_age = nn.Sequential(nn.Linear(in_features=1000, out_features=600),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class1_age = nn.Sequential(nn.Linear(in_features=600, out_features=300),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class0_age = nn.Linear(in_features=300, out_features=1)
        

        # Task 2: Gender Classification
        self.class10_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class9_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class8_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class7_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class6_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class5_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class4_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class3_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class2_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=600),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class1_gender = nn.Sequential(nn.Linear(in_features=600, out_features=300),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class0_gender = nn.Linear(in_features=300, out_features=num_genders)
        
        
        # Task 3: Ethnicity Classification
        self.class10_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class9_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class8_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class7_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class6_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class5_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class4_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class3_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
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
        age_output =  self.class10_age(encoding)
        age_output =  self.class9_age(age_output)
        age_output = self.class8_age(age_output)
        age_output =  self.class7_age(age_output)
        age_output = self.class6_age(age_output)
        age_output =  self.class5_age(age_output)
        age_output = self.class4_age(age_output)
        age_output =  self.class3_age(age_output)
        age_output = self.class2_age(age_output)
        age_output = self.class1_age(age_output)
        age_output = self.class0_age(age_output)
        
        # Task 2: Gender Classification
        gender_output =  self.class10_gender(encoding)
        gender_output =  self.class9_gender(gender_output)
        gender_output = self.class8_gender(gender_output)
        gender_output =  self.class7_gender(gender_output)
        gender_output = self.class6_gender(gender_output)
        gender_output =  self.class5_gender(gender_output)
        gender_output = self.class4_gender(gender_output)
        gender_output =  self.class3_gender(gender_output)
        gender_output = self.class2_gender(gender_output)
        gender_output = self.class1_gender(gender_output)
        gender_output = self.class0_gender(gender_output)
        
        # Task 3: Ethnicity Classification
        ethnicity_output =  self.class10_ethnicity(encoding)
        ethnicity_output =  self.class9_ethnicity(ethnicity_output)
        ethnicity_output = self.class8_ethnicity(ethnicity_output)
        ethnicity_output =  self.class7_ethnicity(ethnicity_output)
        ethnicity_output = self.class6_ethnicity(ethnicity_output)
        ethnicity_output =  self.class5_ethnicity(ethnicity_output)
        ethnicity_output = self.class4_ethnicity(ethnicity_output)
        ethnicity_output =  self.class3_ethnicity(ethnicity_output)
        ethnicity_output = self.class2_ethnicity(ethnicity_output)
        ethnicity_output = self.class1_ethnicity(ethnicity_output)
        ethnicity_output = self.class0_ethnicity(ethnicity_output)
        
        return age_output, gender_output, ethnicity_output


class MTL13Layer(nn.Module):
    def __init__(self, num_channels=3, num_genders=2, num_ethnicities=5):
        super(MTL13Layer, self).__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        # Encoding #
        # Load a pre-trained encoder
        self.encoder = torchvision.models.resnet18(pretrained=True)

        # Task 1: Age Regression
        self.class12_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class11_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class10_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class9_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class8_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class7_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class6_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class5_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class4_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class3_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class2_age = nn.Sequential(nn.Linear(in_features=1000, out_features=600),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class1_age = nn.Sequential(nn.Linear(in_features=600, out_features=300),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class0_age = nn.Linear(in_features=300, out_features=1)
        

        # Task 2: Gender Classification
        self.class12_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class11_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class10_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class9_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class8_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class7_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class6_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class5_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class4_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class3_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class2_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=600),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class1_gender = nn.Sequential(nn.Linear(in_features=600, out_features=300),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class0_gender = nn.Linear(in_features=300, out_features=num_genders)
        
        
        # Task 3: Ethnicity Classification
        self.class12_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class11_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class10_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class9_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class8_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class7_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class6_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class5_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class4_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class3_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
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
        age_output =  self.class12_age(encoding)
        age_output =  self.class11_age(age_output)
        age_output = self.class10_age(age_output)
        age_output =  self.class9_age(age_output)
        age_output = self.class8_age(age_output)
        age_output =  self.class7_age(age_output)
        age_output = self.class6_age(age_output)
        age_output =  self.class5_age(age_output)
        age_output = self.class4_age(age_output)
        age_output =  self.class3_age(age_output)
        age_output = self.class2_age(age_output)
        age_output = self.class1_age(age_output)
        age_output = self.class0_age(age_output)
        
        # Task 2: Gender Classification
        gender_output =  self.class12_gender(encoding)
        gender_output =  self.class11_gender(gender_output)
        gender_output = self.class10_gender(gender_output)
        gender_output =  self.class9_gender(gender_output)
        gender_output = self.class8_gender(gender_output)
        gender_output =  self.class7_gender(gender_output)
        gender_output = self.class6_gender(gender_output)
        gender_output =  self.class5_gender(gender_output)
        gender_output = self.class4_gender(gender_output)
        gender_output =  self.class3_gender(gender_output)
        gender_output = self.class2_gender(gender_output)
        gender_output = self.class1_gender(gender_output)
        gender_output = self.class0_gender(gender_output)
        
        # Task 3: Ethnicity Classification
        ethnicity_output =  self.class12_ethnicity(encoding)
        ethnicity_output =  self.class11_ethnicity(ethnicity_output)
        ethnicity_output = self.class10_ethnicity(ethnicity_output)
        ethnicity_output =  self.class9_ethnicity(ethnicity_output)
        ethnicity_output = self.class8_ethnicity(ethnicity_output)
        ethnicity_output =  self.class7_ethnicity(ethnicity_output)
        ethnicity_output = self.class6_ethnicity(ethnicity_output)
        ethnicity_output =  self.class5_ethnicity(ethnicity_output)
        ethnicity_output = self.class4_ethnicity(ethnicity_output)
        ethnicity_output =  self.class3_ethnicity(ethnicity_output)
        ethnicity_output = self.class2_ethnicity(ethnicity_output)
        ethnicity_output = self.class1_ethnicity(ethnicity_output)
        ethnicity_output = self.class0_ethnicity(ethnicity_output)
        
        return age_output, gender_output, ethnicity_output



class MTL15Layer(nn.Module):
    def __init__(self, num_channels=3, num_genders=2, num_ethnicities=5):
        super(MTL15Layer, self).__init__()
        
        self.maxpool = nn.MaxPool2d(kernel_size=2)
        
        # Encoding #
        # Load a pre-trained encoder
        self.encoder = torchvision.models.resnet18(pretrained=True)

        # Task 1: Age Regression
        self.class14_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class13_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class12_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class11_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class10_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class9_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class8_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class7_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class6_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class5_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class4_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class3_age = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class2_age = nn.Sequential(nn.Linear(in_features=1000, out_features=600),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class1_age = nn.Sequential(nn.Linear(in_features=600, out_features=300),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class0_age = nn.Linear(in_features=300, out_features=1)
        

        # Task 2: Gender Classification
        self.class14_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class13_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class12_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class11_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class10_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class9_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class8_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class7_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class6_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class5_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class4_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class3_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class2_gender = nn.Sequential(nn.Linear(in_features=1000, out_features=600),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class1_gender = nn.Sequential(nn.Linear(in_features=600, out_features=300),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class0_gender = nn.Linear(in_features=300, out_features=num_genders)
        
        
        # Task 3: Ethnicity Classification
        self.class14_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class13_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class12_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class11_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class10_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class9_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class8_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class7_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class6_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class5_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class4_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
        self.class3_ethnicity = nn.Sequential(nn.Linear(in_features=1000, out_features=1000),
                                        nn.ReLU(),
                                        nn.Dropout(p=0.1))
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
        age_output =  self.class14_age(encoding)
        age_output =  self.class13_age(age_output)
        age_output = self.class12_age(age_output)
        age_output =  self.class11_age(age_output)
        age_output = self.class10_age(age_output)
        age_output =  self.class9_age(age_output)
        age_output = self.class8_age(age_output)
        age_output =  self.class7_age(age_output)
        age_output = self.class6_age(age_output)
        age_output =  self.class5_age(age_output)
        age_output = self.class4_age(age_output)
        age_output =  self.class3_age(age_output)
        age_output = self.class2_age(age_output)
        age_output = self.class1_age(age_output)
        age_output = self.class0_age(age_output)
        
        # Task 2: Gender Classification
        gender_output =  self.class14_gender(encoding)
        gender_output =  self.class13_gender(gender_output)
        gender_output = self.class12_gender(gender_output)
        gender_output =  self.class11_gender(gender_output)
        gender_output = self.class10_gender(gender_output)
        gender_output =  self.class9_gender(gender_output)
        gender_output = self.class8_gender(gender_output)
        gender_output =  self.class7_gender(gender_output)
        gender_output = self.class6_gender(gender_output)
        gender_output =  self.class5_gender(gender_output)
        gender_output = self.class4_gender(gender_output)
        gender_output =  self.class3_gender(gender_output)
        gender_output = self.class2_gender(gender_output)
        gender_output = self.class1_gender(gender_output)
        gender_output = self.class0_gender(gender_output)
        
        # Task 3: Ethnicity Classification
        ethnicity_output =  self.class14_ethnicity(encoding)
        ethnicity_output =  self.class13_ethnicity(ethnicity_output)
        ethnicity_output = self.class12_ethnicity(ethnicity_output)
        ethnicity_output =  self.class11_ethnicity(ethnicity_output)
        ethnicity_output = self.class10_ethnicity(ethnicity_output)
        ethnicity_output =  self.class9_ethnicity(ethnicity_output)
        ethnicity_output = self.class8_ethnicity(ethnicity_output)
        ethnicity_output =  self.class7_ethnicity(ethnicity_output)
        ethnicity_output = self.class6_ethnicity(ethnicity_output)
        ethnicity_output =  self.class5_ethnicity(ethnicity_output)
        ethnicity_output = self.class4_ethnicity(ethnicity_output)
        ethnicity_output =  self.class3_ethnicity(ethnicity_output)
        ethnicity_output = self.class2_ethnicity(ethnicity_output)
        ethnicity_output = self.class1_ethnicity(ethnicity_output)
        ethnicity_output = self.class0_ethnicity(ethnicity_output)
        
        return age_output, gender_output, ethnicity_output