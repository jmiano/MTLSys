import math
import torch


def train_mtl_model(num_epochs=10, model=None, optimizer=None,
                    train_loader=None, val_loader=None, 
                    age_criterion=None, gender_criterion=None, ethni_criterion=None,
                    age_coeff=None, gender_coeff=None, ethni_coeff=None):

    best_val_loss = float(math.inf)
    for epoch in range(num_epochs):
        avg_train_loss = 0
        tot_train_loss = 0
        tot_train_age_loss = 0
        tot_train_gender_loss = 0
        tot_train_ethni_loss = 0
        tot_train_samples = 0
        
        # Training Loop
        model.train()
        for i, (img, age, gender, ethnicity) in enumerate(train_loader):
            img = img.cuda()
            age = age.float().cuda()
            gender = gender.long().cuda()
            ethnicity = ethnicity.long().cuda()

            # Clear grad
            optimizer.zero_grad()

            # Get outputs
            age_output, gender_output, ethnicity_output = model(img)
            age_output = age_output.squeeze(1)
            gender_output = gender_output
            ethnicity_output = ethnicity_output

            # Calculate losses
            age_loss = age_criterion(age_output, age)
            gender_loss = gender_criterion(gender_output, gender)
            ethnicity_loss = ethni_criterion(ethnicity_output, ethnicity)
            multi_loss = age_coeff*age_loss + gender_coeff*gender_loss + ethni_coeff*ethnicity_loss
                
            tot_train_loss += multi_loss.item()
            tot_train_age_loss += age_coeff*age_loss.item()
            tot_train_gender_loss += gender_coeff*gender_loss.item()
            tot_train_ethni_loss += ethni_coeff*ethnicity_loss.item()
            tot_train_samples += img.shape[0]

            # Get grad
            multi_loss.backward()

            # Update model weights
            optimizer.step()
            
        avg_train_loss = tot_train_loss / tot_train_samples
        avg_train_age_loss = tot_train_age_loss / tot_train_samples
        avg_train_gender_loss = tot_train_gender_loss / tot_train_samples
        avg_train_ethni_loss = tot_train_ethni_loss / tot_train_samples


        avg_val_loss = 0
        tot_val_loss = 0
        tot_val_age_loss = 0
        tot_val_gender_loss = 0
        tot_val_ethni_loss = 0
        tot_val_samples = 0
        
        # Iterate through the val dataset
        model.eval()
        for i, (img, age, gender, ethnicity) in enumerate(val_loader):
            img = img.cuda()
            age = age.float().cuda()
            gender = gender.long().cuda()
            ethnicity = ethnicity.long().cuda()

            # Clear grad
            optimizer.zero_grad()

            # Get outputs
            age_output, gender_output, ethnicity_output = model(img)
            age_output = age_output.squeeze(1)
            gender_output = gender_output
            ethnicity_output = ethnicity_output

            # Calculate losses
            age_loss = age_criterion(age_output, age)
            gender_loss = gender_criterion(gender_output, gender)
            ethnicity_loss = ethni_criterion(ethnicity_output, ethnicity)
            multi_loss = age_coeff*age_loss + gender_coeff*gender_loss + ethni_coeff*ethnicity_loss
            
            tot_val_loss += multi_loss.item()
            tot_val_age_loss += age_coeff*age_loss.item()
            tot_val_gender_loss += gender_coeff*gender_loss.item()
            tot_val_ethni_loss += ethni_coeff*ethnicity_loss.item()
            tot_val_samples += img.shape[0]
            
        avg_val_loss = tot_val_loss / tot_val_samples
        avg_val_age_loss = tot_val_age_loss / tot_val_samples
        avg_val_gender_loss = tot_val_gender_loss / tot_val_samples
        avg_val_ethni_loss = tot_val_ethni_loss / tot_val_samples

        if (avg_val_loss < best_val_loss):
            torch.save(model.state_dict(),"models/mtl_face_model_v1.pt")
            print (f'Epoch {epoch}, val loss: {best_val_loss:.5f} -> {avg_val_loss:.5f}, train loss: {avg_train_loss:.5f}')
            print (f'Epoch {epoch}, age val loss: {avg_val_age_loss:.5f}, gender val loss: {avg_val_gender_loss:.5f}, ethnicity val loss: {avg_val_ethni_loss:.5f}')
            print()
            best_val_loss = avg_val_loss
        else:
            print (f'Epoch {epoch}, val loss: {avg_val_loss:.5f}, train loss: {avg_train_loss:.5f}')
            print (f'Epoch {epoch}, age val loss: {avg_val_age_loss:.5f}, gender val loss: {avg_val_gender_loss:.5f}, ethnicity val loss: {avg_val_ethni_loss:.5f}')
            print()


def train_age_model(num_epochs=10, model=None, optimizer=None,
                    train_loader=None, val_loader=None, age_criterion=None,
                    age_coeff=None):

    best_val_loss = float(math.inf)
    for epoch in range(num_epochs):
        avg_train_loss = 0
        tot_train_loss = 0
        tot_train_samples = 0
        
        # Training Loop
        model.train()
        for i, (img, age, gender, ethnicity) in enumerate(train_loader):
            img = img.cuda()
            age = age.float().cuda()
            gender = gender.long().cuda()
            ethnicity = ethnicity.long().cuda()

            # Clear grad
            optimizer.zero_grad()

            # Get outputs
            age_output = model(img)
            age_output = age_output.squeeze(1)

            # Calculate losses
            age_loss = age_criterion(age_output, age)
            multi_loss = age_coeff*age_loss
                
            tot_train_loss += multi_loss.item()
            tot_train_samples += img.shape[0]

            # Get grad
            multi_loss.backward()

            # Update model weights
            optimizer.step()
            
        avg_train_loss = tot_train_loss / tot_train_samples


        avg_val_loss = 0
        tot_val_loss = 0
        tot_val_samples = 0
        
        # Iterate through the val dataset
        model.eval()
        for i, (img, age, gender, ethnicity) in enumerate(val_loader):
            img = img.cuda()
            age = age.float().cuda()
            gender = gender.long().cuda()
            ethnicity = ethnicity.long().cuda()

            # Clear grad
            optimizer.zero_grad()

            # Get outputs
            age_output = model(img)
            age_output = age_output.squeeze(1)

            # Calculate losses
            age_loss = age_criterion(age_output, age)
            multi_loss = age_coeff*age_loss
            
            tot_val_loss += multi_loss.item()
            tot_val_samples += img.shape[0]
            
        avg_val_loss = tot_val_loss / tot_val_samples

        if (avg_val_loss < best_val_loss):
            torch.save(model.state_dict(),"models/age_face_model_v1.pt")
            print (f'Epoch {epoch}, val loss: {best_val_loss:.5f} -> {avg_val_loss:.5f}, train loss: {avg_train_loss:.5f}')
            best_val_loss = avg_val_loss
        else:
            print (f'Epoch {epoch}, val loss: {avg_val_loss:.5f}, train loss: {avg_train_loss:.5f}')


def train_gender_model(num_epochs=10, model=None, optimizer=None,
                       train_loader=None, val_loader=None, gender_criterion=None,
                       gender_coeff=None):

    best_val_loss = float(math.inf)
    for epoch in range(num_epochs):
        avg_train_loss = 0
        tot_train_loss = 0
        tot_train_samples = 0
        
        # Training Loop
        model.train()
        for i, (img, age, gender, ethnicity) in enumerate(train_loader):
            img = img.cuda()
            age = age.float().cuda()
            gender = gender.long().cuda()
            ethnicity = ethnicity.long().cuda()

            # Clear grad
            optimizer.zero_grad()

            # Get outputs
            gender_output = model(img)

            # Calculate losses
            gender_loss = gender_criterion(gender_output, gender)
            multi_loss = gender_coeff*gender_loss

            tot_train_loss += multi_loss.item()
            tot_train_samples += img.shape[0]

            # Get grad
            multi_loss.backward()

            # Update model weights
            optimizer.step()
            
        avg_train_loss = tot_train_loss / tot_train_samples


        avg_val_loss = 0
        tot_val_loss = 0
        tot_val_samples = 0
        
        # Iterate through the val dataset
        model.eval()
        for i, (img, age, gender, ethnicity) in enumerate(val_loader):
            img = img.cuda()
            age = age.float().cuda()
            gender = gender.long().cuda()
            ethnicity = ethnicity.long().cuda()

            # Clear grad
            optimizer.zero_grad()

            # Get outputs
            gender_output = model(img)

            # Calculate losses
            gender_loss = gender_criterion(gender_output, gender)
            multi_loss = gender_coeff*gender_loss
            
            tot_val_loss += multi_loss.item()
            tot_val_samples += img.shape[0]
            
        avg_val_loss = tot_val_loss / tot_val_samples

        if (avg_val_loss < best_val_loss):
            torch.save(model.state_dict(),"models/gender_face_model_v1.pt")
            print (f'Epoch {epoch}, val loss: {best_val_loss:.5f} -> {avg_val_loss:.5f}, train loss: {avg_train_loss:.5f}')
            best_val_loss = avg_val_loss
        else:
            print (f'Epoch {epoch}, val loss: {avg_val_loss:.5f}, train loss: {avg_train_loss:.5f}')



def train_ethnicity_model(num_epochs=10, model=None, optimizer=None,
                          train_loader=None, val_loader=None, ethni_criterion=None,
                          ethni_coeff=None):

    best_val_loss = float(math.inf)
    for epoch in range(num_epochs):
        avg_train_loss = 0
        tot_train_loss = 0
        tot_train_samples = 0
        
        # Training Loop
        model.train()
        for i, (img, age, gender, ethnicity) in enumerate(train_loader):
            img = img.cuda()
            age = age.float().cuda()
            gender = gender.long().cuda()
            ethnicity = ethnicity.long().cuda()

            # Clear grad
            optimizer.zero_grad()

            # Get outputs
            ethni_output = model(img)

            # Calculate losses
            ethni_loss = ethni_criterion(ethni_output, ethnicity)
            multi_loss = ethni_coeff*ethni_loss
            
            tot_train_loss += multi_loss.item()
            tot_train_samples += img.shape[0]

            # Get grad
            multi_loss.backward()

            # Update model weights
            optimizer.step()
            
        avg_train_loss = tot_train_loss / tot_train_samples


        avg_val_loss = 0
        tot_val_loss = 0
        tot_val_samples = 0
        
        # Iterate through the val dataset
        model.eval()
        for i, (img, age, gender, ethnicity) in enumerate(val_loader):
            img = img.cuda()
            age = age.float().cuda()
            gender = gender.long().cuda()
            ethnicity = ethnicity.long().cuda()

            # Clear grad
            optimizer.zero_grad()

            # Get outputs
            ethni_output = model(img)

            # Calculate losses
            ethni_loss = ethni_criterion(ethni_output, ethnicity)
            multi_loss = ethni_coeff*ethni_loss
            
            tot_val_loss += multi_loss.item()
            tot_val_samples += img.shape[0]
            
        avg_val_loss = tot_val_loss / tot_val_samples

        if (avg_val_loss < best_val_loss):
            torch.save(model.state_dict(),"models/ethnicity_face_model_v1.pt")
            print (f'Epoch {epoch}, val loss: {best_val_loss:.5f} -> {avg_val_loss:.5f}, train loss: {avg_train_loss:.5f}')
            best_val_loss = avg_val_loss
        else:
            print (f'Epoch {epoch}, val loss: {avg_val_loss:.5f}, train loss: {avg_train_loss:.5f}')

