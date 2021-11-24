import math
import torch


def train_mtl_model(num_epochs=10, model=None, optimizer=None,
                    train_loader=None, val_loader=None, 
                    age_criterion=None, gender_criterion=None, ethnicity_criterion=None,
                    age_coeff=None, gender_coeff=None, ethnicity_coeff=None, save=True,
                    save_name=None):

    best_val_loss = float(math.inf)
    model = model.cuda()
    for epoch in range(num_epochs):
        avg_train_loss = 0
        tot_train_loss = 0
        tot_train_age_loss = 0
        tot_train_gender_loss = 0
        tot_train_ethnicity_loss = 0
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
            ethnicity_loss = ethnicity_criterion(ethnicity_output, ethnicity)
            multi_loss = age_coeff*age_loss + gender_coeff*gender_loss + ethnicity_coeff*ethnicity_loss
                
            tot_train_loss += multi_loss.item()
            tot_train_age_loss += age_coeff*age_loss.item()
            tot_train_gender_loss += gender_coeff*gender_loss.item()
            tot_train_ethnicity_loss += ethnicity_coeff*ethnicity_loss.item()
            tot_train_samples += img.shape[0]

            # Get grad
            multi_loss.backward()

            # Update model weights
            optimizer.step()
            
        avg_train_loss = tot_train_loss / tot_train_samples
        avg_train_age_loss = tot_train_age_loss / tot_train_samples
        avg_train_gender_loss = tot_train_gender_loss / tot_train_samples
        avg_train_ethnicity_loss = tot_train_ethnicity_loss / tot_train_samples


        avg_val_loss = 0
        tot_val_loss = 0
        tot_val_age_loss = 0
        tot_val_gender_loss = 0
        tot_val_ethnicity_loss = 0
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
            ethnicity_loss = ethnicity_criterion(ethnicity_output, ethnicity)
            multi_loss = age_coeff*age_loss + gender_coeff*gender_loss + ethnicity_coeff*ethnicity_loss
            
            tot_val_loss += multi_loss.item()
            tot_val_age_loss += age_coeff*age_loss.item()
            tot_val_gender_loss += gender_coeff*gender_loss.item()
            tot_val_ethnicity_loss += ethnicity_coeff*ethnicity_loss.item()
            tot_val_samples += img.shape[0]
            
        avg_val_loss = tot_val_loss / tot_val_samples
        avg_val_age_loss = tot_val_age_loss / tot_val_samples
        avg_val_gender_loss = tot_val_gender_loss / tot_val_samples
        avg_val_ethnicity_loss = tot_val_ethnicity_loss / tot_val_samples

        if (avg_val_loss < best_val_loss):
            if save:
                torch.save(model, f"models/{save_name}")
            print (f'Epoch {epoch}, val loss: {best_val_loss:.5f} -> {avg_val_loss:.5f}, train loss: {avg_train_loss:.5f}')
            print (f'Epoch {epoch}, age val loss: {avg_val_age_loss:.5f}, gender val loss: {avg_val_gender_loss:.5f}, ethnicity val loss: {avg_val_ethnicity_loss:.5f}')
            best_val_loss = avg_val_loss
        else:
            print (f'Epoch {epoch}, val loss: {avg_val_loss:.5f}, train loss: {avg_train_loss:.5f}')
            print (f'Epoch {epoch}, age val loss: {avg_val_age_loss:.5f}, gender val loss: {avg_val_gender_loss:.5f}, ethnicity val loss: {avg_val_ethnicity_loss:.5f}')


def train_age_model(num_epochs=10, model=None, optimizer=None,
                       train_loader=None, val_loader=None, age_criterion=None,
                       age_coeff=None, save=True, save_name=None):

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
            if save:
                torch.save(model, f"models/{save_name}")
            print (f'Epoch {epoch}, val loss: {best_val_loss:.5f} -> {avg_val_loss:.5f}, train loss: {avg_train_loss:.5f}')
            best_val_loss = avg_val_loss
        else:
            print (f'Epoch {epoch}, val loss: {avg_val_loss:.5f}, train loss: {avg_train_loss:.5f}')


def train_gender_model(num_epochs=10, model=None, optimizer=None,
                       train_loader=None, val_loader=None, gender_criterion=None,
                       gender_coeff=None, save=True, save_name=None):

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
            gender_output= model(img)

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
            if save:
                torch.save(model, f"models/{save_name}")
            print (f'Epoch {epoch}, val loss: {best_val_loss:.5f} -> {avg_val_loss:.5f}, train loss: {avg_train_loss:.5f}')
            best_val_loss = avg_val_loss
        else:
            print (f'Epoch {epoch}, val loss: {avg_val_loss:.5f}, train loss: {avg_train_loss:.5f}')



def train_ethnicity_model(num_epochs=10, model=None, optimizer=None,
                          train_loader=None, val_loader=None, ethnicity_criterion=None,
                          ethnicity_coeff=None):

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
            ethnicity_output = model(img)

            # Calculate losses
            ethnicity_loss = ethnicity_criterion(ethnicity_output, ethnicity)
            multi_loss = ethnicity_coeff*ethnicity_loss
            
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
            ethnicity_output = model(img)

            # Calculate losses
            ethnicity_loss = ethnicity_criterion(ethnicity_output, ethnicity)
            multi_loss = ethnicity_coeff*ethnicity_loss
            
            tot_val_loss += multi_loss.item()
            tot_val_samples += img.shape[0]
            
        avg_val_loss = tot_val_loss / tot_val_samples

        if (avg_val_loss < best_val_loss):
            if save:
                torch.save(model, f"models/{save_name}")
            print (f'Epoch {epoch}, val loss: {best_val_loss:.5f} -> {avg_val_loss:.5f}, train loss: {avg_train_loss:.5f}')
            best_val_loss = avg_val_loss
        else:
            print (f'Epoch {epoch}, val loss: {avg_val_loss:.5f}, train loss: {avg_train_loss:.5f}')


def train_mtl_model_individual(num_epochs=10, model=None, optimizer=None,
                    train_loader=None, val_loader=None, 
                    age_criterion=None, gender_criterion=None, ethnicity_criterion=None,
                    age_coeff=None, gender_coeff=None, ethnicity_coeff=None, save=True, isAge=True, isGender=True, isethnicity=True):

    best_val_loss = float(math.inf)
    for epoch in range(num_epochs):
        avg_train_loss = 0
        tot_train_loss = 0
        tot_train_age_loss = 0
        tot_train_gender_loss = 0
        tot_train_ethnicity_loss = 0
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
            if isAge:
                age_loss = age_criterion(age_output, age)
                multi_loss = age_coeff*age_loss
            
            if isGender:
                gender_loss = gender_criterion(gender_output, gender)
                multi_loss = gender_coeff*gender_loss
            if isethnicity:
                ethnicity_loss = ethnicity_criterion(ethnicity_output, ethnicity)
                multi_loss = ethnicity_coeff*ethnicity_loss
#             multi_loss = age_coeff*age_loss + gender_coeff*gender_loss + ethnicity_coeff*ethnicity_loss
                
            tot_train_loss += multi_loss.item()
#             tot_train_age_loss += age_coeff*age_loss.item()
#             tot_train_gender_loss += gender_coeff*gender_loss.item()
#             tot_train_ethnicity_loss += ethnicity_coeff*ethnicity_loss.item()
            tot_train_samples += img.shape[0]

            # Get grad
            multi_loss.backward()

            # Update model weights
            optimizer.step()
            
        avg_train_loss = tot_train_loss / tot_train_samples
#         avg_train_age_loss = tot_train_age_loss / tot_train_samples
#         avg_train_gender_loss = tot_train_gender_loss / tot_train_samples
#         avg_train_ethnicity_loss = tot_train_ethnicity_loss / tot_train_samples


        avg_val_loss = 0
        tot_val_loss = 0
        tot_val_age_loss = 0
        tot_val_gender_loss = 0
        tot_val_ethnicity_loss = 0
        tot_val_samples = 0

        print (f'Epoch {epoch}, val loss: {avg_val_loss:.5f}, train loss: {avg_train_loss:.5f}')
#         print (f'Epoch {epoch}, age val loss: {avg_val_age_loss:.5f}, gender val loss: {avg_val_gender_loss:.5f}, ethnicity val loss: {avg_val_ethnicity_loss:.5f}')
        print()
