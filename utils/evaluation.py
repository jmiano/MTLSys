import sklearn.metrics as perf
import torch
import matplotlib.pyplot as plt
import time


def get_classification_metrics(gt_vec, pred_vec):
    accuracy = perf.accuracy_score(gt_vec, pred_vec)
    f1 = perf.f1_score(gt_vec, pred_vec, average='macro')
    
    return f1, accuracy


def get_regression_metrics(gt_vec, pred_vec):
    r2 = perf.r2_score(gt_vec, pred_vec)
    rmse = perf.mean_squared_error(gt_vec, pred_vec, squared=False)

    return rmse, r2


def get_predictions(model, data_loader, task, mtl_model=True):
    model = model.cuda()
    model.eval()

    ### Get model performance
    predictions = []
    gt = []

    with torch.no_grad():
        for i, (img, age, gender, ethnicity) in enumerate(data_loader):
            img = img.cuda()
            age = age.float().cuda()
            gender = gender.long().cuda()
            ethnicity = ethnicity.long().cuda()

            # Get outputs
            if mtl_model:
                age_output, gender_output, ethnicity_output = model(img)
                age_output = age_output.squeeze(1)
                gender_output = gender_output
                ethnicity_output = ethnicity_output

                # Get predictions
                if 'age' in task:
                    age_pred = age_output.cpu()
                    predictions.extend(list(age_pred))
                    gt.extend(list(age.cpu()))
                if 'gender' in task:
                    gender_pred = torch.argmax(gender_output, axis=1).cpu()
                    predictions.extend(list(gender_pred))
                    gt.extend(list(gender.cpu()))
                if 'ethni' in task:
                    ethnicity_pred = torch.argmax(ethnicity_output, axis=1).cpu()
                    predictions.extend(list(ethnicity_pred))
                    gt.extend(list(ethnicity.cpu()))
            else:
                output = model(img)
                if 'age' in task:
                    output = output.squeeze(1)  # squeeze output if the task is age
                    pred = output.cpu()
                    predictions.extend(list(pred))
                    gt.extend(list(age.cpu()))
                elif 'gender' in task:
                    pred = torch.argmax(output, axis=1).cpu()
                    predictions.extend(list(pred))
                    gt.extend(list(gender.cpu()))
                elif 'ethni' in task:
                    pred = torch.argmax(output, axis=1).cpu()
                    predictions.extend(list(pred))
                    gt.extend(list(ethnicity.cpu()))
                else:
                    print('Error.')
                    
    return gt, predictions


def run_evaluation(model, data_loader, tasks, mtl_model=True):
    score_dict = {}
    for task in tasks:
        gt, predictions = get_predictions(model, data_loader, task, mtl_model)
        if 'age' in task:
            score_dict[task] = get_regression_metrics(gt, predictions)
        else:
            score_dict[task] = get_classification_metrics(gt, predictions)
    return score_dict


def show_example_predictions(model, data_loader, mtl_model=True, use_gpu=True, num_predictions=10):
    model.eval()
    if use_gpu:
        model = model.cuda()
    predictions = []
    if mtl_model:
        with torch.no_grad():
            for i, (img, age, gender, ethnicity) in enumerate(data_loader):
                start_time = time.time()
                if use_gpu:
                    img = img.cuda()
                    age = age.float().cuda()
                    gender = gender.long().cuda()
                    ethnicity = ethnicity.long().cuda()

                # Get outputs
                age_output, gender_output, ethnicity_output = model(img)
                age_output = age_output.squeeze(1)
                gender_output = gender_output
                ethnicity_output = ethnicity_output

                # Get predictions
                age_pred = age_output
                gender_pred = torch.argmax(gender_output, axis=1)
                ethnicity_pred = torch.argmax(ethnicity_output, axis=1)
                latency = (time.time() - start_time) * 1000
                predictions.append([img, (age_pred, age), (gender_pred, gender), (ethnicity_pred, ethnicity), latency])
                if i >= num_predictions:
                    break

        for img, (age_pred, age), (gender_pred, gender), (ethnicity_pred, ethnicity), latency in predictions:
            img = img.squeeze(0)
            plt.imshow(img.transpose(0, 2).transpose(0, 1).cpu())
            plt.axis('off')
            plt.title(f'''age prediction: {age_pred.cpu().item(): .2f}, true age: {age.cpu().item()}\ngender prediction: {gender_pred.cpu().item()}, true gender: {gender.cpu().item()}\nethnicity prediction: {ethnicity_pred.cpu().item()}, true ethnicity: {ethnicity.cpu().item()}\nlatency: {latency:.3f} ms''')
            plt.show()
    else:
        print('show_example_predictions is only supported for MTL models so far.')

            
            
            
            
            