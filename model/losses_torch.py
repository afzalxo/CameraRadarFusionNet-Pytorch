import torch
import torch.nn as nn

def smooth_l1(sigma=3.0, alpha=1.0):
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        # separate target and state
        regression        = y_pred#[0, :, :]
        regression_target = y_true[:, :, :-1]
        anchor_state      = y_true[:, :, -1]

        regression = regression[y_true[:, :, -1] != -1]
        regression_target = regression_target[y_true[:, :, -1] != -1]
        
        regression_diff = torch.abs(regression - regression_target)
        regression_loss = torch.where(
                torch.lt(regression_diff, 1.0/sigma_squared),
                0.5 * sigma_squared * torch.pow(regression_diff, 2),
                regression_diff - 0.5/sigma_squared
        )
        normalizer = torch.maximum(torch.tensor(1), torch.tensor(regression.shape[0])).float()
        
        return alpha * torch.sum(regression_loss) / normalizer

    return _smooth_l1

def focal(alpha=0.25, gamma=2.0):

    def _focal(y_true, y_pred):
        classification        = y_pred
        labels                = y_true[:, :, :-1]
        anchor_state          = y_true[:, :, -1]

        classification  = classification[y_true[:, :, -1] != -1]
        labels = labels[y_true[:, :, -1] != -1]

        alpha_factor = torch.ones_like(labels) * alpha
        alpha_factor = torch.where(torch.eq(labels, 1), alpha_factor, 1-alpha_factor)
        focal_weight = torch.where(torch.eq(labels, 1), 1-classification, classification)

        focal_weight = alpha_factor * focal_weight ** gamma
        #print('--Classification--'*5)
        #print(classification)
        #print('--labels--'*5)
        #print(labels)
        #print('--End--'*5)
        cls_loss = focal_weight * torch.mean(nn.BCELoss(reduction='none')(classification, labels))#nn.functional.binary_cross_entropy(classification, labels, reduction='none')
        #print(torch.sum(nn.functional.binary_cross_entropy(classification, labels)))
        #print(torch.sum(torch.mean(nn.BCELoss(reduction='none')(classification, labels), axis=-1)))

        normalizer = torch.where(torch.eq(anchor_state, 1))
        #print('Printing Normalizer Torch:')
        #print(normalizer)
        #print('###')
        normalizer = torch.maximum(torch.tensor(1.0), torch.tensor(normalizer[1].shape[0]).float())

        return torch.sum(cls_loss) / normalizer

    return _focal
