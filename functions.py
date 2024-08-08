import torch
from sklearn.metrics import classification_report


def set_seed(seed):
    torch.set_default_tensor_type('torch.FloatTensor')
    torch.manual_seed(seed)

def classification_result(y, pred):

    res = classification_report(y.cpu().numpy(), pred.cpu().numpy(), output_dict=True)

    acc, macro, weighted = res['accuracy'], res['macro avg'], res['weighted avg']
    precision_macro, recall_macro, f1_macro = macro['precision'], macro['recall'], macro['f1-score']
    precision_weighted, recall_weighted, f1_weighted = weighted['precision'], weighted['recall'], weighted['f1-score']

    return acc*100, precision_macro*100, recall_macro*100, f1_macro*100, precision_weighted*100, recall_weighted*100, f1_weighted*100


def contrastive_loss(features, temp=1.0):
    """Compute contrastive loss of data features
    Args:
        features: list of tensors [batch_size, fea_dim]
        temp: temperature, hyperparameter
    Returns:
        loss
    """

    # flatten features
    n_views, batch_size, device = len(features), features[0].shape[0], features[0].device
    features = torch.cat(features, dim=0)
    features = torch.nn.functional.normalize(features, dim=1)

    # get mask
    mask = torch.eye(batch_size, dtype=torch.float32).to(device)
    mask = mask.repeat(n_views, n_views)
    # mask-out self-contrast cases
    logits_mask = torch.scatter(
        torch.ones_like(mask),
        1,
        torch.arange(batch_size * n_views).view(-1, 1).to(device),
        0
    )
    mask = mask * logits_mask
    mask = mask.to(device)

    logits = torch.div(torch.matmul(features, features.T), temp)

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    mean_log_prob_pos = - (mask * log_prob).sum(1) / mask.sum(1)
    loss = mean_log_prob_pos.mean()

    return loss
