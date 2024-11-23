import torch

def concordance_index(risk_scores, events, times):
    """
    Calculate the concordance index given risk scores and times
    :param risk_scores: risk scores for each patient
    :param times: survival times for each patient
    :return: concordance index
    """

    # Precompute event matrix for efficiency
    event_matrix = events.byte().unsqueeze(1)

    # Create matrices for pairwise comparison
    risk_matrix = risk_scores.unsqueeze(1) - risk_scores.unsqueeze(0)
    times_matrix = times.unsqueeze(1) - times.unsqueeze(0)

    # Count concordant and permissible pairs
    concordant = (risk_matrix > 0) & (times_matrix < 0) & event_matrix
    permissible = (times_matrix < 0) & event_matrix

    concordant_count = torch.sum(concordant)
    permissible_count = torch.sum(permissible)

    # Handle case where there are no permissible pairs
    if permissible_count > 0:
        return torch.tensor(concordant_count / permissible_count, dtype=torch.float32, requires_grad=True)
    else:
        return torch.tensor(0.5, dtype=torch.float32, requires_grad=True)

def brier_score(risk_scores, events):
    """
    Calculate the Brier score given risk scores and events
    :param risk_scores: risk scores for each patient
    :param events: events for each patient
    :return: Brier score
    """
    return torch.mean((risk_scores - events) ** 2)
