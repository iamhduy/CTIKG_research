def compute_triple_similarity_roberta(predicted_triples, ground_truth_triples, threshold=0.5):
    """
    Compute similarity between predicted and ground truth triples using RoBERTa.
    Handles different sizes between predicted and ground truth sets.
    
    Args:
        predicted_triples: List of predicted triples
        ground_truth_triples: List of ground truth triples
        threshold: Similarity threshold (default: 0.5)
    
    Returns:
        metrics: Dictionary containing:
            - accuracy: Ratio of matched triples to total predicted triples
            - recall: Ratio of matched triples to total ground truth triples
            - f1_score: Harmonic mean of accuracy and recall
            - precision: Ratio of correct matches to total matches
        matches: List of matched triples with their similarity scores
    """
    import warnings
    from transformers import AutoTokenizer, AutoModel, AutoConfig
    import torch
    import torch.nn.functional as F
    
    # Suppress the specific warning about uninitialized weights
    warnings.filterwarnings("ignore", message='''Some weights of RobertaModel were not initialized from the model checkpoint at roberta-base and are newly initialized: ['pooler.dense.bias', 'pooler.dense.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.''')
    
    # Load pre-trained RoBERTa model and tokenizer with specific configuration
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    config = AutoConfig.from_pretrained(
        'roberta-base',
        output_hidden_states=True,
        use_cache=True
    )
    model = AutoModel.from_pretrained(
        'roberta-base',
        config=config,
        ignore_mismatched_sizes=True
    )
    model.eval()  # Set model to evaluation mode
    
    def get_embeddings(triples):
        texts = [' '.join(triple) for triple in triples]
        encoded = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = model(**encoded)
            # Use the last hidden state instead of the pooler output
            embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings
    
    # Get embeddings
    pred_embeddings = get_embeddings(predicted_triples)
    gt_embeddings = get_embeddings(ground_truth_triples)
    
    # Compute similarity matrix
    similarity_matrix = torch.mm(pred_embeddings, gt_embeddings.t())
    
    # Find matches
    matches = []
    used_gt_indices = set()  # Track which ground truth triples have been matched
    
    for i, pred in enumerate(predicted_triples):
        best_match_idx = torch.argmax(similarity_matrix[i]).item()
        best_match_score = similarity_matrix[i][best_match_idx].item()
        
        if best_match_score >= threshold and best_match_idx not in used_gt_indices:
            matches.append({
                'predicted': pred,
                'ground_truth': ground_truth_triples[best_match_idx],
                'similarity': best_match_score
            })
            used_gt_indices.add(best_match_idx)
    
    # Calculate metrics
    num_matches = len(matches)
    num_predicted = len(predicted_triples)
    num_ground_truth = len(ground_truth_triples)
    
    # Handle edge cases
    if num_predicted == 0 or num_ground_truth == 0:
        return {
            'accuracy': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'precision': 0.0
        }, []
    
    # Calculate metrics
    accuracy = num_matches / num_predicted
    recall = num_matches / num_ground_truth
    precision = num_matches / num_predicted if num_predicted > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics = {
        'accuracy': accuracy,
        'recall': recall,
        'f1_score': f1_score,
    }
    
    return metrics, matches


