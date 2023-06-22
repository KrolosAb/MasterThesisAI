import torch
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from sampling_techniques import node_type_sampling, edge_type_sampling, pagerank_sampling

def weight_tuning(rdf_to_data, GNN, rdf_file, sampling_technique):
    """This function performs weight tuning for the given sampling technique"""
    
    # Defining the parameter grid
    if sampling_technique == node_type_sampling:
        param_grid = {'uri': [0.2, 0.4, 0.6, 0.8, 1], 'literal': [0.2, 0.4, 0.6, 0.8, 1], 'predicate': [0.2, 0.4, 0.6, 0.8, 1]}
        grid = ParameterGrid(param_grid)
    elif sampling_technique == edge_type_sampling:
        param_grid = {'subj_pred': [0.2, 0.4, 0.6, 0.8, 1], 'pred_obj': [0.2, 0.4, 0.6, 0.8, 1]}
        grid = ParameterGrid(param_grid)
    elif sampling_technique == pagerank_sampling:
        grid = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    best_score = -np.inf
    best_weights = None

    # Iterating over each combination of parameters
    for params in grid:      
        # Sampling the graph using the specified technique and parameters  
        data_sampled = rdf_to_data(rdf_file, sampling_technique, params)

        num_features = data_sampled.x.shape[1]
        num_classes = len(torch.unique(data_sampled.y))

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = GNN(num_features=num_features, num_classes=num_classes).to(device)
        data_sampled = data_sampled.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

        loss_fn = torch.nn.CrossEntropyLoss()

        model.train()

        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        sampled_losses = []

        for epoch in range(1000):
            optimizer.zero_grad()
            out = model(data_sampled)
            
            loss = loss_fn(out[data_sampled.train_mask], data_sampled.y[data_sampled.train_mask])
            loss.backward()
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_out = model(data_sampled)
                val_loss = loss_fn(val_out[data_sampled.val_mask], data_sampled.y[data_sampled.val_mask])

            if val_loss >= best_val_loss:
                patience_counter += 1
            else:
                best_val_loss = val_loss
                torch.save(model.state_dict(), 'best_model.pth')
                patience_counter = 0

            if patience_counter >= patience:
                break
            
            sampled_losses.append(loss.item())

            model.train()

        model.load_state_dict(torch.load('best_model.pth'))

        model.eval()
        out = model(data_sampled)
        _, pred = out.max(dim=1)

        probs = torch.nn.functional.softmax(out, dim=1)

        y_true = data_sampled.y[data_sampled.test_mask].cpu().numpy()
        y_pred = pred[data_sampled.test_mask].cpu().numpy()
        y_prob = probs.detach().cpu().numpy()
        y_prob_test = y_prob[data_sampled.test_mask.cpu().numpy()]

        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))

        score = roc_auc_score(y_true_bin, y_prob_test, multi_class='ovr', average='macro')

        # Updating the best score and best weights if this score is better than the current best
        if score > best_score:
            best_score = score
            best_weights = params

    return best_weights