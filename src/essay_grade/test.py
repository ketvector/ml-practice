import math

def MySigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def MyBCE(y_true, y_pred, from_logits = True):
    y_pred_probs = [MySigmoid(x) for x in y_pred]
    loss = 0.0
    for y_true_sample, y_pred_sample in zip(y_true, y_pred_probs):
        loss = loss - y_true_sample * math.log(y_pred_sample) - (1 - y_true_sample) * math.log(1 - y_pred_sample)
    return loss / len(y_true)

print(MyBCE([0, 1, 0, 0] , [-18.6, 0.51, 2.94, -12.8]))