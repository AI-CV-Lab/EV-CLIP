def compute_accuracy(output, target, topk=(1, )):
    """Computes the accuracy over the k top predictions for
    the specified values of k.

    Args:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor): ground truth labels with shape (batch_size).
        topk (tuple, optional): accuracy at top-k will be computed. For example,
            topk=(1, 5) means accuracy at top-1 and top-5 will be computed.

    Returns:
        list: accuracy at top-k.
    """
    maxk = max(topk)
    batch_size = target.size(0)

    if isinstance(output, (tuple, list)):
        output = output[0]

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size)
        res.append(acc)

    return res


def vn_accuracy(output, target, noun_size=10):
    """Computes the accuracy over verb & noun.

    Args:
        output (torch.Tensor): prediction matrix with shape (batch_size, num_classes).
        target (torch.LongTensor): ground truth labels with shape (batch_size).
        noun_size (int): the number of noun labels

    Returns:
        verb_acc, noun_acc (torch.FloatTensor): verb and noun accuracy.
    """
    batch_size = target.size(0)

    output_idx = output.argmax(1)
    verb_output, noun_output = output_idx // noun_size, output_idx % noun_size
    verb_label, noun_label = target // noun_size, target % noun_size
    
    verb_acc = 100.0 * verb_label.eq(verb_output).float().sum() / batch_size
    noun_acc = 100.0 * noun_label.eq(noun_output).float().sum() / batch_size

    return verb_acc, noun_acc
