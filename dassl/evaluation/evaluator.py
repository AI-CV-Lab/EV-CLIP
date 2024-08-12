import numpy as np
import os.path as osp
from collections import OrderedDict, defaultdict
import torch
from sklearn.metrics import f1_score, confusion_matrix
from torch.nn import functional as F

from .build import EVALUATOR_REGISTRY


class EvaluatorBase:
    """Base evaluator."""

    def __init__(self, cfg):
        self.cfg = cfg

    def reset(self):
        raise NotImplementedError

    def process(self, mo, gt):
        raise NotImplementedError

    def evaluate(self):
        raise NotImplementedError


@EVALUATOR_REGISTRY.register()
class Classification(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = 0
        self._total = 0
        self._loss = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct = 0
        self._total = 0
        self._loss = 0
        self._y_true = []
        self._y_pred = []
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        self._loss += F.cross_entropy(mo, gt)
        pred = mo.max(1)[1]
        matches = pred.eq(gt).float()
        self._correct += int(matches.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matches_i = int(matches[i].item())
                self._per_class_res[label].append(matches_i)

    def evaluate(self):
        results = OrderedDict()
        acc = 100.0 * self._correct / self._total
        err = 100.0 - acc
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )

        # The first value will be returned by trainer.test()
        results["accuracy"] = acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1

        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct: {self._correct:,}\n"
            f"* accuracy: {acc:.1f}%\n"
            f"* error: {err:.1f}%\n"
            f"* macro_f1: {macro_f1:.1f}%\n"
            f"* loss: {self._loss:.6f}"
        )

        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print("=> per-class result")
            accs = []

            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100.0 * correct / total
                accs.append(acc)
                print(
                    f"* class: {label} ({classname})\t"
                    f"total: {total:,}\t"
                    f"correct: {correct:,}\t"
                    f"acc: {acc:.1f}%"
                )
            mean_acc = np.mean(accs)
            print(f"* average: {mean_acc:.1f}%")

            results["perclass_accuracy"] = mean_acc

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize="true"
            )
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
            torch.save(cmat, save_path)
            print(f"Confusion matrix is saved to {save_path}")

        return results

@EVALUATOR_REGISTRY.register()
class Classification_TopK(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct = [0, 0]
        self._total = 0
        self._loss = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        self.cfg = cfg
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct = [0, 0]
        self._total = 0
        self._loss = 0
        self._y_true = []
        self._y_pred = []
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        self._loss += F.cross_entropy(mo, gt)
        
        maxk = max((1, self.cfg.TEST.TOPK))
        if isinstance(mo, (tuple, list)):
            mo = mo[0]
        _, pred = mo.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(gt.view(1, -1).expand_as(pred))
        
        self._correct[0] += int(correct[:1].contiguous().view(-1).float().sum(0, keepdim=True).item())
        self._correct[1] += int(correct[:self.cfg.TEST.TOPK].contiguous().view(-1).float().sum(0, keepdim=True).item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(mo.max(1)[1].data.cpu().numpy().tolist())

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matches_i = int(matches[i].item())
                self._per_class_res[label].append(matches_i)

    def evaluate(self):
        results = OrderedDict()
        acc = torch.tensor([100.0 * correct / self._total for correct in self._correct])
        err = torch.tensor([100.0 - a for a in acc])
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )

        # The first value will be returned by trainer.test()
        results[f"top1_accuracy"] = acc[0]
        results[f"top{self.cfg.TEST.TOPK}_accuracy"] = acc[1]
        results[f"top1_error_rate"] = err[0]
        results[f"top{self.cfg.TEST.TOPK}_error_rate"] = err[1]
        results["macro_f1"] = macro_f1

        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* correct(top1 / top{self.cfg.TEST.TOPK}): {self._correct[0]:,} / {self._correct[1]:,}\n"
            f"* accuracy(top1 / top{self.cfg.TEST.TOPK}): {acc[0]:.1f}% / {acc[1]:.1f}%\n"
            f"* error(top1 / top{self.cfg.TEST.TOPK}): {err[0]:.1f}% / {err[1]:.1f}%\n"
            f"* macro_f1: {macro_f1:.1f}%\n"
            f"* loss: {self._loss:.6f}"
        )

        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print("=> per-class result")
            accs = []

            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100.0 * correct / total
                accs.append(acc)
                print(
                    f"* class: {label} ({classname})\t"
                    f"total: {total:,}\t"
                    f"correct: {correct:,}\t"
                    f"acc: {acc:.1f}%"
                )
            mean_acc = np.mean(accs)
            print(f"* average: {mean_acc:.1f}%")

            results["perclass_accuracy"] = mean_acc

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize="true"
            )
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
            torch.save(cmat, save_path)
            print(f"Confusion matrix is saved to {save_path}")

        return results

# @EVALUATOR_REGISTRY.register()
# class Classification_TopK(EvaluatorBase):
#     """Evaluator for classification."""

#     def __init__(self, cfg, lab2cname=None, **kwargs):
#         super().__init__(cfg)
#         self._lab2cname = lab2cname
#         self._correct = [0, 0]
#         self._total = 0
#         self._loss = 0
#         self._per_class_res = None
#         self._y_true = []
#         self._y_pred = []
#         self.cfg = cfg
#         if cfg.TEST.PER_CLASS_RESULT:
#             assert lab2cname is not None
#             self._per_class_res = defaultdict(list)

#     def reset(self):
#         self._correct = [0, 0]
#         self._total = 0
#         self._loss = 0
#         self._y_true = []
#         self._y_pred = []
#         if self._per_class_res is not None:
#             self._per_class_res = defaultdict(list)

#     def process(self, mo, gt):
#         # mo (torch.Tensor): model output [batch, num_classes]
#         # gt (torch.LongTensor): ground truth [batch]
#         self._loss += F.cross_entropy(mo, gt)
        
#         maxk = max((1, self.cfg.TEST.TOPK))
#         if isinstance(mo, (tuple, list)):
#             mo = mo[0]
#         _, pred = mo.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))
        
#         self._correct[0] += int(correct[:1].view(-1).float().sum(0, keepdim=True).item())
#         self._correct[1] += int(correct[:self.cfg.TEST.TOPK].view(-1).float().sum(0, keepdim=True).item())
#         self._total += gt.shape[0]

#         self._y_true.extend(gt.data.cpu().numpy().tolist())
#         self._y_pred.extend(pred.data.cpu().numpy().tolist())

#         if self._per_class_res is not None:
#             for i, label in enumerate(gt):
#                 label = label.item()
#                 matches_i = int(matches[i].item())
#                 self._per_class_res[label].append(matches_i)

#     def evaluate(self):
#         results = OrderedDict()
#         acc = [100.0 * correct / self._total for correct in self._correct]
#         err = [100.0 - a for a in acc]
#         macro_f1 = 100.0 * f1_score(
#             self._y_true,
#             self._y_pred,
#             average="macro",
#             labels=np.unique(self._y_true)
#         )

#         # The first value will be returned by trainer.test()
#         results[f"accuracy(top_1 / top_{self.cfg.TEST.TOPK})"] = acc
#         results[f"error_rate(top_1 / top_{self.cfg.TEST.TOPK})"] = err
#         results["macro_f1"] = macro_f1

#         print(
#             "=> result\n"
#             f"* total: {self._total:,}\n"
#             f"* correct(top_1 / top_{self.cfg.TEST.TOPK}): {self._correct[0]:,} / {self._correct[1]:,}\n"
#             f"* accuracy(top_1 / top_{self.cfg.TEST.TOPK}): {acc[0]:.1f}% / {acc[1]:.1f}%\n"
#             f"* error(top_1 / top_{self.cfg.TEST.TOPK}): {err[0]:.1f}% / {err[1]:.1f}%\n"
#             f"* macro_f1: {macro_f1:.1f}%\n"
#             f"* loss: {self._loss:.6f}"
#         )

#         if self._per_class_res is not None:
#             labels = list(self._per_class_res.keys())
#             labels.sort()

#             print("=> per-class result")
#             accs = []

#             for label in labels:
#                 classname = self._lab2cname[label]
#                 res = self._per_class_res[label]
#                 correct = sum(res)
#                 total = len(res)
#                 acc = 100.0 * correct / total
#                 accs.append(acc)
#                 print(
#                     f"* class: {label} ({classname})\t"
#                     f"total: {total:,}\t"
#                     f"correct: {correct:,}\t"
#                     f"acc: {acc:.1f}%"
#                 )
#             mean_acc = np.mean(accs)
#             print(f"* average: {mean_acc:.1f}%")

#             results["perclass_accuracy"] = mean_acc

#         if self.cfg.TEST.COMPUTE_CMAT:
#             cmat = confusion_matrix(
#                 self._y_true, self._y_pred, normalize="true"
#             )
#             save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
#             torch.save(cmat, save_path)
#             print(f"Confusion matrix is saved to {save_path}")

#         return results
    

@EVALUATOR_REGISTRY.register()
class Classification_VNA(EvaluatorBase):
    """Evaluator for classification."""

    def __init__(self, cfg, lab2cname=None, **kwargs):
        super().__init__(cfg)
        self._lab2cname = lab2cname
        self._correct_action = 0
        self._correct_verb = 0
        self._correct_noun = 0
        self._total = 0
        self._per_class_res = None
        self._y_true = []
        self._y_pred = []
        if cfg.TEST.PER_CLASS_RESULT:
            assert lab2cname is not None
            self._per_class_res = defaultdict(list)

    def reset(self):
        self._correct_action = 0
        self._correct_verb = 0
        self._correct_noun = 0
        self._total = 0
        self._y_true = []
        self._y_pred = []
        if self._per_class_res is not None:
            self._per_class_res = defaultdict(list)

    def process(self, mo, gt, noun_size=10):
        # mo (torch.Tensor): model output [batch, num_classes]
        # gt (torch.LongTensor): ground truth [batch]
        pred = mo.max(1)[1]
        matches_action = pred.eq(gt).float()
        
        batch_size = gt.shape[0]
        verb_mo, noun_mo = pred // noun_size, pred % noun_size
        verb_gt, noun_gt = gt // noun_size, gt % noun_size
        
        matches_verb = verb_mo.eq(verb_gt).float()
        matches_noun = noun_mo.eq(noun_gt).float()
        
        self._correct_verb += int(matches_verb.sum().item())
        self._correct_noun += int(matches_noun.sum().item())
        self._correct_action += int(matches_action.sum().item())
        self._total += gt.shape[0]

        self._y_true.extend(gt.data.cpu().numpy().tolist())
        self._y_pred.extend(pred.data.cpu().numpy().tolist())

        if self._per_class_res is not None:
            for i, label in enumerate(gt):
                label = label.item()
                matches_i = int(matches[i].item())
                self._per_class_res[label].append(matches_i)

    def evaluate(self):
        results = OrderedDict()
        verb_acc = 100.0 * self._correct_verb / self._total
        noun_acc = 100.0 * self._correct_noun / self._total
        action_acc = 100.0 * self._correct_action / self._total
        err = 100.0 - action_acc
        macro_f1 = 100.0 * f1_score(
            self._y_true,
            self._y_pred,
            average="macro",
            labels=np.unique(self._y_true)
        )

        # The first value will be returned by trainer.test()
        results["action_accuracy"] = action_acc
        results["verb_accuracy"] = verb_acc
        results["noun_accuracy"] = noun_acc
        results["error_rate"] = err
        results["macro_f1"] = macro_f1

        print(
            "=> result\n"
            f"* total: {self._total:,}\n"
            f"* verb correct: {self._correct_verb:,}\n"
            f"* noun correct: {self._correct_noun:,}\n"
            f"* action correct: {self._correct_action:,}\n"
            f"* verb accuracy: {verb_acc:.1f}%\n"
            f"* noun accuracy: {noun_acc:.1f}%\n"
            f"* action accuracy: {action_acc:.1f}%\n"
            f"* action error: {err:.1f}%\n"
            f"* macro_f1: {macro_f1:.1f}%"
        )

        if self._per_class_res is not None:
            labels = list(self._per_class_res.keys())
            labels.sort()

            print("=> per-class result")
            accs = []

            for label in labels:
                classname = self._lab2cname[label]
                res = self._per_class_res[label]
                correct = sum(res)
                total = len(res)
                acc = 100.0 * correct / total
                accs.append(acc)
                print(
                    f"* class: {label} ({classname})\t"
                    f"total: {total:,}\t"
                    f"correct: {correct:,}\t"
                    f"acc: {acc:.1f}%"
                )
            mean_acc = np.mean(accs)
            print(f"* average: {mean_acc:.1f}%")

            results["perclass_accuracy"] = mean_acc

        if self.cfg.TEST.COMPUTE_CMAT:
            cmat = confusion_matrix(
                self._y_true, self._y_pred, normalize="true"
            )
            save_path = osp.join(self.cfg.OUTPUT_DIR, "cmat.pt")
            torch.save(cmat, save_path)
            print(f"Confusion matrix is saved to {save_path}")

        return results
