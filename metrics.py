import torch


# TODO: if we need to add a bind step, make sure it's a consistent interface across all metrics
def calc_category_accuracy():
    functor = CategoryAccuracy()
    return {"on_item": functor.on_item, "on_end": functor.on_end}


# helper functor for calc_category_accuracy, since it's stateful
class CategoryAccuracy:

    def __init__(self):
        self.correct, self.total = 0, 0

    def on_item(self, inputs, outputs, gtruth):
        _, predicted = torch.max(outputs.data, 1)
        self.total += gtruth.size(0)
        self.correct += (predicted == gtruth).sum().item()

    def on_end(self):
        return self.correct / self.total
