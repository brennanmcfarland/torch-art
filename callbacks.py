import torch.utils.tensorboard as tensorboard


def tensorboard_record_loss():
    def _bind(steps_per_epoch):
        tensorboard_writer = tensorboard.SummaryWriter()

        def on_step(loss, step, epoch):
            tensorboard_writer.add_scalar('loss', loss, epoch * steps_per_epoch + step)

        return {"on_step": on_step}
    return _bind


def calc_interval_avg_loss(print_interval):
    def _bind(steps_per_epoch):
        functor = IntervalAvgLoss(steps_per_epoch, print_interval)
        return {"on_step": functor.on_step}
    return _bind


# helper functor for calc_interval_avg_loss, since it's stateful
class IntervalAvgLoss:

    def __init__(self, steps_per_epoch, print_interval):
        self.interval_avg_loss = 0.0
        self.steps_per_epoch = steps_per_epoch
        self.print_interval = print_interval

    def on_step(self, loss, step, epoch):
        interval_avg_loss = self.interval_avg_loss + loss.item()
        if step % self.print_interval == 0:
            print(
                'EPOCH ', epoch,
                ' STEP ', step, '/', self.steps_per_epoch,
                interval_avg_loss / self.print_interval
            )
            interval_avg_loss = 0
        return interval_avg_loss


def validate(validation_func, net, loader, metrics=None, device=None):
    def _bind(steps_per_epoch):
        def on_epoch_end():
            validation_func(net, loader, metrics, device)
        return {"on_epoch_end": on_epoch_end}
    return _bind
