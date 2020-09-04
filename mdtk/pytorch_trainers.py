import logging
import sys
from collections import defaultdict

import numpy as np
import torch
import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader

from mdtk.degradations import MAX_PITCH_DEFAULT, MIN_PITCH_DEFAULT
from mdtk.eval import get_f1, helpfulness


class BaseTrainer:
    """Provides methods to train pytorch models. Adapted from:
    https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/trainer/pretrain.py"""  # noqa:E501

    def __init__(
        self,
        model,
        criterion,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader = None,
        lr: float = 1e-4,
        betas=(0.9, 0.999),
        weight_decay: float = 0.01,
        with_cuda: bool = True,
        batch_log_freq=None,
        epoch_log_freq=1,
        formatter=None,
        log_file=None,
    ):
        """
        Parameters
        ----------
        model : torch.nn.module
            the pytorch model to train

        criterion : torch loss object
            the pytorch loss to optimise for

        train_dataloader : pytorch.utils.data.Dataloader
            train dataset data loader

        test_dataloader : pytorch.utils.data.Dataloader
            test dataset data loader

        lr : float
            learning rate of optimizer

        betas: tuple[float]
            Adam optimizer betas

        weight_decay: float
            Adam optimizer weight decay param

        with_cuda:
            traning with cuda

        batch_log_freq: int
            How many batch iterations to run before logging results

        epoch_log_freq: int
            How many epoch iterations to run before logging results

        formatter : dict
            A formatter defined in formatters.py.FORMATTERS.

        log_file : filehandle
            A file handle with a write method to write logs to
        """

        # Set up cuda device
        self.device = torch.device("cpu")
        cuda_condition = torch.cuda.is_available()
        if with_cuda:
            if not cuda_condition:
                print("You set with_cuda to True, but cuda was not available")
                print("Using cpu")
            else:
                self.device = torch.device("cuda:0")

        # This model will be saved every epoch
        self.model = model.to(self.device)

        #        # Distributed GPU training if CUDA can detect more than 1 GPU
        #        if with_cuda and torch.cuda.device_count() > 1:
        #            print(f"Using {torch.cuda.device_count()} GPUS for training")
        #            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        # Setting the train and test data loader
        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optimizer = Adam(
            self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay
        )

        self.criterion = criterion

        self.log_file = log_file if log_file is not None else sys.stdout
        # Defines the columns of self.log_file written in iteration
        # best as an attribute of class as then can be used to head the file
        self.log_cols = []
        self.batch_log_freq = batch_log_freq
        self.epoch_log_freq = epoch_log_freq

        self.formatter = formatter

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        log_info = self.iteration(epoch, self.train_data)
        self.log_file.flush()
        return log_info

    def test(self, epoch, evaluate=False):
        with torch.no_grad():
            log_info = self.iteration(
                epoch, self.test_data, train=False, evaluate=evaluate
            )
        self.log_file.flush()
        return log_info

    def iteration(self, epoch, data_loader, train=True, evaluate=False):
        """This must be overwritten by classes inheriting this"""
        raise NotImplementedError()

    def save(self, file_path=None, epoch=None):
        """
        Saving the current model on file_path
        :param file_path: model output path. If epoch is not None then this is
            appended with .ep{epoch}
        :param epoch: current epoch number
        :return: final_output_path
        """
        if file_path is None:
            file_path = "trained.model"
        if epoch is not None:
            output_path = file_path + ".ep%d" % epoch
        else:
            output_path = file_path
        torch.save(self.model.cpu(), output_path)
        self.model.to(self.device)
        print(f"Model saved {output_path}")
        return output_path


class ErrorDetectionTrainer(BaseTrainer):
    """Trains Task 1 - Error detection. The model provided is expected to be
    an mdtk.pytorch_models.ErrorDetectionNet. Expects a DataLoader using an
    mdtk.pytorch_datasets.CommandDataset."""

    def __init__(
        self,
        model,
        criterion,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader = None,
        lr: float = 1e-4,
        betas=(0.9, 0.999),
        weight_decay: float = 0.01,
        with_cuda: bool = True,
        batch_log_freq=None,
        epoch_log_freq=1,
        formatter=None,
        log_file=None,
    ):
        if formatter["task_labels"][0] is None:
            raise NotImplementedError(
                "Formatter " + formatter["name"] + " has not"
                " implemented a ground truth for this task."
            )
        super().__init__(
            model=model,
            criterion=criterion,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            with_cuda=with_cuda,
            batch_log_freq=batch_log_freq,
            epoch_log_freq=epoch_log_freq,
            formatter=formatter,
            log_file=log_file,
        )
        self.log_cols = ["epoch", "batch", "mode", "avg_loss", "avg_acc"]

    def iteration(self, epoch, data_loader, train=True, evaluate=False):
        """
        The loop used for train and test methods. Loops over the provided
        data_loader for one epoch getting only the necessary data for the task.
        If in train mode, a backward pass is performed, updating the parameters
        and saving the model.

        Paremeters
        ----------
        epoch: int
            current epoch index, only used for progress bar

        data_loader: torch.utils.data.DataLoader
            dataloader to get the data from

        train: bool
            Whether to operate in train or test mode. Train mode performs
            backpropagation and saves the model.

        Returns
        -------
        None
        """
        if train:
            self.model.train()
        else:
            self.model.eval()  # informs batchnorm/dropout layers

        str_code = "train" if train else "test"

        # Values to accumulate over the batch
        total_loss = 0.0
        total_correct = 0
        total_element = 0
        total_positive = 0
        total_positive_labels = 0
        total_true_pos = 0

        total_correct_per_deg = defaultdict(lambda: 0)
        total_element_per_deg = defaultdict(lambda: 0)

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc=f"{str_code} epoch {epoch}",
            postfix={"avg_loss": 0},
            bar_format="{l_bar}{bar} batch {r_bar}",
            total=len(data_loader),
        )

        for ii, data in data_iter:
            input_lengths = np.array(data["deg_len"]) if "deg_len" in data else None
            # N tensors of integers representing (potentially) degraded midi
            input_data = data[self.formatter["deg_label"]].to(self.device)
            # N integers of the labels - 0 assumed to be no degradation
            # N.B. CrossEntropy expects this to be of type long
            labels = (data[self.formatter["task_labels"][0]] > 0).long().to(self.device)
            model_output = self.model.forward(input_data, input_lengths)
            loss = self.criterion(model_output, labels)

            # backward pass and optimization only in train
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # values for logging
            predictions = model_output.argmax(dim=-1)
            correct = predictions.eq(labels).sum().item()
            true_pos = (predictions & labels).sum().item()
            total_loss += loss.item() * labels.nelement()
            # N.B. if loss is using reduction='mean'
            # summing the average losses over the
            # batches and then dividing by the number
            # of batches does not give you the true
            # mean loss (though it is at least an
            # unbiased estimate...)
            # Using total rather than sum to account
            # For the last batch being a different size
            total_correct += correct
            total_element += labels.nelement()
            total_positive += predictions.sum().item()
            total_positive_labels += labels.sum().item()
            total_true_pos += true_pos

            log_info = {
                "epoch": epoch,
                "batch": ii,
                "mode": str_code,
                "avg_loss": total_loss / total_element,
                "avg_acc": total_correct / total_element * 100,
            }

            # I'm only calculating avg acc per deg_type if evaluating so as not to
            # affect training speed
            if evaluate:
                avg_acc_per_deg = {}
                degradation_type_labels = data[self.formatter["task_labels"][0]]
                for deg_label_value in degradation_type_labels.unique():
                    idx = degradation_type_labels == deg_label_value
                    deg_labels = labels[idx]
                    deg_predictions = predictions[idx]
                    deg_correct = deg_predictions.eq(deg_labels).sum().item()
                    total_correct_per_deg[deg_label_value] += deg_correct
                    total_element_per_deg[deg_label_value] += deg_labels.nelement()
                    avg_acc_per_deg[deg_label_value] = (
                        total_correct_per_deg[deg_label_value]
                        / total_element_per_deg[deg_label_value]
                    )
                log_info["avg_acc_per_deg"] = avg_acc_per_deg

            if evaluate:
                # labels are 0, 1 for task1, rather than the deg type
                degradation_type_labels = data[self.formatter["task_labels"][0]]
                for label, pred, deg_label in zip(
                    labels.cpu(),
                    predictions.cpu().data.numpy(),
                    degradation_type_labels.cpu().data.numpy(),
                ):
                    deg_label
                    total_element_per_deg[deg_label] += 1
                    if label == pred:
                        total_correct_per_deg[deg_label] += 1

            if self.batch_log_freq is not None:
                if ii % self.batch_log_freq == 0:
                    print(
                        ",".join([str(log_info[kk]) for kk in self.log_cols]),
                        file=self.log_file,
                    )

            data_iter.set_postfix(avg_loss=round(log_info["avg_loss"], ndigits=3))

        if self.epoch_log_freq is not None:
            if epoch % self.epoch_log_freq == 0:
                print(
                    ",".join([str(log_info[kk]) for kk in self.log_cols]),
                    file=self.log_file,
                )

        if evaluate:
            tp = total_true_pos
            fn = total_positive_labels - tp
            fp = total_positive - tp
            tn = total_element - tp - fn - fp
            p, r, f = get_f1(tp, fp, fn)
            rev_p, rev_r, rev_f = get_f1(tn, fn, fp)
            log_info["p"] = p
            log_info["r"] = r
            log_info["f"] = f
            log_info["rev_p"] = rev_p
            log_info["rev_r"] = rev_r
            log_info["rev_f"] = rev_f
            print(f"P, R, F-measure: {p}, {r}, {f}")
            print(f"Reverse P, R, F-measure: {rev_p}, {rev_r}, {rev_f}")
            print(f"Avg loss: {total_loss / total_element}")
            log_info["avg_acc_per_deg"] = np.array(
                [
                    total_correct_per_deg[deg] / total_element_per_deg[deg]
                    for deg in sorted(total_element_per_deg.keys())
                ]
            )

        return log_info


class ErrorClassificationTrainer(BaseTrainer):
    """Trains Task 2 - Error classification. The model provided is expected to be
    an mdtk.pytorch_models.ErrorClassificationNet. Expects a DataLoader using an
    mdtk.pytorch_datasets.CommandDataset."""

    def __init__(
        self,
        model,
        criterion,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader = None,
        lr: float = 1e-4,
        betas=(0.9, 0.999),
        weight_decay: float = 0.01,
        with_cuda: bool = True,
        batch_log_freq=None,
        epoch_log_freq=1,
        formatter=None,
        log_file=None,
    ):
        if formatter["task_labels"][1] is None:
            raise NotImplementedError(
                "Formatter " + formatter["name"] + " has not"
                " implemented a ground truth for this task."
            )
        super().__init__(
            model=model,
            criterion=criterion,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            with_cuda=with_cuda,
            batch_log_freq=batch_log_freq,
            epoch_log_freq=epoch_log_freq,
            formatter=formatter,
            log_file=log_file,
        )
        self.log_cols = ["epoch", "batch", "mode", "avg_loss", "avg_acc"]

    def iteration(self, epoch, data_loader, train=True, evaluate=False):
        """
        The loop used for train and test methods. Loops over the provided
        data_loader for one epoch getting only the necessary data for the task.
        If in train mode, a backward pass is performed, updating the parameters
        and saving the model.

        Paremeters
        ----------
        epoch: int
            current epoch index, only used for progress bar

        data_loader: torch.utils.data.DataLoader
            dataloader to get the data from

        train: bool
            Whether to operate in train or test mode. Train mode performs
            backpropagation and saves the model.

        Returns
        -------
        None
        """
        if train:
            self.model.train()
        else:
            self.model.eval()  # informs batchnorm/dropout layers

        str_code = "train" if train else "test"

        # Values to accumulate over the batch
        total_loss = 0
        total_correct = 0
        total_element = 0
        confusion_mat = None

        total_correct_per_deg = defaultdict(lambda: 0)
        total_element_per_deg = defaultdict(lambda: 0)

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc=f"{str_code} epoch {epoch}",
            postfix={"avg_loss": 0},
            bar_format="{l_bar}{bar} batch {r_bar}",
            total=len(data_loader),
        )

        for ii, data in data_iter:
            input_lengths = np.array(data["deg_len"]) if "deg_len" in data else None
            # N tensors of integers representing (potentially) degraded midi
            input_data = data[self.formatter["deg_label"]].to(self.device)
            # N integers of the labels - 0 assumed to be no degradation
            # N.B. CrossEntropy expects this to be of type long
            labels = (data[self.formatter["task_labels"][1]]).long().to(self.device)
            model_output = self.model.forward(input_data, input_lengths)
            loss = self.criterion(model_output, labels)

            # backward pass and optimization only in train
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # values for logging
            predictions = model_output.argmax(dim=-1)
            correct = predictions.eq(labels).sum().item()
            total_loss += loss.item() * labels.nelement()
            # N.B. if loss is using reduction='mean'
            # summing the average losses over the
            # batches and then dividing by the number
            # of batches does not give you the true
            # mean loss (though it is at least an
            # unbiased estimate...)
            # Using total rather than sum to account
            # For the last batch being a different size
            total_correct += correct
            total_element += labels.nelement()

            # Confusion matrix
            if evaluate:
                if confusion_mat is None:
                    num_degs = model_output.shape[1]
                    confusion_mat = np.zeros((num_degs, num_degs))
                for label, output in zip(labels.cpu(), model_output.cpu().data.numpy()):
                    confusion_mat[label, np.argmax(output)] += 1

                degradation_type_labels = data[self.formatter["task_labels"][0]]
                for label, pred, deg_label in zip(
                    labels.cpu(),
                    predictions.cpu().data.numpy(),
                    degradation_type_labels.cpu().data.numpy(),
                ):
                    deg_label
                    total_element_per_deg[deg_label] += 1
                    if label == pred:
                        total_correct_per_deg[deg_label] += 1

            log_info = {
                "epoch": epoch,
                "batch": ii,
                "mode": str_code,
                "avg_loss": total_loss / total_element,
                "avg_acc": total_correct / total_element * 100,
            }

            if self.batch_log_freq is not None:
                if ii % self.batch_log_freq == 0:
                    print(
                        ",".join([str(log_info[kk]) for kk in self.log_cols]),
                        file=self.log_file,
                    )

            data_iter.set_postfix(avg_loss=round(log_info["avg_loss"], ndigits=3))

        if self.epoch_log_freq is not None:
            if epoch % self.epoch_log_freq == 0:
                print(
                    ",".join([str(log_info[kk]) for kk in self.log_cols]),
                    file=self.log_file,
                )

        if evaluate:
            confusion_mat /= np.sum(confusion_mat, axis=1, keepdims=True)
            log_info["confusion_mat"] = confusion_mat
            print(f"Accuracy: {total_correct / total_element * 100}")
            print(f"Avg loss: {total_loss / total_element}")
            print(f"Confusion matrix (as [label, output]):\n{confusion_mat}")
            # Not required since in confusion matrix - here for verification / same
            # accross classes
            log_info["avg_acc_per_deg"] = np.array(
                [
                    total_correct_per_deg[deg] / total_element_per_deg[deg]
                    for deg in sorted(total_element_per_deg.keys())
                ]
            )
        return log_info


class ErrorLocationTrainer(BaseTrainer):
    """Trains Task 3 - Error Location. The model provided is expected to be
    an mdtk.pytorch_models.ErrorLocationNet. Expects a DataLoader using an
    mdtk.pytorch_datasets.CommandDataset."""

    def __init__(
        self,
        model,
        criterion,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader = None,
        lr: float = 1e-4,
        betas=(0.9, 0.999),
        weight_decay: float = 0.01,
        with_cuda: bool = True,
        batch_log_freq=None,
        epoch_log_freq=1,
        formatter=None,
        log_file=None,
    ):
        if formatter["task_labels"][2] is None:
            raise NotImplementedError(
                "Formatter " + formatter["name"] + " has not"
                " implemented a ground truth for this task."
            )
        super().__init__(
            model=model,
            criterion=criterion,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            with_cuda=with_cuda,
            batch_log_freq=batch_log_freq,
            epoch_log_freq=epoch_log_freq,
            formatter=formatter,
            log_file=log_file,
        )
        self.log_cols = ["epoch", "batch", "mode", "avg_loss", "avg_acc"]

    def iteration(self, epoch, data_loader, train=True, evaluate=False):
        """
        The loop used for train and test methods. Loops over the provided
        data_loader for one epoch getting only the necessary data for the task.
        If in train mode, a backward pass is performed, updating the parameters
        and saving the model.

        Paremeters
        ----------
        epoch: int
            current epoch index, only used for progress bar

        data_loader: torch.utils.data.DataLoader
            dataloader to get the data from

        train: bool
            Whether to operate in train or test mode. Train mode performs
            backpropagation and saves the model.

        Returns
        -------
        None
        """
        if train:
            self.model.train()
        else:
            self.model.eval()  # informs batchnorm/dropout layers

        str_code = "train" if train else "test"

        # Values to accumulate over the batch
        total_loss = 0
        total_correct = 0
        total_element = 0
        total_positive = 0
        total_positive_labels = 0
        total_true_pos = 0

        total_positive_per_deg = defaultdict(lambda: 0)
        total_positive_labels_per_deg = defaultdict(lambda: 0)
        total_true_pos_per_deg = defaultdict(lambda: 0)
        total_correct_per_deg = defaultdict(lambda: 0)
        total_element_per_deg = defaultdict(lambda: 0)

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc=f"{str_code} epoch {epoch}",
            postfix={"avg_loss": 0},
            bar_format="{l_bar}{bar} batch {r_bar}",
            total=len(data_loader),
        )

        for ii, data in data_iter:
            # N tensors of integers representing (potentially) degraded midi
            input_data = data[self.formatter["deg_label"]].to(self.device)
            # N integers of the labels - 0 assumed to be no degradation
            # N.B. CrossEntropy expects this to be of type long
            labels = (data[self.formatter["task_labels"][2]]).long().to(self.device)
            labels = labels.reshape(labels.shape[0] * labels.shape[1])
            model_output = self.model.forward(input_data)
            model_output = model_output.reshape(
                (model_output.shape[0] * model_output.shape[1], -1)
            )
            loss = self.criterion(model_output, labels)

            # backward pass and optimization only in train
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # values for logging
            predictions = model_output.argmax(dim=-1)
            correct = predictions.eq(labels).sum().item()
            true_pos = (predictions & labels).sum().item()
            total_loss += loss.item() * labels.nelement()
            # N.B. if loss is using reduction='mean'
            # summing the average losses over the
            # batches and then dividing by the number
            # of batches does not give you the true
            # mean loss (though it is at least an
            # unbiased estimate...)
            # Using total rather than sum to account
            # For the last batch being a different size
            total_correct += correct
            total_element += labels.nelement()
            total_positive += predictions.sum().item()
            total_positive_labels += labels.sum().item()
            total_true_pos += true_pos

            log_info = {
                "epoch": epoch,
                "batch": ii,
                "mode": str_code,
                "avg_loss": total_loss / total_element,
                "avg_acc": total_correct / total_element * 100,
            }

            if evaluate:
                # we are considering each time point seperately, so we must repeat
                # the deg_type labels for the length of the sequences
                seq_len = data[self.formatter["task_labels"][2]].shape[1]
                degradation_type_labels = np.repeat(
                    data[self.formatter["task_labels"][0]], seq_len,
                )

                for label, pred, deg_label in zip(
                    labels.cpu().data.numpy(),
                    predictions.cpu().data.numpy(),
                    degradation_type_labels.cpu().data.numpy(),
                ):
                    total_element_per_deg[deg_label] += 1
                    if label == pred:
                        total_correct_per_deg[deg_label] += 1

                    if pred == 1:
                        total_positive_per_deg[deg_label] += 1
                    if label == 1:
                        total_positive_labels_per_deg[deg_label] += 1
                    if pred & label:
                        total_true_pos_per_deg[deg_label] += 1

            if self.batch_log_freq is not None:
                if ii % self.batch_log_freq == 0:
                    print(
                        ",".join([str(log_info[kk]) for kk in self.log_cols]),
                        file=self.log_file,
                    )

            data_iter.set_postfix(avg_loss=round(log_info["avg_loss"], ndigits=3))

        if self.epoch_log_freq is not None:
            if epoch % self.epoch_log_freq == 0:
                print(
                    ",".join([str(log_info[kk]) for kk in self.log_cols]),
                    file=self.log_file,
                )

        if evaluate:
            tp = total_true_pos
            fn = total_positive_labels - tp
            fp = total_positive - tp
            p, r, f = get_f1(tp, fp, fn)
            log_info["p"] = p
            log_info["r"] = r
            log_info["f"] = f
            print(f"P, R, F-measure: {p}, {r}, {f}")

            p_per_deg = []
            r_per_deg = []
            f_per_deg = []
            for deg in sorted(total_element_per_deg.keys()):
                tp = total_true_pos_per_deg[deg]
                fn = total_positive_labels_per_deg[deg] - tp
                fp = total_positive_per_deg[deg] - tp
                p, r, f = get_f1(tp, fp, fn)
                p_per_deg += [p]
                r_per_deg += [r]
                f_per_deg += [f]
            log_info["p_per_deg"] = np.array(p_per_deg)
            log_info["r_per_deg"] = np.array(r_per_deg)
            log_info["f_per_deg"] = np.array(f_per_deg)

            print(f"Avg loss: {total_loss / total_element}")
            log_info["avg_acc_per_deg"] = np.array(
                [
                    total_correct_per_deg[deg] / total_element_per_deg[deg]
                    for deg in sorted(total_element_per_deg.keys())
                ]
            )

        return log_info


class ErrorCorrectionTrainer(BaseTrainer):
    """Trains Task 4 - Error Correction. The model provided is expected to be
    an mdtk.pytorch_models.ErrorCorrectionNet. Expects a DataLoader using an
    mdtk.pytorch_datasets.CommandDataset."""

    def __init__(
        self,
        model,
        criterion,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader = None,
        lr: float = 1e-4,
        betas=(0.9, 0.999),
        weight_decay: float = 0.01,
        with_cuda: bool = True,
        batch_log_freq=None,
        epoch_log_freq=1,
        formatter=None,
        log_file=None,
    ):
        if formatter["task_labels"][3] is None:
            raise NotImplementedError(
                "Formatter " + formatter["name"] + " has not"
                " implemented a ground truth for this task."
            )
        super().__init__(
            model=model,
            criterion=criterion,
            train_dataloader=train_dataloader,
            test_dataloader=test_dataloader,
            lr=lr,
            betas=betas,
            weight_decay=weight_decay,
            with_cuda=with_cuda,
            batch_log_freq=batch_log_freq,
            epoch_log_freq=epoch_log_freq,
            formatter=formatter,
            log_file=log_file,
        )
        self.log_cols = ["epoch", "batch", "mode", "avg_loss", "avg_acc"]

    def iteration(self, epoch, data_loader, train=True, evaluate=False):
        """
        The loop used for train and test methods. Loops over the provided
        data_loader for one epoch getting only the necessary data for the task.
        If in train mode, a backward pass is performed, updating the parameters
        and saving the model.

        Paremeters
        ----------
        epoch: int
            current epoch index, only used for progress bar

        data_loader: torch.utils.data.DataLoader
            dataloader to get the data from

        train: bool
            Whether to operate in train or test mode. Train mode performs
            backpropagation and saves the model.

        Returns
        -------
        None
        """
        if train:
            self.model.train()
        else:
            self.model.eval()  # informs batchnorm/dropout layers

        str_code = "train" if train else "test"

        # Values to accumulate over the batch
        total_loss = 0
        total_correct = 0
        total_element = 0
        total_help = 0
        total_fm = 0
        total_data_points = 0

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(
            enumerate(data_loader),
            desc=f"{str_code} epoch {epoch}",
            postfix={"avg_loss": 0},
            bar_format="{l_bar}{bar} batch {r_bar}",
            total=len(data_loader),
        )

        for ii, data in data_iter:
            input_lengths = np.array(data["deg_len"]) if "deg_len" in data else None
            # N tensors of integers representing (potentially) degraded midi
            input_data = data[self.formatter["deg_label"]].to(self.device)
            # N integers of the labels - 0 assumed to be no degradation
            # N.B. CrossEntropy expects this to be of type long
            labels = (data[self.formatter["task_labels"][3]]).float().to(self.device)
            model_output = self.model.forward(input_data, input_lengths)
            loss = self.criterion(model_output, labels)

            # backward pass and optimization only in train
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # values for logging
            predictions = model_output.round()
            correct = predictions.eq(labels).sum().item()
            total_loss += loss.item() * labels.nelement()
            # N.B. if loss is using reduction='mean'
            # summing the average losses over the
            # batches and then dividing by the number
            # of batches does not give you the true
            # mean loss (though it is at least an
            # unbiased estimate...)
            # Using total rather than sum to account
            # For the last batch being a different size
            total_correct += correct
            total_element += labels.nelement()

            log_info = {
                "epoch": epoch,
                "batch": ii,
                "mode": str_code,
                "avg_loss": total_loss / total_element,
                "avg_acc": total_correct / total_element * 100,
            }

            if evaluate:
                total_data_points += len(input_data)
                for in_data, out_data, clean_data in zip(
                    input_data, model_output, labels
                ):
                    logging.disable(logging.WARNING)
                    deg_df = self.formatter["model_to_df"](
                        in_data.cpu().data.numpy(),
                        min_pitch=MIN_PITCH_DEFAULT,
                        max_pitch=MAX_PITCH_DEFAULT,
                        time_increment=40,
                    )
                    model_out_df = self.formatter["model_to_df"](
                        out_data.round().cpu().data.numpy(),
                        min_pitch=MIN_PITCH_DEFAULT,
                        max_pitch=MAX_PITCH_DEFAULT,
                        time_increment=40,
                    )
                    clean_df = self.formatter["model_to_df"](
                        clean_data.cpu().data.numpy(),
                        min_pitch=MIN_PITCH_DEFAULT,
                        max_pitch=MAX_PITCH_DEFAULT,
                        time_increment=40,
                    )
                    logging.disable(logging.NOTSET)
                    h, f = helpfulness(model_out_df, deg_df, clean_df)
                    total_help += h
                    total_fm += f

            if self.batch_log_freq is not None:
                if ii % self.batch_log_freq == 0:
                    print(
                        ",".join([str(log_info[kk]) for kk in self.log_cols]),
                        file=self.log_file,
                    )

            data_iter.set_postfix(avg_loss=round(log_info["avg_loss"], ndigits=3))

        if self.epoch_log_freq is not None:
            if epoch % self.epoch_log_freq == 0:
                print(
                    ",".join([str(log_info[kk]) for kk in self.log_cols]),
                    file=self.log_file,
                )

        if evaluate:
            helpfulness_val = total_help / total_data_points
            fmeasure = total_fm / total_data_points
            log_info["helpfulness"] = helpfulness_val
            log_info["fmeasure"] = fmeasure
            print(f"Helpfulness: {helpfulness_val}")
            print(f"F-measure: {fmeasure}")
            print(f"Avg loss: {total_loss / total_element}")

        return log_info
