import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import numpy as np
import tqdm



class BaseTrainer:
    """Provides methods to train pytorch models. Adapted from:
    https://github.com/codertimo/BERT-pytorch/blob/master/bert_pytorch/trainer/pretrain.py"""
    def __init__(self, model, criterion, train_dataloader: DataLoader,
                 test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999),
                 weight_decay: float=0.01, with_cuda: bool=True,
                 batch_log_freq=None, epoch_log_freq=1, formatter=None):
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
        """

        # Setup cuda device for BERT training, argument -c, --cuda should be true
        self.device = torch.device("cpu")
        cuda_condition = torch.cuda.is_available()
        if with_cuda:
            if not cuda_condition:
                print('You set with_cuda to True, but cuda was not available')
                print('Using cpu')
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
        self.optimizer = Adam(self.model.parameters(), lr=lr, betas=betas,
                              weight_decay=weight_decay)

        self.criterion = criterion

        self.batch_log_freq = batch_log_freq if batch_log_freq else np.inf
        self.epoch_log_freq = epoch_log_freq if epoch_log_freq else np.inf
        
        self.formatter = formatter
        
        print("Total Parameters:", 
              sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        with torch.no_grad():
            self.iteration(epoch, self.test_data, train=False)

    def iteration(self, epoch, data_loader, train=True):
        """This must be overwritten by classes inheriting this"""
        raise NotImplementedError()

    def save(self, epoch, file_path=None):
        """
        Saving the current model on file_path
        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        if file_path is None:
            file_path = "trained.model"
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.model.cpu(), output_path)
        self.model.to(self.device)
        print(f"EP:{epoch} Model saved {output_path}")
        return output_path


class ErrorDetectionTrainer(BaseTrainer):
    """Trains Task 1 - Error detection. The model provided is expected to be
    an mdtk.pytorch_models.ErrorDetectionNet. Expects a DataLoader using an
    mdtk.pytorch_datasets.CommandDataset."""
    def __init__(self, model, criterion, train_dataloader: DataLoader,
                 test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999),
                 weight_decay: float=0.01, with_cuda: bool=True,
                 batch_log_freq=None, epoch_log_freq=1, formatter=None):
        if formatter['task_labels'][0] is None:
            raise NotImplementedError('Formatter ' + formatter['name'] + ' has not'
                                      ' implemented a ground truth for this task.')
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
            formatter=formatter
        )

    def iteration(self, epoch, data_loader, train=True):
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

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc=f"EP_{str_code}: {epoch}",
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        # Values to accumulate over the batch
        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        
        for ii, data in data_iter:
            # N tensors of integers representing (potentially) degraded midi
            input_data = data[self.formatter['deg_label']].to(self.device)
            # N integers of the labels - 0 assumed to be no degradation
            # N.B. CrossEntropy expects this to be of type long
            labels = (data[self.formatter['task_labels'][0]] > 0).long().to(self.device)
            model_output = self.model.forward(input_data)
            loss = self.criterion(model_output, labels)
            
            # backward pass and optimization only in train
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # values for logging
            correct = model_output.argmax(dim=-1).eq(labels).sum().item()
            avg_loss += loss.item()  # N.B. if loss is using reduction='mean'
                                     # summing the average losses over the
                                     # batches and then dividing by the number
                                     # of batches does not give you the true
                                     # mean loss (though it is at least an
                                     # unbiased estimate...)
            total_correct += correct
            total_element += labels.nelement()

            post_fix = {
                "epoch": epoch,
                "iter": ii,
                "avg_loss": avg_loss / (ii + 1),
                "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }
            
            if ii % self.batch_log_freq == 0:
                data_iter.write(str(post_fix))
        
        if epoch % self.epoch_log_freq == 0:
            data_iter.write(str(post_fix))
#        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), "total_acc=",
#              total_correct * 100.0 / total_element)



class ErrorClassificationTrainer(BaseTrainer):
    """Trains Task 2 - Error classification. The model provided is expected to be
    an mdtk.pytorch_models.ErrorClassificationNet. Expects a DataLoader using an
    mdtk.pytorch_datasets.CommandDataset."""
    def __init__(self, model, criterion, train_dataloader: DataLoader,
                 test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999),
                 weight_decay: float=0.01, with_cuda: bool=True,
                 batch_log_freq=None, epoch_log_freq=1, formatter=None):
        if formatter['task_labels'][1] is None:
            raise NotImplementedError('Formatter ' + formatter['name'] + ' has not'
                                      ' implemented a ground truth for this task.')
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
            formatter=formatter
        )

    def iteration(self, epoch, data_loader, train=True):
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

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc=f"EP_{str_code}: {epoch}",
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        # Values to accumulate over the batch
        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        
        for ii, data in data_iter:
            # N tensors of integers representing (potentially) degraded midi
            input_data = data[self.formatter['deg_label']].to(self.device)
            # N integers of the labels - 0 assumed to be no degradation
            # N.B. CrossEntropy expects this to be of type long
            labels = (data[self.formatter['task_labels'][1]]).long().to(self.device)
            model_output = self.model.forward(input_data)
            loss = self.criterion(model_output, labels)
            
            # backward pass and optimization only in train
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # values for logging
            correct = model_output.argmax(dim=-1).eq(labels).sum().item()
            avg_loss += loss.item()  # N.B. if loss is using reduction='mean'
                                     # summing the average losses over the
                                     # batches and then dividing by the number
                                     # of batches does not give you the true
                                     # mean loss (though it is at least an
                                     # unbiased estimate...)
            total_correct += correct
            total_element += labels.nelement()

            post_fix = {
                "epoch": epoch,
                "iter": ii,
                "avg_loss": avg_loss / (ii + 1),
                "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }
            
            if ii % self.batch_log_freq == 0:
                data_iter.write(str(post_fix))
        
        if epoch % self.epoch_log_freq == 0:
            data_iter.write(str(post_fix))
#        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), "total_acc=",
#              total_correct * 100.0 / total_element)



class ErrorIdentificationTrainer(BaseTrainer):
    """Trains Task 3 - Error identification. The model provided is expected to be
    an mdtk.pytorch_models.ErrorIdentificationNet. Expects a DataLoader using an
    mdtk.pytorch_datasets.CommandDataset."""
    def __init__(self, model, criterion, train_dataloader: DataLoader,
                 test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999),
                 weight_decay: float=0.01, with_cuda: bool=True,
                 batch_log_freq=None, epoch_log_freq=1, formatter=None):
        if formatter['task_labels'][2] is None:
            raise NotImplementedError('Formatter ' + formatter['name'] + ' has not'
                                      ' implemented a ground truth for this task.')
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
            formatter=formatter
        )

    def iteration(self, epoch, data_loader, train=True):
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

        # Setting the tqdm progress bar
        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc=f"EP_{str_code}: {epoch}",
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        # Values to accumulate over the batch
        avg_loss = 0.0
        total_correct = 0
        total_element = 0
        
        for ii, data in data_iter:
            # N tensors of integers representing (potentially) degraded midi
            input_data = data[self.formatter['deg_label']].to(self.device)
            # N integers of the labels - 0 assumed to be no degradation
            # N.B. CrossEntropy expects this to be of type long
            labels = (data[self.formatter['task_labels'][2]]).long().to(self.device)
            model_output = self.model.forward(input_data)
            loss = self.criterion(model_output, labels)
            
            # backward pass and optimization only in train
            if train:
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # values for logging
            correct = model_output.argmax(dim=-1).eq(labels).sum().item()
            avg_loss += loss.item()  # N.B. if loss is using reduction='mean'
                                     # summing the average losses over the
                                     # batches and then dividing by the number
                                     # of batches does not give you the true
                                     # mean loss (though it is at least an
                                     # unbiased estimate...)
            total_correct += correct
            total_element += labels.nelement()

            post_fix = {
                "epoch": epoch,
                "iter": ii,
                "avg_loss": avg_loss / (ii + 1),
                "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }
            
            if ii % self.batch_log_freq == 0:
                data_iter.write(str(post_fix))
        
        if epoch % self.epoch_log_freq == 0:
            data_iter.write(str(post_fix))
#        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), "total_acc=",
#              total_correct * 100.0 / total_element)



class ErrorCorrectionTrainer(BaseTrainer):
    """
    """
    def __init__(self):
        super().__init__()

    def iteration(self, epoch, data_loader, train=True):
        raise NotImplementedError()
