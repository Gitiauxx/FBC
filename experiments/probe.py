import yaml
import copy

from torch.utils.data.dataloader import DataLoader
from sklearn.manifold import TSNE

from source.dataset import *
from source.model import Model
from source.auditors import *
from source.losses import *
from source.utils import get_logger, accuracy_from_logits, conditional_error_rate_from_logits, positive_from_logits

logger = get_logger(__name__)

class Probe(object):

    def __init__(self, autoencoder_path, probes_list,
                 decode=False, beta=None, seed=None, fairness=None, checkpoints=None, task_list=None):

        with open(autoencoder_path, 'r') as stream:
            self.config_autoencoder = yaml.load(stream, Loader=yaml.SafeLoader)

        if beta is not None:
            self.config_autoencoder['beta'] = beta

        if fairness is not None:
            self.config_autoencoder['gamma'] = fairness

        if seed is not None:
            self.config_autoencoder['data']['seed'] = seed

        save = False
        if checkpoints is not None:
            save = True

        self.results = {}
        self.results['autoencoder'] = copy.deepcopy(self.config_autoencoder['net'])
        self.results['loss'] = copy.deepcopy(self.config_autoencoder['loss'])
        self.results['method'] = self.config_autoencoder['experiment']
        self.results['beta'] = self.config_autoencoder['beta']

        self.results['training'] = {}
        self.results['validation'] = {}
        self.results['training']['rec_loss'] = {}
        self.results['validation']['rec_loss'] = {}
        self.results['validation']['bit_rate'] = {}

        self.autoencoder = Model.from_dict(self.config_autoencoder)
        n_epochs = self.config_autoencoder['n_epochs']

        if 'run' in self.config_autoencoder:
            run =  self.config_autoencoder['run']
            logger.info(f'Loading checkpoint {run}')
            checkpoint = torch.load(run, map_location='cpu')
            self.autoencoder.net.load_state_dict(checkpoint['model_state_dict'])

        self.transfer = False
        if 'transfer' in self.config_autoencoder:
            self.transfer = self.config_autoencoder['transfer']
            self.config_autoencoder['data']['transfer'] = True

        self.transfer_small = False
        if 'transfer_small' in self.config_autoencoder:
            self.transfer_small = self.config_autoencoder['transfer_small']
            self.config_autoencoder['data']['transfer_small'] = True

        if 'prun' in self.config_autoencoder:
            prun =  self.config_autoencoder['prun']
            logger.info(f'Loading checkpoint {prun}')
            checkpoint = torch.load(prun, map_location='cpu')
            self.autoencoder.pmodel.load_state_dict(checkpoint['model_state_dict'])

        self.probes_list = probes_list

        if self.transfer | self.transfer_small | (task_list is not None):
            self.task_list = task_list
        else:
            self.taks_list = probes_list

        self.decode = decode

        dataname = self.config_autoencoder['data'].pop('name')
        self.results['dataname'] = dataname

        train_dset = globals()[dataname].from_dict(self.config_autoencoder['data'], type='train')
        test_dset = globals()[dataname].from_dict(self.config_autoencoder['data'], type='test')
        validate_dset = globals()[dataname].from_dict(self.config_autoencoder['data'], type='validate')

        train_loader = DataLoader(train_dset,
                                  batch_size=self.config_autoencoder['batch_size'],
                                  shuffle=True)
        validate_loader = DataLoader(validate_dset,
                                 batch_size=self.config_autoencoder['batch_size'])

        logger.info(f'Train autoencoder to generate representations')
        self.autoencoder.train(train_loader, validate_loader, n_epochs, self.results,
                                   save=save, chkpt_dir=checkpoints)
        self.autoencoder.net.eval()

        rec_loss, mask_loss, accuracy, bitrate = self.autoencoder.eval(validate_loader)
        self.results['validation']['bit_rate'] = bitrate.item()
        self.results['validation']['rec_loss_final'] = rec_loss.item()

        self.device = self.autoencoder.device
        self.nclass = self.config_autoencoder['nclass_outcome']

        if self.transfer:
            self.nclass_sensitive = self.config_autoencoder['nclass_sensitive_transfer']
        else:
            self.nclass_sensitive = self.config_autoencoder['nclass_sensitive']

        self.train_rep_loader = self.generate_representation(train_dset, shuffle=True)
        self.test_rep_loader = self.generate_representation(test_dset, shuffle=True)
        self.validate_rep_loader = self.generate_representation(validate_dset, shuffle=True)

    def generate_representation(self, dset, shuffle=True):
        """
        Using the representation mapping from the autoencoder,
        generate representation z or (if self.decode=True), decoder(z)
        :return:
        """
        rep_generator = RepDataset(dset, self.autoencoder, device=self.device, transfer=self.transfer)
        rep_loader = DataLoader(rep_generator, shuffle=shuffle,
                                batch_size=self.config_autoencoder['batch_size'])

        self.zdim = rep_generator.zdim
        return rep_loader

    def generate_new_data(self, size=100):
        """
        Generate data using sensitive and then random sensitive attribute
        :param size: number of samples
        :return:
        """
        count = 0
        output_list = []
        output_rand_list = []
        x_list = []

        for _, batch in enumerate(self.test_rep_loader):
            if count > size:
                break

            x = batch['input'].to(self.autoencoder.device)
            s = batch['sensitive']

            output, z, _ = self.autoencoder.net.forward(x, torch.ones_like(s))
            output_random, _, _ = self.autoencoder.net.forward(x, torch.zeros_like(s))

            output = output.detach().cpu().numpy()
            output_list.append(output)
            x_list.append(x.cpu().numpy())

            output_random = output_random.detach().cpu().numpy()
            output_rand_list.append(output_random)

            count += s.shape[0]

        out = np.concatenate(output_list)
        out_rand = np.concatenate(output_rand_list)
        x = np.concatenate(x_list)

        return out, out_rand, x

    def generate_tnse(self, ncomponent=2, perplexity=20):
        """
        Generate a t-sne plot of the latent space colored by sensitive attribute
        :param ncomponent:
        :param perplexity:
        :return:
        """
        tsne = TSNE(n_components=ncomponent, verbose=1, perplexity=perplexity, n_iter=1000)

        z_list = []
        sensitive_list = []
        outcome_list = []

        for _, batch in enumerate(self.test_rep_loader):
            input_mean = batch['input_mean'].to(self.autoencoder.device)
            sensitive = batch['sensitive']
            outcome = batch['target'].cpu().numpy()

            z = input_mean.cpu().numpy()


            sensitive_list.append(sensitive.detach().numpy())
            z_list.append(z)
            outcome_list.append(outcome)

        z = np.concatenate(z_list)
        s = np.concatenate(sensitive_list, axis=0)

        if len(s.shape) > 1:
            s = np.argmax(s, -1)
        else:
            s = s.ravel()

        o = np.concatenate(outcome_list, axis=0).ravel()

        tsne_results = tsne.fit_transform(z)
        return tsne_results, s, o

    def probe_sensitive(self):
        """
        for each path in probe_list construct the corresponding probe
        and train it to predict sensitive attribute
        :return:
        """
        if 'probes' not in self.results:
            self.results['probes'] = {}
        for i, config_probe_path in enumerate(self.probes_list):
            self.results['probes'][i] = {}

            with open(config_probe_path, 'r') as stream:
                config_probe = yaml.load(stream, Loader=yaml.SafeLoader)

            self.results['probes'][i]["depth"] = config_probe["classifier"]["depth"]
            self.results['probes'][i]["width"] = config_probe["classifier"]["width"]

            logger.info(f'Probing with model {config_probe["classifier"]["name"]} of '
                        f'width {config_probe["classifier"]["width"]} and '
                        f'depth {config_probe["classifier"]["depth"]}')

            n_epochs = config_probe['n_epochs']

            if self.autoencoder.method == 'compression':
                config_probe['classifier']['zdim'] = self.config_autoencoder['net']['zk']  * self.config_autoencoder['net']['k']
            else:
                config_probe['classifier']['zdim'] = self.config_autoencoder['net']['zdim']

            config_probe['classifier']['nclass'] = self.nclass_sensitive

            probe = ProbeFairness.from_dict(config_probe)
            probe.train(self.validate_rep_loader, self.test_rep_loader, n_epochs,
                        nclass=self.nclass_sensitive, writer=self.results['probes'])

            probe_loss, accuracy, demographic_parity = probe.eval(self.test_rep_loader, nclass=self.nclass_sensitive)
            self.results['probes'][i]['accuracy'] = accuracy.item()
            self.results['probes'][i]['demographic_parity'] = demographic_parity.item()
            self.results['probes'][i]['loss'] = probe_loss.item()

    def classify_from_representation(self, transfer=False):
        """
        for each path in probe_list construct the corresponding classifier
        and train it to predict an outcome as defined in the corresponding
        dataset
        :return:
        """
        if 'classifier' not in self.results:
            self.results['classifier'] = {}

        for i, config_classifier_path in enumerate(self.task_list):
            tag = i
            if transfer:
                tag = f'{i}_transfer'
            self.results['classifier'][tag] = {}

            with open(config_classifier_path, 'r') as stream:
                config_classifier = yaml.load(stream, Loader=yaml.SafeLoader)

            self.results['classifier'][tag]["depth"] = config_classifier["classifier"]["depth"]
            self.results['classifier'][tag]["with"] = config_classifier["classifier"]["width"]

            logger.info(f'Probing with model {config_classifier["classifier"]["name"]} of '
                        f'width {config_classifier["classifier"]["width"]} and '
                        f'depth {config_classifier["classifier"]["depth"]}')

            n_epochs = config_classifier['n_epochs']
            if self.decode:
                config_classifier['classifier']['zdim'] = self.config_autoencoder['net']['input_dim']
            else:
                if self.autoencoder.method == 'compression':
                    config_classifier['classifier']['zdim'] = self.config_autoencoder['net']['zk'] * self.config_autoencoder['net']['k']
                else:
                    config_classifier['classifier']['zdim'] = self.config_autoencoder['net']['zdim']
            config_classifier['device'] = self.autoencoder.device
            config_classifier['classifier']['nclass'] = self.nclass

            probe = ProbePareto.from_dict(config_classifier)
            probe.train(self.validate_rep_loader, self.test_rep_loader, n_epochs,
                        self.autoencoder.net,
                        self.results['classifier'][tag], nclass=self.nclass, transfer=transfer)
            accuracy, demographic_parity, eo = probe.eval(self.test_rep_loader,
                                                      self.autoencoder.net, nclass=self.nclass,
                                                      transfer=transfer)
            self.results['classifier'][tag]['accuracy'] = accuracy.item()
            self.results['classifier'][tag]['demographic_parity'] = demographic_parity.item()
            self.results['classifier'][tag]['eo'] = eo.item()


class ProbeFairness(object):
    """
    Generate a downstream probe/user/test function train to predict sensitive
    attribute from representation
    """

    def __init__(self, classifier, loss, learning_rate=0.01, device='cpu'):

        device = torch.device(device)
        self.classifier = classifier.to(device)
        self.loss = loss
        self.device = device

        self.optimizer = torch.optim.Adam(self.classifier.parameters(), lr=learning_rate, betas=(0.5, 0.999),
                                          weight_decay=1e-5)

    @classmethod
    def from_dict(cls, config_dict):
        """
        Create a model input configuration from a eval config dictionary

        Parameters
        ----------
        config_dict : configuration dictionary

        """
        name_net = config_dict['classifier'].pop('name')
        classifier = globals()[name_net].from_dict(config_dict['classifier'])

        name_loss = config_dict['loss'].pop('name')
        loss = globals()[name_loss].from_dict(config_dict['loss'])

        learning_rate = config_dict['learning_rate']
        device = config_dict['device']

        return cls(classifier, loss, learning_rate=learning_rate, device=device)

    def optimize_parameters(self, x, sensitive, **kwargs):
        """
        implement one forward pass and one backward propagation pass
        Parameters
        ----------
        x: (B, zdim)
        target (B, 1)

        Returns
        -------
        """
        prelogits = self.classifier.forward(x)
        loss = self.loss.forward(sensitive, prelogits)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

    def train(self, train_loader, validation_loader, n_epochs, nclass=1, writer=None):

        writer['validation'] = {}

        for epoch in range(n_epochs):
            train_loss = 0

            for _, batch in enumerate(train_loader):
                input = batch['input_mean'].to(self.device)
                sensitive = batch['sensitive'].to(self.device)

                if nclass > 1:
                    sensitive = torch.argmax(sensitive, -1)

                loss = self.optimize_parameters(input, sensitive)
                train_loss += loss.detach().cpu() * len(input) / len(train_loader.dataset)

            logger.info(f'Epoch: {epoch} Train loss: {train_loss}')

            if (epoch % 5 == 0) | (epoch == n_epochs - 1):
                val_loss, accuracy, demographic_parity = self.eval(validation_loader, nclass=nclass)
                logger.info(f'Epoch: {epoch} Probe Accuracy: {accuracy}')
                logger.info(f'Epoch: {epoch} Probe DP:{demographic_parity}')
                logger.info(f'Epoch: {epoch} Validation loss: {val_loss}')

                writer['validation'][epoch] = {}
                writer['validation'][epoch]['accuracy'] = accuracy.item()

    def eval(self, data_loader, nclass=1):
        """
        Measure reconstruction loss for self.net and accuracy and demographic parity for self.classifier
        :param data_loader: torch.DataLoader
        :return: probe loss, accuracy, demographic_parity
        """

        probe_loss = 0
        accuracy = 0
        ber0 = 0
        ber1 = 0
        ber = 0
        s0 = 0
        s1 = 0

        for _, batch in enumerate(data_loader):
            input = batch['input_mean'].to(self.device)
            sensitive = batch['sensitive'].to(self.device)

            s = sensitive
            if nclass > 1:
                s = torch.argmax(sensitive, -1)

            output = self.classifier.forward(input)

            loss = self.loss.forward(s, output)
            probe_loss += loss.detach().cpu() * len(input) / len(data_loader.dataset)

            acc = accuracy_from_logits(s, output, nclass=nclass)
            accuracy += acc * len(input) / len(data_loader.dataset)

            if nclass == 1:
                p0, p1 = conditional_error_rate_from_logits(output, s)
                sensitive_1 = s.mean().detach().cpu()
                ber1 += p1 * sensitive_1 * len(input)
                ber0 += p0 * (1 - sensitive_1) * len(input)
                s1 += sensitive_1 * len(input)
                s0 += (1 - sensitive_1) * len(input)

            else:
                predictions = torch.argmax(output, -1).detach()
                error = (predictions != s).float()
                ber += torch.sum(error[..., None] * sensitive, 0).cpu()

                s1 += sensitive.sum(0).cpu()
                s0 += (1 - sensitive).sum(0).cpu()

        if nclass == 1:
            ber1 = ber1 / s1
            ber0 = ber0 / s0
            demographic_parity = 1 - 2 * (ber0 + ber1) / 2
        else:
            ber = ber / s1

            demographic_parity = torch.mean(ber)

        return probe_loss, accuracy, demographic_parity

class ProbePareto(ProbeFairness):
    """
    Generate a downstream user with a classification task.
    from a sample of the representation distribution.
    It is used to generate Pareto-front, i.e. accuracy/demographic parity trade-off
    """

    def train(self, train_loader, validation_loader, n_epochs, generator, writer, nclass=1, transfer=False):

        writer['validation'] = {}

        for epoch in range(n_epochs):
            train_loss = 0

            for _, batch in enumerate(train_loader):
                x = batch['input'].to(self.device)
                input = batch['input_mean'].to(self.device)

                outcome = batch['target'].to(self.device)

                loss = self.optimize_parameters(input, outcome)
                train_loss += loss.detach().cpu() * x.shape[0] / len(train_loader.dataset)

            logger.info(f'Epoch: {epoch} Train loss: {train_loss}')

            if (epoch % 5 == 0) | (epoch == n_epochs - 1):
                accuracy, demographic_parity, eo = self.eval(validation_loader,
                                                         generator, nclass=nclass)
                logger.info(f'Epoch: {epoch} Classifier Accuracy: {accuracy}')
                logger.info(f'Epoch: {epoch} Classifier DP:{demographic_parity}')
                logger.info(f'Epoch: {epoch} Classifier EO:{eo}')

                writer['validation'][epoch] = {}
                writer['validation'][epoch]['accuracy'] = accuracy.item()
                writer['validation'][epoch]['dp'] = demographic_parity.item()
                writer['validation'][epoch]['pp'] = eo.item()

    def eval(self, data_loader, generator, nclass=1, transfer=False):
        """
        Measure accuracy and demographic parity for self.classifier
        :param data_loader: torch.DataLoader
        :return: accuracy, demographic_parity
        """

        accuracy = 0
        P0 = 0
        P1 = 0
        s0 = 0
        s1 = 0
        PP1 = 0
        SP1 = 0
        PP0 = 0
        SP0 = 0

        for _, batch in enumerate(data_loader):
            sensitive = batch['sensitive'].to(self.device)

            outcome = batch['target'].to(self.device)

            input = batch['input_mean'].to(self.device)
            output = self.classifier.forward(input)

            acc = accuracy_from_logits(outcome, output, nclass=nclass)
            accuracy += acc * len(input) / len(data_loader.dataset)

            sensitive_1 = sensitive.mean(0).detach()

            if nclass == 1:
                p0, p1, pp1, pp0 = positive_from_logits(output, sensitive, y=outcome)

                P1 += p1 * sensitive_1 * len(input)
                P0 += p0 * (1 - sensitive_1) * len(input)
                PP1 += pp1 \
                       #* (torch.argmax(sensitive, -1) * outcome).sum(0)
                PP0 += pp0 \
                       #* ((1 - torch.argmax(sensitive, -1)) * outcome).sum(0)

            else:
                predictions = torch.argmax(output, -1).detach().cpu()
                pred_one_zero = torch.zeros_like(output)
                pred_one_zero[range(pred_one_zero.shape[0]), predictions] = 1
                pred_one_zero = pred_one_zero[..., None]
                sensitive = sensitive[:, None, ...]

                p0 = torch.sum(pred_one_zero * sensitive, 0).cpu()
                p1 = torch.sum(pred_one_zero * (1 - sensitive), 0).cpu()

                P0 += p0
                P1 += p1

            s1 += sensitive_1 * len(input)
            s0 += (1 - sensitive_1) * len(input)

            SP1 += (torch.argmax(sensitive, -1) * (1 - outcome)).sum(0)
            SP0 += (torch.argmax(sensitive, -1) *  (1 - outcome)).sum(0)

        if nclass > 1:
            s1 = s1[None, ...]
            s0 = s0[None, ...]

        P1 = P1 / s1
        P0 = P0 / s0
        PP1 = PP1 / SP1
        PP0 = PP0 / SP0

        demographic_parity = torch.mean(torch.abs(P1 - P0)).cpu()
        eo = torch.mean(torch.abs(PP1 - PP0)).cpu()

        return accuracy, demographic_parity, eo
