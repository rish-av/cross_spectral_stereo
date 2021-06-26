from torch.utils.tensorboard import SummaryWriter
import datetime
import torch
import torch.utils.data as data
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from os.path import join


def _get_optimizer(net,lr=0.0001,weight_decay=0.0):

    '''
        returns optimizer, needs to be made more generic
    '''
    return torch.optim.Adam(net.parameters(),lr=lr,weight_decay=weight_decay)


def get_summary_writer(rootdir):

    return SummaryWriter(join(rootdir,get_log_dir()))


def get_log_dir():
    '''
    New log dir at every run according to the time at that point in time.
    '''
    now = datetime.datetime.now()
    return "logs/run-%d-%d-%d-%d-%d-%d"%(now.year,now.month,now.day,now.hour,now.minute,now.second)

def tensor_to_image(tensors):
    '''
    takes tensor(s) and coverts it into numpy array(s)
    returns: a list of images if B > 1 else the individual image
    '''
    shape = tensors.shape
    if len(shape) > 3:
        b,c,_,_ = shape

        imgs = []
        for img in tensors:
            if c == 1:
                np_img = img.squeeze(0).cpu().numpy()
            else:
                np_img = img.permute(1,2,0).cpu().numpy()

            imgs.append(np_img)
        return imgs
    else:
        c,_,_ = shape
        if c==1:
            return tensors.squeeze(0).cpu().numpy()
        else:
            return tensors.permute(1,2,0).cpu().numpy()

def _log_grads(net,writer,step):

    '''
    log histogram of gradients
    '''

    for tag, value in net.named_parameters():
        tag = tag.replace('.', '/')
        writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), step)
        writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(),step)

def _log(writer,step,scalars=None,images=None):

    '''
    log everything on tensorboard
    writer: summary_write object
    scalars: dict of scalar values to write: eg. {"Train/Loss_a":55.0,"Train/Loss_b":66.6}
    images: dicts of images: eg. {"img1":image tensor1, "img2":image tensor2}  
    modify to add more functionality
    '''
    if scalars:
        for tag, scalar in scalars.items():    
            writer.add_scalar(tag,scalar,step)

    if images:
        for tag, image in images.items():
            writer.add_image(tag,image,step)

def _save_weights(net,savepath):
    '''
    save torch checkpoints
    '''

    torch.save(net.state_dict(),savepath)


def _load_weights(net,loadpath):
    '''
    load weights
    '''

    net.load_state_dict(torch.load(loadpath))

def _kill_grad(x):

    x.requires_grad = False

def _get_loader(dataset,batch_size,num_workers=4,collate_fn=None,shuffle=False,sampler=None):
    '''
    returns dataloader for a dataset object
    add collate_fn support
    '''
    return data.DataLoader(dataset,batch_size,shuffle=shuffle,collate_fn=collate_fn,sampler=sampler)


def _split(joint_dataset,val_percent, sampler=SubsetRandomSampler,random_seed=42):

    '''
    function useful there is no explicit valitdation scripts available
    joint_dataset = train + val items
    returns train and validation samplers based on sampling strategy
    '''
    dataset_size = len(joint_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(val_percent * dataset_size))
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    return train_sampler, valid_sampler


def block_grad(net):

    for name, param in net.named_parameters():
        param.requires_grad = False


def start_grad(net):

    for name, param in net.named_parameters():
        param.requires_grad = True

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def get_device():

    if torch.cuda.is_available():
        return 'cuda:0'
    else:
        return 'cpu:0'
