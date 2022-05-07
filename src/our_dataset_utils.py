import torch
import torchvision.transforms as transforms
from torchvision import datasets

def get_dataloader(name,normalizer,bs, is_large_image=False):
    if name == 'cifar10':
        dataloader = cifar10(normalizer,bs, is_large_image=is_large_image)
    elif name == 'cifar100':
        dataloader = cifar100(normalizer,bs, is_large_image=is_large_image)
    elif name == 'mnist':
        dataloader = mnist(normalizer,bs, is_large_image=is_large_image)
    elif name == 'kmnist':
        dataloader = kmnist(normalizer,bs, is_large_image=is_large_image)
    elif name == 'fasionmnist':
        dataloader = fasionmnist(normalizer,bs, is_large_image=is_large_image)
    elif name == 'svhn':
        dataloader = svhn(normalizer,bs, is_large_image=is_large_image)
    elif name == 'stl10':
        dataloader = stl10(normalizer,bs, is_large_image=is_large_image)
    elif name == 'dtd':
        dataloader = dtd(normalizer,bs, is_large_image=is_large_image)
    elif name == 'place365':
        dataloader = place365(normalizer,bs, is_large_image=is_large_image)
    elif name == 'lsun':
        dataloader = lsun(normalizer,bs, is_large_image=is_large_image)
    elif name == 'lsunR':
        dataloader = lsunR(normalizer,bs, is_large_image=is_large_image)
    elif name == 'isun':
        dataloader = isun(normalizer,bs, is_large_image=is_large_image)
    elif name == 'celebA':
        dataloader = celebA(normalizer,bs, is_large_image=is_large_image)
    elif name == 'TinyC':
        dataloader = tinyC(normalizer, bs, is_large_image=is_large_image)
    elif name == 'TinyR':
        dataloader = tinyR(normalizer, bs, is_large_image=is_large_image)
    elif name == 'imagenet30':
        dataloader = imagenet30(normalizer, bs, is_large_image=is_large_image)
    else:
        print('the dataset is not used in this project')
        return None
    return dataloader


def cifar10(normalizer,bs, is_large_image=False):
    transform_cifar10 = transforms.Compose([transforms.ToTensor(),
                                            normalizer
                                            ])
    if is_large_image:
        transform_cifar10 = transforms.Compose([
            transforms.Resize((254, 254)),
            transforms.ToTensor(),
            normalizer
        ])
    dataloader = torch.utils.data.DataLoader(
                datasets.CIFAR10('../data/cifar10',
                                train=False, 
                                download=True,
                                transform=transform_cifar10),
                batch_size=bs, 
                shuffle=False, 
                num_workers=1, 
                pin_memory=True)
    return dataloader

def celebA(normalizer,bs, is_large_image=False):
    transformer = transforms.Compose([transforms.Resize(32),
                                      transforms.ToTensor(),
                                      normalizer
                                      ])
    if is_large_image:
        transformer = transforms.Compose([transforms.Resize((254, 254)),
                                          transforms.ToTensor(),
                                          normalizer
                                          ])
    dataloader = torch.utils.data.DataLoader(
                datasets.CelebA('../data/celebA',
                                split='test', 
                                download=True,
                                transform=transformer),
                batch_size=bs, 
                shuffle=False, 
                num_workers=1, 
                pin_memory=True)
    return dataloader

def cifar100(normalizer,bs, is_large_image=False):
    transform_cifar100 = transforms.Compose([transforms.ToTensor(),
                                            normalizer
                                            ])
    if is_large_image:
        transform_cifar100 = transforms.Compose([transforms.Resize((254, 254)),
                                                 transforms.ToTensor(),
                                                 normalizer
                                                 ])
    dataloader = torch.utils.data.DataLoader(
                datasets.CIFAR100('../data/cifar100',
                                 train=False, 
                                 download=True,
                                 transform=transform_cifar100),
                batch_size=bs, 
                shuffle=False, 
                num_workers=1, 
                pin_memory=True)
    return dataloader
def mnist(normalizer,bs, is_large_image=False):
    transformer = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                      transforms.Pad(padding=2),
                                      transforms.ToTensor(),
                                      normalizer
                                      ])
    if is_large_image:
        transformer = transforms.Compose(
            [transforms.Grayscale(num_output_channels=3),
             transforms.Resize((254, 254)),
             transforms.ToTensor(),
             normalizer
             ])
    dataloader = torch.utils.data.DataLoader(
                datasets.MNIST('../data/mnist',
                                train=False, 
                                download=True,
                                transform=transformer),
                batch_size=bs, 
                shuffle=False, 
                num_workers=1, 
                pin_memory=True)
    return dataloader
def kmnist(normalizer,bs,is_large_image=False):
    transformer = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                      transforms.Pad(padding=2),
                                      transforms.ToTensor(),
                                      normalizer
                                      ])
    if is_large_image:
        transformer = transforms.Compose(
            [
             transforms.Grayscale(num_output_channels=3),
             transforms.Resize((254, 254)),
             transforms.ToTensor(),
             normalizer
             ])

    dataloader = torch.utils.data.DataLoader(
                datasets.KMNIST('../data/kmnist',
                                train=False, 
                                download=True,
                                transform=transformer),
                batch_size=bs, 
                shuffle=False, 
                num_workers=1, 
                pin_memory=True)
    return dataloader
def fasionmnist(normalizer,bs, is_large_image=False):
    transformer = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                      transforms.Pad(padding=2),
                                      transforms.ToTensor(),
                                      normalizer
                                      ])
    if is_large_image:
        transformer = transforms.Compose(
            [transforms.Grayscale(num_output_channels=3),
             transforms.Resize((254, 254)),
             transforms.ToTensor(),
             normalizer
             ])
    dataloader = torch.utils.data.DataLoader(
                datasets.FashionMNIST('../data/fasionmnist',
                                train=False, 
                                download=True,
                                transform=transformer),
                batch_size=bs, 
                shuffle=False, 
                num_workers=1, 
                pin_memory=True)
    return dataloader
'''
def svhn(normalizer,bs):
    transformer = transforms.Compose([transforms.ToTensor(),
                                      normalizer
                                      ])
    dataloader = torch.utils.data.DataLoader(
                datasets.SVHN('data/svhn', 
                              split='test', 
                              download=True,
                              transform=transformer),
                batch_size=bs, 
                shuffle=False, 
                num_workers=1, 
                pin_memory=True)
    return dataloader
'''
def stl10(normalizer,bs, is_large_image=False):
    transformer = transforms.Compose([transforms.Resize(32),
                                      transforms.ToTensor(),
                                      normalizer
                                      ])
    if is_large_image:
        transformer = transforms.Compose([transforms.Resize((254, 254)),
                                          transforms.ToTensor(),
                                          normalizer
                                          ])
    dataloader = torch.utils.data.DataLoader(
                datasets.STL10('../data/STL10',
                                split='test',
                                folds=0,
                                download=(True),
                                transform=transformer),
                batch_size=bs, 
                shuffle=False, 
                num_workers=1, 
                pin_memory=True)
    return dataloader

def svhn(normalizer,bs, is_large_image=False):
    import my_utils.svhn_loader as svhn
    transformer = transforms.Compose([transforms.ToTensor(),
                                      normalizer
                                      ])
    if is_large_image:
        transformer = transforms.Compose([
            transforms.Resize((254, 254)),
            transforms.ToTensor(),
            normalizer
        ])
    info_svhn_dataset = svhn.SVHN('../data/svhn', split='test',
                                  transform=transformer, download=True)
    dataloader = torch.utils.data.DataLoader(
                info_svhn_dataset,
                batch_size=bs, 
                shuffle=False, 
                num_workers=1, 
                pin_memory=True)
    return dataloader

def dtd(normalizer,bs, is_large_image=False):
    import torchvision
    transformer = transforms.Compose([transforms.Resize(32),
                                      transforms.CenterCrop(32),#32*40 exist
                                      transforms.ToTensor(),
                                      normalizer
                                      ])
    if is_large_image:
        transformer = transforms.Compose([transforms.Resize((254, 254)),
                                          transforms.CenterCrop(254),
                                          # 32*40 exist
                                          transforms.ToTensor(),
                                          normalizer
                                          ])
    info_dtd_dataset = torchvision.datasets.ImageFolder(root="data/dtd/images",
                                                        transform=transformer)
    dataloader = torch.utils.data.DataLoader(
                info_dtd_dataset,
                batch_size=bs, 
                shuffle=False, 
                num_workers=1, 
                pin_memory=True)
    return dataloader
def place365(normalizer,bs, is_large_image=False):
    import torchvision
    transformer = transforms.Compose([transforms.Resize(32),
                                      #transforms.CenterCrop(32),
                                      transforms.ToTensor(),
                                      normalizer
                                      ])
    if is_large_image:
        transformer = transforms.Compose([transforms.Resize((254, 254)),
                                          # transforms.CenterCrop(32),
                                          transforms.ToTensor(),
                                          normalizer
                                          ])
    info_place365_dataset = torchvision.datasets.ImageFolder(root="../data/places365/test_subset",
                                                             transform=transformer)
    dataloader = torch.utils.data.DataLoader(
                info_place365_dataset,
                batch_size=bs, 
                shuffle=False, 
                num_workers=1, 
                pin_memory=True)
    return dataloader
def lsun(normalizer,bs, is_large_image=False):
    import torchvision
    transformer = transforms.Compose([transforms.Resize(32),
                                      #transforms.CenterCrop(32),
                                      transforms.ToTensor(),
                                      normalizer
                                      ])
    if is_large_image:
        transformer = transforms.Compose([transforms.Resize((254, 254)),
                                          # transforms.CenterCrop(32),
                                          transforms.ToTensor(),
                                          normalizer
                                          ])
    info_lsun_dataset = torchvision.datasets.ImageFolder("../data/LSUN",
                                                         transform=transformer)
    dataloader = torch.utils.data.DataLoader(
                info_lsun_dataset,
                batch_size=bs, 
                shuffle=False, 
                num_workers=1, 
                pin_memory=True)
    return dataloader
def lsunR(normalizer,bs, is_large_image=False):
    import torchvision
    transformer = transforms.Compose([transforms.Resize(32),
                                      #transforms.CenterCrop(32),
                                      transforms.ToTensor(),
                                      normalizer
                                      ])
    if is_large_image:
        transformer = transforms.Compose([transforms.Resize((254, 254)),
                                          # transforms.CenterCrop(32),
                                          transforms.ToTensor(),
                                          normalizer
                                          ])
    info_lsunR_dataset = torchvision.datasets.ImageFolder("../data/LSUN_resize",
                                                          transform=transformer)
    dataloader = torch.utils.data.DataLoader(
                info_lsunR_dataset,
                batch_size=bs, 
                shuffle=False, 
                num_workers=1, 
                pin_memory=True)
    return dataloader
def isun(normalizer,bs, is_large_image=False):
    import torchvision
    transformer = transforms.Compose([transforms.Resize(32),
                                      #transforms.CenterCrop(32),
                                      transforms.ToTensor(),
                                      normalizer
                                      ])
    if is_large_image:
        transformer = transforms.Compose([transforms.Resize((254, 254)),
                                          # transforms.CenterCrop(32),
                                          transforms.ToTensor(),
                                          normalizer
                                          ])
    info_isun_dataset = torchvision.datasets.ImageFolder("../data/iSUN",
                                                         transform=transformer)
    dataloader = torch.utils.data.DataLoader(
                info_isun_dataset,
                batch_size=bs, 
                shuffle=False, 
                num_workers=1, 
                pin_memory=True)
    return dataloader


def tinyC(normalizer, bs, is_large_image=False):
    import torchvision
    transformer = transforms.Compose([transforms.Resize(32),
                                      # transforms.CenterCrop(32),
                                      transforms.ToTensor(),
                                      normalizer
                                      ])
    if is_large_image:
        transformer = transforms.Compose([transforms.Resize((254, 254)),
                                          # transforms.CenterCrop(32),
                                          transforms.ToTensor(),
                                          normalizer
                                          ])
    tinyc_dataset = torchvision.datasets.ImageFolder(
        "../data/Tiny-Imagenet-Crop/",
        transform=transformer)

    dataloader = torch.utils.data.DataLoader(
        tinyc_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=1,
        pin_memory=True)
    return dataloader


def tinyR(normalizer, bs, is_large_image=False):
    import torchvision
    transformer = transforms.Compose([transforms.Resize(32),
                                      # transforms.CenterCrop(32),
                                      transforms.ToTensor(),
                                      normalizer
                                      ])
    if is_large_image:
        transformer = transforms.Compose([transforms.Resize((254, 254)),
                                          # transforms.CenterCrop(32),
                                          transforms.ToTensor(),
                                          normalizer
                                          ])
    tinyr_dataset = torchvision.datasets.ImageFolder(
        "../data/Tiny-Imagenet-Resize/",
        transform=transformer)
    dataloader = torch.utils.data.DataLoader(
        tinyr_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=1,
        pin_memory=True)
    return dataloader

def imagenet30(normalizer, bs, is_large_image=False):
    import torchvision
    transformer = transforms.Compose([transforms.Resize(32),
                                      # transforms.CenterCrop(32),
                                      transforms.ToTensor(),
                                      normalizer
                                      ])
    if is_large_image:
        transformer = transforms.Compose([transforms.Resize((254, 254)),
                                          # transforms.CenterCrop(32),
                                          transforms.ToTensor(),
                                          normalizer
                                          ])
    tinyr_dataset = torchvision.datasets.ImageFolder(
        "../data/ImageNet-30/one_class_test",
        transform=transformer)
    dataloader = torch.utils.data.DataLoader(
        tinyr_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=1,
        pin_memory=True)
    return dataloader
