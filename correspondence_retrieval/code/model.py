from pathlib import Path

from tqdm import tqdm

import torch
import torchvision
import torchvision.transforms as transforms


def _get_finetuned_backbone(net, device, num_workers, sample=False, epochs=350):
    mean, std = [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]  # color normalization

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    '''
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    '''
    if sample:
        trainset.data = trainset.data[:100]

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=num_workers)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        net.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
    )

    def set_lr(optimizer, new_lr):
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr

    for epoch in tqdm(range(epochs), ncols=80):
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        if epoch < 150:
            lr = 0.1
        elif epoch < 250:
            lr = 0.01
        else:
            lr = 0.001
        set_lr(optimizer, lr)
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        print_str = "epoch: {}, Loss: {:.3f} | Acc: {:.3f}".format(
            epoch + 1,
            train_loss / len(trainloader),
            100. * correct / total
        )
        tqdm.write(print_str)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    if sample:
        testset.data = testset.data[:100]

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=num_workers)

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print("Acc: {:.3f}".format(100. * correct / total))
    return net, correct / total


def save_model(net, acc, path, sample=False):
    path.mkdir(exist_ok=True)
    if sample:
        name = "sample_acc_{}.torch".format(acc)
    else:
        name = "acc_{}.torch".format(acc)
    path = path / name
    torch.save(net.state_dict(), str(path))


def get_finetuned_backbone(net, device, num_workers, sample=False):
    path = './data/ckpt'
    path = Path(path).resolve()
    paths = list(path.glob('*.torch'))
    if len(paths) > 0:
        # load
        path = sorted(paths, reverse=True)[0]  # get highest acc
        weights = torch.load(str(path))
        print("loading backbone from {}".format(path))
        net.load_state_dict(weights)
    else:
        print("finetuning backbone")
        net, acc = _get_finetuned_backbone(net, device, num_workers, sample=sample)
        print("saving backbone to {}".format(path))
        save_model(net, acc, path, sample=sample)
    return net


def get_backbone(device, num_workers, finetune=False, sample=False,
                 model_name='ResNet50'):
    # TODO: support multiple types of backbones
    net = torchvision.models.resnet50(pretrained=True)
    net = net.to(device=device)

    if finetune:
        net = get_finetuned_backbone(net, device, num_workers, sample=sample)
    '''
    else:
        print("using model without finetuning")
    '''
    return net


class Model(torch.nn.Module):
    def __init__(self, backbone):
        super(Model, self).__init__()
        module_list = list(backbone.children())
        self.conv_net = torch.nn.Sequential(*module_list[:-1])

    def forward(self, x):
        x = self.conv_net(x)
        x = torch.flatten(x, 1)
        return x


class LayerModel(torch.nn.Module):
    def __init__(self, backbone, layer=0):
        super(LayerModel, self).__init__()

        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool

        self.layer = layer

        self.layers = []
        for i in range(self.layer):
            self.layers.append(getattr(backbone, f'layer{i + 1}'))
        self.layers = torch.nn.ModuleList(self.layers)
        if len(self.layers) == 4:
            self.fc = backbone.fc
        self.avgpool = backbone.avgpool

    def prepare(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

    def forward(self, x):
        x = self.prepare(x)

        if len(self.layers) > 0:
            for layer in self.layers:
                x = layer(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        if hasattr(self, 'fc'):
            x = self.fc(x)
        return x


def get_layer_extractors(backbone):
    assert isinstance(backbone, torchvision.models.ResNet), "layer extraction is only supported for resnet models for now"

    models = {}
    for i in range(5):
        models[f'layer_{i}'] = LayerModel(backbone, i)
    return models


def get_model(device, num_workers, finetune=False, sample=False,
              model_name='ResNet50', extract_each_layer=False):
    backbone = get_backbone(device, num_workers, finetune=finetune, sample=sample,
                            model_name=model_name)
    backbone.eval()
    backbone = backbone.cpu()
    if extract_each_layer:
        models = get_layer_extractors(backbone)
    else:
        models = {'model': Model(backbone)}
    models = {k: model.to(device=device).eval() for k, model in models.items()}

    return models
