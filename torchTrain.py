from torchvision import datasets
import torch.nn as nn
import torch
import torch.optim as optim

from train_utils import get_transformers, get_model, TrainDto, perform_training


transformers = get_transformers()

data_set = {
  'train': datasets.ImageFolder('data/frutas', transformers['train']),
  'test': datasets.ImageFolder('data/validacion/', transformers['test'])
}

data_loaders = {
  'train': torch.utils.data.DataLoader(
    data_set['train'],
    batch_size=32,
    shuffle=True,
    num_workers=4
  ),
  'test': torch.utils.data.DataLoader(
    data_set['test'],
    batch_size=32,
    shuffle=False,
    num_workers=4
  )
}

device = "cuda"

model = get_model(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters())

dto = TrainDto(
  device=device,
  criterion=criterion,
  loaders=data_loaders,
  optimizer=optimizer,
  dataset=data_set,
  epochs=20
)

trained_model = perform_training(model, dto)

torch.save(trained_model.state_dict(), 'placa.h5')