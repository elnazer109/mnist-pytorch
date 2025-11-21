import torch
from torch import nn
from model import model
from dataset import train_dataloader , test_dataloader
from sklearn.metrics import accuracy_score

device  = "cuda" if torch.cuda.is_available() else "cpu"



loss = nn.CrossEntropyLoss()

model = model().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

def training(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  model.train() # start training mode

  for batch , (X,y) in enumerate(dataloader):
    X , y = X.to(device) , y.to(device)
    pred = model(X)
    loss = loss_fn(pred , y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step() # update w and b


def test(dataloader , model , loss_fn):
  model.eval() # start evaluation
  model.to(device)

  all_labels =[]
  all_preds=[]
  total_loss = 0
  with torch.no_grad():
    for image , label in dataloader:
      image , label = image.to(device) , label.to(device)
      pred = model(image)

      total_loss += loss_fn(pred , label).item()

      _ , pred2 = torch.max(pred , axis=1) # get highest prob
      all_labels.extend(label.cpu().numpy())
      all_preds.extend(pred2.cpu().numpy())
      avg_loss = total_loss / len(dataloader)
      accuracy = accuracy_score(all_labels, all_preds)

  return avg_loss, accuracy, all_preds, all_labels


if __name__ == "__main__":
    training(train_dataloader, model, loss, optimizer)
    test(test_dataloader, model, loss)

  

