import random
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.ops import complete_box_iou_loss


alpha = 0.3
p = 1


class TrainModel:
    def __init__(self, model, train_dataset, test_dataset):
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available()else "cpu")
        self.model = model.to(self.device)
        self.count_epoch = 0
        self.train_loss = []
        self.test_loss = []

    def setting(self, batch_size, learning_rate):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=True)
        self.loss_fn = complete_box_iou_loss
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate)

    def train_loop(self):
        loss_sum = 0
        for batch, (X, y) in enumerate(tqdm(self.train_dataloader)):
            X = X.to(self.device).to(torch.float32).requires_grad_(True)
            y = y.to(self.device).to(torch.float32).requires_grad_(True)
            pred = self.model(X)
            loss = self.loss_fn(pred, y, reduction='sum')
            l_norm = torch.tensor(0., requires_grad=True)
            for w in self.model.parameters():
                l_norm = l_norm + torch.linalg.norm(w.flatten(), p)
            loss = loss + alpha * l_norm
            loss_sum += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.train_loss.append(loss_sum)
        print(loss_sum)

    def test_loop(self):
        loss_sum = 0
        with torch.no_grad():
            for X, y in self.test_dataloader:
                X = X.to(self.device).to(torch.float32)
                y = y.to(self.device).to(torch.float32)
                pred = self.model(X)
                loss = self.loss_fn(pred, y, reduction='sum')
                loss_sum += loss.item()
        self.count_epoch += 1
        self.test_loss.append(loss_sum)
        a, b = self.count_epoch, loss_sum
        print(f"epoch : {a} loss : {b}")

    def predict_face(self, show_num):
        self.test_dataloader = DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for X, y in self.test_dataloader:
                X = X.to(self.device).to(torch.float32)
                y = y.to(self.device).to(torch.float32)
                pred = self.model(X)
                break
        pred = pred.to("cpu").detach().numpy().copy()
        print(len(pred))
        show_list = random.sample(range(len(pred)), k=show_num)
        for i in show_list:
            self.test_dataset.image_show(i, pred[i])
