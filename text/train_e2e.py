import torch as ch
from tqdm import tqdm
import numpy as np
import os
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from utils import CustomBertModel, log
from transformers import AlbertTokenizer, AlbertModel
from transformers import AdamW


def train_model(model, t_loader, v_loader, epochs, save_dir):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    data_fields = ['input_ids', 'token_type_ids', 'attention_mask']

    def acc_fn(x, y):
        return ch.sum((y == (x >= 0)))

    for e in range(epochs):
        # Train
        running_loss, running_acc = 0.0, 0.0
        num_samples = 0
        model.train()
        iterator = tqdm(t_loader)
        for datum in iterator:
            x = {k: datum[k].cuda() for k in data_fields}
            y = datum['stars'].cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(x)[:, 0]
            loss = loss_fn(outputs, y.float())
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.shape[0]
            running_acc += acc_fn(outputs, y)
            num_samples += y.shape[0]

            iterator.set_description("Epoch %d : [Train] "
                                     "Loss: %.5f Accuacy: %.2f" % (
                                      e, running_loss / num_samples,
                                      100 * running_acc / num_samples))

        # Validation
        model.eval()
        running_loss, running_acc = 0.0, 0.0
        num_samples = 0
        for datum in v_loader:
            x = {k: datum[k].cuda() for k in data_fields}
            y = datum['stars'].cuda()

            with ch.no_grad():
                outputs = model(x)[:, 0]
                loss = loss_fn(outputs, y.float())
                running_loss += loss.item() * y.shape[0]
                running_acc += acc_fn(outputs, y)
                num_samples += y.shape[0]

        print("[Val] Loss: %.5f Accuacy: %.2f\n" %
              (running_loss / num_samples, 100 * running_acc / num_samples))

        # Save model at end of epoch
        ch.save(model.state_dict(), os.path.join(
            save_dir, "%d_%.3f.pth" % (e, running_acc / num_samples)))


if __name__ == "__main__":
    import sys
    model_dir = sys.argv[1]
    subpick = sys.argv[2]
    # data_path = sys.argv[2]
    # model_num = int(sys.argv[3])
    # not_want_prop = int(sys.argv[4]) == 0

    batch_size = 32
    train_indices = np.load("./data/splits/%s/train.npy" % subpick)
    val_indices = np.load("./data/splits/%s/val.npy" % subpick)
    test_indices = np.load("./data/splits/%s/test.npy" % subpick)

    # Property filter
    def dfilter(x):
        return np.logical_and(x['product_category'] != 'home',
                              x['product_category'] != 'home_improvement')

    # Rating binarizer
    def rating_binarizer(x):
        x['stars'] = 1 * (np.array(x['stars']) > 3)
        return x

    # Load dataset
    dataset = load_dataset("amazon_reviews_multi", 'en')
    # Use specified indices
    dataset['train'] = dataset['train'].select(train_indices)
    dataset['validation'] = dataset['validation'].select(val_indices)
    dataset['test'] = dataset['test'].select(test_indices)
    log("[Loaded dataset]")

    # Load tokenizer
    tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')

    # Get rid of neutral ratings and
    # Map ratings to binary values
    dataset = dataset.filter(lambda e: e['stars'] != 3)
    dataset = dataset.map(rating_binarizer, batched=True)
    log("[Binarized ratings]")

    # Tokenize sentences
    dataset = dataset.map(lambda e: tokenizer(
        e['review_body'], truncation=True, padding='max_length',
        max_length=512), batched=True)
    log("[Tokenized sentences]")

    # Apply property on dataset if requested
    # if not_want_prop:
        # dataset = dataset.filter(dfilter)

    dataset.set_format(type='torch', columns=[
        'input_ids', 'token_type_ids', 'attention_mask', 'stars'])

    train_loader = DataLoader(dataset['train'], batch_size=batch_size)
    val_loader = DataLoader(dataset['validation'], batch_size=512)

    # Create model
    bert_base = AlbertModel.from_pretrained('albert-base-v2')    
    model = CustomBertModel(bert_base).cuda()
    model = nn.DataParallel(model)
    model.train()

    # Train model
    train_model(model, train_loader, val_loader, epochs=10, save_dir=model_dir)
