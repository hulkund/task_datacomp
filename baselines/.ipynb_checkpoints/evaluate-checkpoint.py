from all_datasets.COOS_dataset import COOSDataset
from all_datasets.FMoW_dataset import FMoWDataset
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPProcessor
from sklearn.metrics import precision_score, recall_score, accuracy_score

def load_model_processor(outputs_path):
    model = CLIPModel.from_pretrained(outputs_path+"model.pt")
    processor = CLIPProcessor.from_pretrained(outputs+path+"processor.pt")
    return model, processor

def get_test_dataset(dataset_name, split):
    if dataset_name == "COOS":
        dataset = COOSDataset(split)
    if dataset_name == "FMoW":
        dataset = FMoWDataset(split)
    if dataset_name == "iWildCam":
        continue
    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    return dataset, train_dataloader

def evaluate(model, processor, test_dataloader):
    model.to(device)
    model.eval()  
    predictions = []
    ground_truth = []
    uids_all=[]
    with torch.no_grad():
         for data, texts, labels, uids in test_dataloader:
            data, labels, texts = data.to(device), labels.to(device), texts.to(device)
            with torch.no_grad():
                inputs = processor(text=texts, images=images, return_tensors="pt", padding=True)
            logits = model(**inputs).logits_per_image
            pred_label = torch.argmax(logits, dim=1)
            predictions.extend(pred_label.cpu().numpy())
            ground_truth.extend(labels.cpu().numpy())
            uids_all.extend(uids)
    return predictions, ground_truth, logits

def get_metrics(predictions, ground_truth, logits):
    acc = accuracy_score(ground_truth, predictions)
    top_5_acc = top_k_accuracy(logits=logits, labels=ground_truth)
    precision = precision_score(ground_truth, predictions, average='macro')
    recall = recall_score(ground_truth, predictions, average='macro')
    metrics = {"acc":acc, "top_5_acc":top_5_acc, "precision": precision, "recall":recall}
    with open(args.outputs_path+"metrics.json", "w") as json_file:
        json_dump(metrics, json_file, indent=4)
    return metrics

def top_k_accuracy(logits, labels, k=5):
    top_k_preds = np.argsort(logits, axis=1)[:, -k:]
    top_k_correct = np.any(top_k_preds == labels[:, None], axis=1)
    return np.mean(top_k_correct)

def main():
    parser.add_argument('--config', help='Path to config file')
    parser.add_argument('--dataset', type=str, help='Name of dataset')
    parser.add_argument('--task_num', type=int)
    parser.add_argument('--outputs_path', type=str)

    test_dataset, test_dataloader = get_test_dataset(args.dataset, split=f"test{args.task_num}")
    model, processor = get_model_processor(test_dataset.num_classes)
    predictions, ground_truth, logits = evaluate(model=model, processor=processor, test_dataloader=test_dataloader)
    save_metrics(predictions=predictions, ground_trith=ground_truth, logits=logits)

    

