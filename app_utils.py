import torch
from transformers import AutoTokenizer, AutoConfig
import pandas as pd 
import numpy as np
from transformers import RobertaModel
import torch.nn as nn
import json




target2index_file_address = "E:\\Paid projects\\NLP_ICD10\\app\\model_checkpoints\\files\\model_checkpoints\\mimiciv_icd10\\plm-icd\\target2index.json"
model_weights_path = "E:\\Paid projects\\NLP_ICD10\\app\\model_checkpoints\\files\\model_checkpoints\\mimiciv_icd10\\plm-icd\\final_model.pt"
pretrained_model_path = "E:\\Paid projects\\NLP_ICD10\\roberta-pm-m3-voc-da-laat-3072-128-bs8-e20-warmup-50\\" 
procedures_description_directory = "E:\\Paid projects\\NLP_ICD10\\app\\d_icd_procedures.csv\\d_icd_procedures.csv"
diagnoses_description_directory = "E:\\Paid projects\\NLP_ICD10\\app\\d_icd_diagnoses.csv\\d_icd_diagnoses.csv"

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

# Set the device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LabelAttention(nn.Module):
    def __init__(self, input_size: int, projection_size: int, num_classes: int):
        super().__init__()
        self.first_linear = nn.Linear(input_size, projection_size, bias=False)
        self.second_linear = nn.Linear(projection_size, num_classes, bias=False)
        self.third_linear = nn.Linear(input_size, num_classes)
        self._init_weights(mean=0.0, std=0.03)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """LAAT attention mechanism

        Args:
            x (torch.Tensor): [batch_size, seq_len, input_size]

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """
        weights = torch.tanh(self.first_linear(x))
        att_weights = self.second_linear(weights)
        att_weights = torch.nn.functional.softmax(att_weights, dim=1).transpose(1, 2)
        weighted_output = att_weights @ x
        return (
            self.third_linear.weight.mul(weighted_output)
            .sum(dim=2)
            .add(self.third_linear.bias)
        )

    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        """
        Initialise the weights

        Args:
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 0.03.
        """

        torch.nn.init.normal_(self.first_linear.weight, mean, std)
        torch.nn.init.normal_(self.second_linear.weight, mean, std)
        torch.nn.init.normal_(self.third_linear.weight, mean, std)


class PLMICD(nn.Module):
    def __init__(self, num_classes: int, model_path: str, **kwargs):
        super().__init__()
        self.config = AutoConfig.from_pretrained(
            model_path, num_labels=num_classes, finetuning_task=None
        )
        self.roberta = RobertaModel(
            self.config
        ).from_pretrained(model_path, config=self.config)
        self.attention = LabelAttention(
            input_size=self.config.hidden_size,
            projection_size=self.config.hidden_size,
            num_classes=num_classes,
        )
        self.loss = torch.nn.functional.binary_cross_entropy_with_logits

    def get_loss(self, logits, targets):
        return self.loss(logits, targets)

    def training_step(self, batch) -> dict[str, torch.Tensor]:
        data, targets, attention_mask = batch.data, batch.targets, batch.attention_mask
        logits = self(data, attention_mask)
        loss = self.get_loss(logits, targets)
        logits = torch.sigmoid(logits)
        return {"logits": logits, "loss": loss, "targets": targets}

    def validation_step(self, batch) -> dict[str, torch.Tensor]:
        data, targets, attention_mask = batch.data, batch.targets, batch.attention_mask
        logits = self(data, attention_mask)
        loss = self.get_loss(logits, targets)
        logits = torch.sigmoid(logits)
        return {"logits": logits, "loss": loss, "targets": targets}

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
    ):
        r"""
        input_ids (torch.LongTensor of shape (batch_size, num_chunks, chunk_size))
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, num_labels)`, `optional`):
        """

        batch_size, num_chunks, chunk_size = input_ids.size()
        outputs = self.roberta(
            input_ids.view(-1, chunk_size),
            attention_mask=attention_mask.view(-1, chunk_size)
            if attention_mask is not None
            else None,
            return_dict=False,
        )

        hidden_output = outputs[0].view(batch_size, num_chunks * chunk_size, -1)
        logits = self.attention(hidden_output)
        return logits


model = PLMICD(num_classes=7942, model_path=pretrained_model_path)

checkpoint = torch.load(model_weights_path, map_location=torch.device(device))
model.load_state_dict(checkpoint["model"], strict=False)

def get_codes(above_threshold_indices):
    with open(target2index_file_address, "r") as file:
        data = json.load(file)
    df = pd.DataFrame(list(data.items()), columns=['Code', 'Index'])
    codes = df.loc[np.array(above_threshold_indices).flatten(), 'Code']
    codes = codes.astype(str)
    codes = codes.apply(lambda x: x.replace('.', '') if '.' in x else x)
    return codes

def get_descriptions(codes):
    matched_titles = []
    code_types = []
    procedures = pd.read_csv(procedures_description_directory)
    diagnoses = pd.read_csv(diagnoses_description_directory)
    descriptions = pd.concat([procedures, diagnoses])
    descriptions_10 = descriptions[descriptions['icd_version'] == 10]
    

    for code in codes:
        if code in procedures['icd_code'].values:
            code_types.append(0)
        elif code in diagnoses['icd_code'].values:
            code_types.append(1)

        matched_title = descriptions_10.loc[descriptions_10['icd_code'] == code, 'long_title'].values
        matched_titles.append(matched_title[0])
    return matched_titles, code_types

# Perform inference
def inference_icd(inputs, threshold):
    columns = ['Codes', 'Confidence', 'Descriptions', 'Code Types']
    results = pd.DataFrame(columns = columns)
    with torch.no_grad():
        # Get the input_ids tensor
        input_ids = inputs["input_ids"]
        attention_mask = inputs['attention_mask']
        
        # # Reshape input_ids to match the expected shape (assuming batch_size=1)
        batch_size, chunk_size = input_ids.size()
        # print(chunk_size)
        num_chunks = 1
        input_ids = input_ids.view(batch_size, num_chunks, chunk_size)
        
        # Move inputs to the device
        input_ids = input_ids.to(device)
        
        # Call the model's forward method
        logits = model(input_ids, attention_mask)
        logits = torch.sigmoid(logits[0])
        above_threshold_indices = (logits > threshold).nonzero()
        confidence_scores = logits[above_threshold_indices]
        codes = get_codes(above_threshold_indices)
        descriptions, code_types = get_descriptions(codes)
        results['Codes'] = codes.values
        results['Confidence'] = confidence_scores.numpy()
        results['Descriptions'] = descriptions
        results['Code Types'] = code_types

        result_pcs = results[results['Code Types'] == 0].drop(columns=['Code Types'])
        result_cm = results[results['Code Types'] == 1].drop(columns=['Code Types'])
        result_pcs = result_pcs.reset_index(drop=True)
        result_cm = result_cm.reset_index(drop=True)

    return result_pcs, result_cm
