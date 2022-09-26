exp_name = 'bert_exp'

row_data_path = 'data/row_data.csv'
data_path = 'data/train_data.csv'

model_name = 'bert-large-uncased'
model_type = 'BertForSequenceClassification'
tokenizer_type = 'BertTokenizer'

tokenizer = dict(
    type=tokenizer_type,
    pretrained_model_name_or_path=model_name,
    do_lower_case=True,
    infer_params=dict(
        max_length=185,
        add_special_tokens=True,
        padding='max_length',
        truncation='longest_first',
        return_attention_mask = True,
        return_tensors='pt'
    )
)

train_dataset = dict(
    type='BotDetectionDataset',
    mode='train'
)

val_dataset = dict(
    type='BotDetectionDataset',
    mode='val'
)

train_dataloader = dict(
    batch_size=8,
    num_workers=6,
)

val_dataloader = dict(
    batch_size=8,
    num_workers=6
)

train_procedure = dict(
    folds=5,
    epochs=60,
    val_step=1,
    accum_grad_steps=1,

    device_name='cuda',
    use_wandb=True
)

losses = [
    dict(
        type='BinaryCrossEntropy',
        display_name='BCE',
        pred_name='model_output',
        target_name='target',
        loss_weight=1,
        neg_samples_weight=1,
        pos_samples_weight=1
    )
]

metrics = [
    dict(
        type='F1',
        display_name='F1',
        pred_name='model_output',
        target_name='target',
        threshold=0.5
    )
]

model = dict(
    type='HuggingFaceModel',
    model_type=model_type,
    model_name=model_name,
)

optimizer = dict(
    type='AdamW',
    betas=(0.9, 0.999),
    weight_decay=1e-2
)

scheduler = dict(
    start_lr=1e-8,
    max_lr=5e-6,
    min_lr=1e-8,
    warmup_epochs=2
)