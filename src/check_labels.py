from transformers import AutoConfig

config = AutoConfig.from_pretrained("tabularisai/multilingual-emotion-classification")
print(config.id2label)
print(config.num_labels)