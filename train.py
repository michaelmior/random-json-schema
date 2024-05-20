import bert_score
import datasets
from transformers import (
    AutoTokenizer,
    EncoderDecoderModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)


MAX_LEN = 512
TRAIN_BATCH_SIZE = 16
TRAIN_EPOCHS = 1


tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")

dataset = datasets.load_dataset(
    "dataunitylab/json-schema-store", revision="definition-examples-short"
)


def process_data_to_model_inputs(batch):
    # Tokenize the input and target data
    inputs = tokenizer(
        batch["defn_schema"], padding="max_length", truncation=True, max_length=MAX_LEN
    )
    outputs = tokenizer(
        batch["value"], padding="max_length", truncation=True, max_length=MAX_LEN
    )

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = outputs.input_ids
    batch["decoder_attention_mask"] = outputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()

    batch["labels"] = [
        [-100 if token == tokenizer.pad_token_id else token for token in labels]
        for labels in batch["labels"]
    ]

    return batch


def get_data(dataset, split):
    data = dataset[split].map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=TRAIN_BATCH_SIZE,
        remove_columns=["name", "defn_name", "defn_schema", "value"],
    )
    data.set_format(
        type="torch",
        columns=[
            "input_ids",
            "attention_mask",
            "decoder_input_ids",
            "decoder_attention_mask",
            "labels",
        ],
    )
    return data


val_data = get_data(dataset, "validation")
train_data = get_data(dataset, "train")

bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained(
    "microsoft/codebert-base", "microsoft/codebert-base", tie_encoder_decoder=True
)
# set special tokens
bert2bert.config.decoder_start_token_id = tokenizer.bos_token_id
bert2bert.config.eos_token_id = tokenizer.eos_token_id
bert2bert.config.pad_token_id = tokenizer.pad_token_id

# load rouge for validation
bert_score.utils.model2layers["microsoft/codebert-base"] = 6
bertscore = datasets.load_metric("bertscore", trust_remote_code=True)


def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    # all unnecessary tokens are removed
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    bertscore_output = bertscore.compute(
        predictions=pred_str, references=label_str, model_type="microsoft/codebert-base"
    )

    return {
        "bertscore_precision": round(bertscore_output.precision, 4),
        "bertscore_recall": round(bertscore_output.recall, 4),
        "bertscore_fmeasure": round(bertscore_output.fmeasure, 4),
    }


training_args = Seq2SeqTrainingArguments(
    output_dir="model",
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=TRAIN_BATCH_SIZE,
    predict_with_generate=True,
    # evaluate_during_training=True,
    eval_strategy="epoch",
    do_train=True,
    do_eval=True,
    logging_steps=1024,
    save_steps=2048,
    warmup_steps=1024,
    # max_steps=1500, # delete for full training
    num_train_epochs=TRAIN_EPOCHS,
    overwrite_output_dir=True,
    save_total_limit=1,
    fp16=True,
)

trainer = Seq2SeqTrainer(
    model=bert2bert,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,
)
trainer.train()
trainer.save_model("model")
