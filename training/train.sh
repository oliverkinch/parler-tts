# Hyper parameters
export WARMUP_STEPS=200
export SAVE_STEPS=1000 # 5000
export MAX_STEPS=1000 # 50000
export EVAL_STEPS=10000
export PER_DEVICE_EVAL_BATCH_SIZE=4
export MAX_EVAL_SAMPLES=20 # 20
export GRADIENT_ACCUMULATION_STEPS=24
export LEARNING_RATE=0.000005
export LR_SCHEDULER_TYPE="constant_with_warmup" # cosine, constant_with_warmup

# Train
export TRAIN_DATASET_NAME="oliverkinch/coral-tts-filtered-mic"
export TRAIN_METADATA_DATASET_NAME="oliverkinch/coral-tts-filtered-mic-tagged"
export OVERWRITE_OUTPUT_DIR=true
# export RESUME_FROM_CHECKPOINT=""
export MAX_TRAIN_SAMPLES=10
export NO_DESCRIPTIONS=true

# Eval
export EVAL_SPLIT_NAME="eval"
export ASR_MODEL_NAME_OR_PATH="jstoone/whisper-medium-da" # alexandrainst/roest-315m jstoone/whisper-medium-da
export SAVE_TOTAL_LIMIT=1
export DO_EVAL=false

# Output paths
export OUTPUT_DIR="./output/output_dir_training/"
export TEMPORARY_SAVE_TO_DISK="./output/audio_code_tmp/"
export SAVE_TO_DISK="./output/tmp_dataset_audio/"

accelerate launch ./training/run_parler_tts_training.py \
    --model_name_or_path "./parler-tts-untrained-600M/parler-tts-untrained-600M/" \
    --feature_extractor_name "parler-tts/dac_44khZ_8kbps" \
    --description_tokenizer_name "google/mt5-large" \
    --prompt_tokenizer_name "google/mt5-large" \
    --report_to "wandb" \
    --overwrite_output_dir $OVERWRITE_OUTPUT_DIR \
    --train_dataset_name $TRAIN_DATASET_NAME \
    --train_metadata_dataset_name $TRAIN_METADATA_DATASET_NAME \
    --train_dataset_config_name "default" \
    --train_split_name "train" \
    --target_audio_column_name "audio" \
    --description_column_name "text_description" \
    --prompt_column_name "text" \
    --max_duration_in_seconds 30 \
    --min_duration_in_seconds 2.0 \
    --max_text_length 600 \
    --add_audio_samples_to_wandb true \
    --preprocessing_num_workers 1 \
    --do_train true \
    --num_train_epochs 1000 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --gradient_checkpointing false \
    --per_device_train_batch_size 1 \
    --learning_rate $LEARNING_RATE \
    --adam_beta1 0.9 \
    --adam_beta2 0.99 \
    --weight_decay 0.01 \
    --lr_scheduler_type $LR_SCHEDULER_TYPE \
    --warmup_steps $WARMUP_STEPS \
    --logging_steps 500 \
    --freeze_text_encoder true \
    --include_inputs_for_metrics true \
    --save_steps $SAVE_STEPS \
    --audio_encoder_per_device_batch_size 5 \
    --dtype "fp16" \
    --seed 123 \
    --output_dir $OUTPUT_DIR \
    --temporary_save_to_disk $TEMPORARY_SAVE_TO_DISK \
    --save_to_disk $SAVE_TO_DISK \
    --dataloader_num_workers 4 \
    --group_by_length true \
    --attn_implementation "sdpa" \
    --max_steps $MAX_STEPS \
    --hub_model_id "oliverkinch/coral-parler-tts-800M" \
    --save_total_limit 1 \
    --do_eval $DO_EVAL \
    --eval_split_name $EVAL_SPLIT_NAME \
    --eval_steps $EVAL_STEPS \
    --asr_model_name_or_path $ASR_MODEL_NAME_OR_PATH \
    --predict_with_generate true \
    --report_to "wandb" \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --max_eval_samples $MAX_EVAL_SAMPLES \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --max_train_samples $MAX_TRAIN_SAMPLES \
    --no_descriptions $NO_DESCRIPTIONS
    # --resume_from_checkpoint $RESUME_FROM_CHECKPOINT
