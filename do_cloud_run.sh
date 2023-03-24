# wandb environment variables
# https://docs.wandb.ai/guides/track/environment-variables
lightning run app app.py --cloud \
--env WANDB_CONFIG_DIR=logs/wandb \
--env WANBD_DIR=logs/wandb \
--env WANDB_DISABLE_GIT=True \
--env WANDB_DISABLE_CODE=True \
--secret WANDB_API_KEY=WANDB-API-KEY \
--secret WANDB_ENTITY=WANDB-ENTITY
