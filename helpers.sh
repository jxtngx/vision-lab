# setting mps fallback
export PYTORCH_ENABLE_MPS_FALLBACK=1
# running a nested app
lightning run app visionpod/components/train/work.py --open-ui=False --without-server
