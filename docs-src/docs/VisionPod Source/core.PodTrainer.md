### LabTrainer

```python
visionlab.core.trainer.LabTrainer(
    logger=None, profiler=None, callbacks=[], plugins=[], set_seed=True, **trainer_init_kwargs
)
```

A custom Lightning.LightningTrainer

**Arguments**

- **logger** `lightning.pytorch.loggers.logger.Logger | None`: None
- **profiler** `lightning.pytorch.profilers.profiler.Profiler | None`: None
- **callbacks** `List | None`: []
- **plugins** `List | None`: []
- **set_seed** `bool`: True
- **trainer_init_kwargs** `Dict[str, Any]`:

---

### persist_predictions

```python
LabTrainer.persist_predictions(
    predictions_dir="/Users/justin/Developer/lightning-vision-projects/lightning-vision-vision/data/predictions/predictions.pt",
)
```

helper method to persist predictions on completion of a training run

**Arguments**

- **predictions_dir** `str | pathlib.Path | None`: the directory path where predictions should be saved to

---
