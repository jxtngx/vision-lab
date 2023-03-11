### PodTrainer


```python
visionpod.core.trainer.PodTrainer(
    logger=None, profiler=None, callbacks=[], plugins=[], set_seed=True, **trainer_init_kwargs
)
```


A custom Lightning.LightningTrainer

__Arguments__

- __logger__ `lightning.pytorch.loggers.logger.Logger | None`: None
- __profiler__ `lightning.pytorch.profilers.profiler.Profiler | None`: None
- __callbacks__ `List | None`: []
- __plugins__ `List | None`: []
- __set_seed__ `bool`: True
- __trainer_init_kwargs__ `Dict[str, Any]`:


----

### persist_predictions


```python
PodTrainer.persist_predictions(
    predictions_dir="/Users/justin/Developer/lightning-pod-projects/lightning-pod-vision/data/predictions/predictions.pt",
)
```


helper method to persist predictions on completion of a training run

__Arguments__

- __predictions_dir__ `str | pathlib.Path | None`: the directory path where predictions should be saved to


----
