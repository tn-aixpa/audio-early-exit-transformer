## Traning a model

The training phase is based on the LibriSpeech ASR corpus (https://www.openslr.org/12).

1. Initializing the project
 ```Python
import digitalhub as dh

project = dh.get_or_create_project("demo-early-exit-eng")
```

2. Define a funtion that handle the training process
```Python
train_fn = project.new_function(name="train-eng-function",
                                kind="python",
                                python_version="PYTHON3_10",
                                code_src="git+https://github.com/tn-aixpa/audio-early-exit-transformer",
                                handler="train_eng_model:dh_train",
                                requirements=["torch==2.5.0", "torchaudio==2.5.0", "tensorboard==2.18.0", "flashlight==0.1.1", "flashlight-text==0.0.7", "sentencepiece==0.2.0",
                                "soundfile==0.12.1", "editdistance==0.8.1"])
```                                              

3. Run the training function
```Python
train_fn.run(action="job", 
             profile="1xa100",
             parameters={"librispeech_train_dataset": "train-clean-100", "num_epochs": 100, "model_name": "early-exit-eng-model", "base_dir": "/shared/"},
             volumes=[
                {
                    "volume_type": "persistent_volume_claim", 
                    "name": "early-exit-demo-shared", 
                    "mount_path": "/shared", 
                    "spec": { "claim_name": "early-exit-demo-shared" }
                },
                {
                    "volume_type": "persistent_volume_claim", 
                    "name": "early-exit-demo-data", 
                    "mount_path": "/data", 
                    "spec": { "claim_name": "early-exit-demo-data" }
                },
            ])
``` 

The following parameters are used:
- librispeech_train_dataset: the training dataset, could be one of train-clean-100, train-clean-360, train-other-500
- num_epochs: number of epochs 
- model_name: name of the model log on the platform
- base_dir: path where the git project is cloned, default is "/shared/"

With KRM tool two volumes must be defined:
- early-exit-demo-shared: where the git project is cloned
- early-exit-demo-data: where the training dataset id downloaded and extracted. In this folder also for each epoch a new file with the generated model is stored.

Once complete, the function log a new model for the project.