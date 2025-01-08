## Serving a model


1. Initializing the project
 ```Python
import digitalhub as dh

project = dh.get_or_create_project("demo-early-exit-eng")
```

2. Define a funtion that handle the serving process
```Python
function = project.new_function(
    kind='container',
    name='serve_eng_function',
    image="ghcr.io/tn-aixpa/audio-early-exit-eng:0.1",
    args=["early-exit-eng-model"]
)
```
As args the name of the model logged on the project in the training phase must be provided

3. Run the serving function
```Python
run = function.run(
    action="serve",
    profile="1xa100",
    fs_group=100,
    volumes=[
        {
            "volume_type": "persistent_volume_claim",
            "name": "early-exit-demo-serve",
            "mount_path": "/data",
            "spec": {
                "claim_name": "early-exit-demo-serve"
            }
        }
    ],
    service_ports = [
        {
            "port": 8051,
            "target_port": 8051
        }
    ]    
)
```
With KRM tool a volume must be defined in order to store the model file used in the prediction service. The volume is also used to store the uploaded audio file.

4. In order to invoke the service, an HTTP form-multipart post request must be invoked
```Python
import requests

url = "http://" + run.status.service.name
file_path = "test-audio.wav"

with open(file_path, "rb") as file:
    files = {'file': file}
    response = requests.post(url, files=files)

print(f"response code:{response.status_code}")
print(f"response body:{response.text}")
```

