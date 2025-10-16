# Early Exit ASR

The component allows for creating a service from a predefined audio transcription model using the early exit transformer architectire. 

Specifically, a service that exposes ASR funcitonality providing an input audio file, may be created. The service
is exposed as a Serverless function on the platform. The details on how the operation may be accomplished is documented
[here](./howto/serving.md).

For more details regarding the implementation of this functionality refer to [this repo](https://github.com/SpeechTechLab/early-exit-transformer.git).
