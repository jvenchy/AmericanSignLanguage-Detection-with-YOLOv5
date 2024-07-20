For this project I used YOLOv5 and PyTorch to train a custom model on a dataset of 120 images, each labeled with one of six common signs. 

The actual training data is not provided in this github repository, but the metrics (and some images) from the most recent training run of 500 iterations is available to see in the env7 folder.

Currently, the model struggles with differentiating 'Please' and 'Sorry', as well as 'Hello' and 'I Love You' to a lesser extent.

Here's a breakdown of all possible signs the model was capable of:

<img width="579" alt="Hello!" src="https://github.com/user-attachments/assets/102fdbc0-09f8-4097-a077-8002c699b425">
<img width="591" alt="I Love You" src="https://github.com/user-attachments/assets/cb3bdd48-0bf5-4c33-b879-2e2b9b4b717e">
<img width="575" alt="Yes, True" src="https://github.com/user-attachments/assets/48a7f1c1-b8c8-43be-9e8a-b70ccd7fa618">
<img width="575" alt="No, False" src="https://github.com/user-attachments/assets/a18543c1-4403-403b-8ec1-18fddc6a421a">
<img width="539" alt="Please" src="https://github.com/user-attachments/assets/ac2e8d9e-9c33-4173-9a0e-abc1c9b93056">
<img width="537" alt="Sorry" src="https://github.com/user-attachments/assets/6f783183-d9ad-4b02-9435-638fa605611b">
