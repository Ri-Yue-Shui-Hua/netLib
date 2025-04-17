# SCN (Spacial Confguration Net)

## Integrating Spatial Configuration into Heatmap Regression Based CNNs for Landmark Localization
### Usage
This example implements the networks of the papers [Regressing Heatmaps for Multiple Landmark Localization Using CNNs](https://doi.org/10.1007/978-3-319-46723-8_27) and [Integrating Spatial Configuration into Heatmap Regression Based CNNs for Landmark Localization](https://doi.org/10.1016/j.media.2019.03.007). Look into the subfolders of this repository for individual examples and more details.

You need to have the [MedicalDataAugmentationTool](https://github.com/christianpayer/MedicalDataAugmentationTool) framework downloaded and in you PYTHONPATH for the scripts to work. If you have problems/questions/suggestions about the code, write me a [mail](christian.payer@gmx.net)!

### Train models
Run the main.py files inside the example folders to train the network.

Train and test other datasets
In order to train and test on other datasets, modify the dataset.py files. See the example files and documentation for the specific file formats. Set the parameter save_debug_images = True in order to see, if the network input images are reasonable.

### Citation
If you use this code for your research, please cite our [MIA paper](https://doi.org/10.1016/j.media.2019.03.007) or [MICCAI paper](https://doi.org/10.1007/978-3-319-75541-0_20):
```
@article{Payer2019a,
  title   = {Integrating Spatial Configuration into Heatmap Regression Based {CNNs} for Landmark Localization},
  author  = {Payer, Christian and {\v{S}}tern, Darko and Bischof, Horst and Urschler, Martin},
  journal = {Medical Image Analysis},
  volume  = {54},
  year    = {2019},
  month   = {may},
  pages   = {207--219},
  doi     = {10.1016/j.media.2019.03.007},
}
```
```
@inproceedings{Payer2016,
  title     = {Regressing Heatmaps for Multiple Landmark Localization Using {CNNs}},
  author    = {Payer, Christian and {\v{S}}tern, Darko and Bischof, Horst and Urschler, Martin},
  booktitle = {Medical Image Computing and Computer-Assisted Intervention - {MICCAI} 2016},
  doi       = {10.1007/978-3-319-46723-8_27},
  pages     = {230--238},
  year      = {2016},
}
```


### DataSet

[Cephalometric landmarks](https://www.kaggle.com/datasets/c34a0ef0cd3cfd5c5afbdb30f8541e887171f19f196b1ad63790ca5b28c0ec93)
[Digital Hand Atlas](https://ipilab.usc.edu/research/baaweb/)
[Chest Xray Masks and Labels](https://storage.googleapis.com/kaggle-data-sets/108201/258315/bundle/archive.zip?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=gcp-kaggle-com%40kaggle-161607.iam.gserviceaccount.com%2F20250417%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20250417T080104Z&X-Goog-Expires=259200&X-Goog-SignedHeaders=host&X-Goog-Signature=485f3675b0986dc38914eaa980588e9402f5ae3d754a7cc5ca80d5f3b7d572ec1f0efc6705e6e902d2e64f50b10e0d504da6831f3657f8e97bd253e88a6d02859b06451aab88c0b70f9d0cf0b823465ff5cd4c84fe1a33aad08ce6e6b8d5d9c77e7d4a6924a929ac8c09d160954871967250a067017171ff4bb748d08948867e37497e0b5f1cf7d512cb669a245348fee63b7d87b8c9b133ad2aed7b8e98bf1ed7e371ea069c9b875a8ce64d65f8b3bc7f303162b518df419a245ea489c085af0e559c7256b0717cd7db80c9bd5246291309e338beab38ea66507b42facd5b9cfe9b9e106803cf113e482a93a0d0567faf3486fcc05fae4a9b10c9dc55f36e94)
https://www.kaggle.com/datasets/nikhilpandey360/chest-xray-masks-and-labels?resource=download






