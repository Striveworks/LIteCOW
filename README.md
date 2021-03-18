<br />
<p align="center">

  <img src="docs/source/_static/icow_final.svg" alt="" draggable="false" width="300" height="300">


  <h3 align="center">Inference with Collected ONNX Weights</h3>

  <p align="center">
    Easily deploy inference models to dev, test, and production at scale
    <br />
    <a href="https://striveworks.github.io/LIteCOW/index.html"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href=https://github.com/Striveworks/LIteCOW/issues">Report Bug</a>
    ·
    <a href="https://github.com/Striveworks/LIteCOW/issues">Request Feature</a>
  </p>
</p>
<br>




<!-- GETTING STARTED -->
## Getting Started
![](docs/source/_static/icow.gif)

### Installation 🚀

```
pip install litecow
pip install litecow-models
```

### Usage 🐄
Try out ICOW with the sandbox!

Run the sandbox
```
curl -s https://raw.githubusercontent.com/Striveworks/LIteCOW/main/sandbox/setup.sh | bash
```
Import a model

```
litecow import-model --source https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/tiny-yolov3/model/tiny-yolov3-11.onnx tinyyolov3
```

Run the example object de
```
curl -s https://raw.githubusercontent.com/Striveworks/LIteCOW/main/sandbox/sandbox.py | python - https://github.com/Striveworks/LIteCOW/raw/main/sandbox/cow.jpeg
```


### Testing 🧪

`make sandbox`


### Generating documentation 📖

`make docs`

<!-- ROADMAP -->
## Roadmap 🛣️

See the [open issues]() for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the Server Side Public License (SSPL). See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

[Striveworks](https://www.striveworks.us/)
