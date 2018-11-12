First, create the Conda environment:
`conda create -f environment.yml`

After this is done, run `source activate torch`. This will activate a Python environment with the right dependencies pre-loaded.

Next, clone the cocoapi Github project: <https://github.com/cocodataset/cocoapi>

Download the 2017 Validation Images and the 2017 Train/Val Annotations from: <http://cocodataset.org/#download>.

Move the val2017 folder into a folder titled `images` inside `cocoapi`. Move the `annotations` folder into the `cocoapi` folder.
Put the entire `cocoapi` folder inside a folder titled `coco`
Go to `coco/cocoapi/PythonAPI` and run `make install`

Clone: `git@github.com:ankmathur96/edge-serving.git`
Use jupyter notebook to open the notebook and run your experiments!
