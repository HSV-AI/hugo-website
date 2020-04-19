+++
title =  "Fast.AI Lesson 3"
date = 2020-03-25T22:40:19-05:00
tags = []
featured_image = ""
description = ""
+++

For this session, we covered the results of the Fast.AI Lesson 3 and learned about multi-label classification with a CNN. The full notebook is included in this post.

<!--more-->

<a href="https://colab.research.google.com/github/HSV-AI/presentations/blob/master/2020/200325_lesson3_planet.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Multi-label prediction with Planet Amazon dataset


```
%reload_ext autoreload
%autoreload 2
%matplotlib inline
```


```
!curl -s https://course.fast.ai/setup/colab | bash
```

    Updating fastai...
    Done.



```
from fastai.vision import *
```

## Getting the data

The planet dataset isn't available on the [fastai dataset page](https://course.fast.ai/datasets) due to copyright restrictions. You can download it from Kaggle however. Let's see how to do this by using the [Kaggle API](https://github.com/Kaggle/kaggle-api) as it's going to be pretty useful to you if you want to join a competition or use other Kaggle datasets later on.

First, install the Kaggle API by uncommenting the following line and executing it, or by executing it in your terminal (depending on your platform you may need to modify this slightly to either add `source activate fastai` or similar, or prefix `pip` with a path. Have a look at how `conda install` is called for your platform in the appropriate *Returning to work* section of https://course.fast.ai/. (Depending on your environment, you may also need to append "--user" to the command.)


```
! {sys.executable} -m pip install kaggle --upgrade
```

    Requirement already up-to-date: kaggle in /usr/local/lib/python3.6/dist-packages (1.5.6)
    Requirement already satisfied, skipping upgrade: six>=1.10 in /usr/local/lib/python3.6/dist-packages (from kaggle) (1.12.0)
    Requirement already satisfied, skipping upgrade: certifi in /usr/local/lib/python3.6/dist-packages (from kaggle) (2019.11.28)
    Requirement already satisfied, skipping upgrade: requests in /usr/local/lib/python3.6/dist-packages (from kaggle) (2.21.0)
    Requirement already satisfied, skipping upgrade: python-dateutil in /usr/local/lib/python3.6/dist-packages (from kaggle) (2.8.1)
    Requirement already satisfied, skipping upgrade: urllib3<1.25,>=1.21.1 in /usr/local/lib/python3.6/dist-packages (from kaggle) (1.24.3)
    Requirement already satisfied, skipping upgrade: tqdm in /usr/local/lib/python3.6/dist-packages (from kaggle) (4.38.0)
    Requirement already satisfied, skipping upgrade: python-slugify in /usr/local/lib/python3.6/dist-packages (from kaggle) (4.0.0)
    Requirement already satisfied, skipping upgrade: chardet<3.1.0,>=3.0.2 in /usr/local/lib/python3.6/dist-packages (from requests->kaggle) (3.0.4)
    Requirement already satisfied, skipping upgrade: idna<2.9,>=2.5 in /usr/local/lib/python3.6/dist-packages (from requests->kaggle) (2.8)
    Requirement already satisfied, skipping upgrade: text-unidecode>=1.3 in /usr/local/lib/python3.6/dist-packages (from python-slugify->kaggle) (1.3)


Then you need to upload your credentials from Kaggle on your instance. Login to kaggle and click on your profile picture on the top left corner, then 'My account'. Scroll down until you find a button named 'Create New API Token' and click on it. This will trigger the download of a file named 'kaggle.json'.

Upload this file to the directory this notebook is running in, by clicking "Upload" on your main Jupyter page, then uncomment and execute the next two commands (or run them in a terminal). For Windows, uncomment the last two commands.


```
! mkdir -p ~/.kaggle/
! mv kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
# For Windows, uncomment these two commands
# ! mkdir %userprofile%\.kaggle
# ! move kaggle.json %userprofile%\.kaggle
```

You're all set to download the data from [planet competition](https://www.kaggle.com/c/planet-understanding-the-amazon-from-space). You **first need to go to its main page and accept its rules**, and run the two cells below (uncomment the shell commands to download and unzip the data). If you get a `403 forbidden` error it means you haven't accepted the competition rules yet (you have to go to the competition page, click on *Rules* tab, and then scroll to the bottom to find the *accept* button).


```
path = 'planet/planet'
```


```
!kaggle datasets download nikitarom/planets-dataset
!unzip planets-dataset.zip
# ! kaggle competitions download -c planet-understanding-the-amazon-from-space -f train-jpg.tar.7z -p {path}  
# ! kaggle competitions download -c planet-understanding-the-amazon-from-space -f train_v2.csv -p {path}  
# ! unzip -q -n {path}/train_v2.csv.zip -d {path}
```

To extract the content of this file, we'll need 7zip, so uncomment the following line if you need to install it (or run `sudo apt install p7zip-full` in your terminal).


```
# ! conda install --yes --prefix {sys.prefix} -c haasad eidl7zip
```

And now we can unpack the data (uncomment to run - this might take a few minutes to complete).


```
# ! 7za -bd -y -so x {path}/train-jpg.tar.7z | tar xf - -C {path.as_posix()}
```

## Multiclassification

Contrary to the pets dataset studied in last lesson, here each picture can have multiple labels. If we take a look at the csv file containing the labels (in 'train_v2.csv' here) we see that each 'image_name' is associated to several tags separated by spaces.


```
df = pd.read_csv('planet/planet/train_classes.csv')
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>image_name</th>
      <th>tags</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>train_0</td>
      <td>haze primary</td>
    </tr>
    <tr>
      <th>1</th>
      <td>train_1</td>
      <td>agriculture clear primary water</td>
    </tr>
    <tr>
      <th>2</th>
      <td>train_2</td>
      <td>clear primary</td>
    </tr>
    <tr>
      <th>3</th>
      <td>train_3</td>
      <td>clear primary</td>
    </tr>
    <tr>
      <th>4</th>
      <td>train_4</td>
      <td>agriculture clear habitation primary road</td>
    </tr>
  </tbody>
</table>
</div>



To put this in a `DataBunch` while using the [data block API](https://docs.fast.ai/data_block.html), we then need to using `ImageList` (and not `ImageDataBunch`). This will make sure the model created has the proper loss function to deal with the multiple classes.


```
tfms = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)
```

We use parentheses around the data block pipeline below, so that we can use a multiline statement without needing to add '\\'.


```
np.random.seed(42)
src = (ImageList.from_csv('planet/planet', 'train_classes.csv', folder='train-jpg', suffix='.jpg')
       .split_by_rand_pct(0.2)
       .label_from_df(label_delim=' '))
```


```
data = (src.transform(tfms, size=128)
        .databunch().normalize(imagenet_stats))
```

`show_batch` still works, and show us the different labels separated by `;`.


```
data.show_batch(rows=3, figsize=(12,9))
```


![png](200325_lesson3_planet_26_0.png)


To create a `Learner` we use the same function as in lesson 1. Our base architecture is resnet50 again, but the metrics are a little bit differeent: we use `accuracy_thresh` instead of `accuracy`. In lesson 1, we determined the predicition for a given class by picking the final activation that was the biggest, but here, each activation can be 0. or 1. `accuracy_thresh` selects the ones that are above a certain threshold (0.5 by default) and compares them to the ground truth.

As for Fbeta, it's the metric that was used by Kaggle on this competition. See [here](https://en.wikipedia.org/wiki/F1_score) for more details.


```
arch = models.resnet34
```


```
acc_02 = partial(accuracy_thresh, thresh=0.2)
f_score = partial(fbeta, thresh=0.2)
learn = cnn_learner(data, arch, metrics=[acc_02, f_score])
```

    Downloading: "https://download.pytorch.org/models/resnet34-333f7ec4.pth" to /root/.cache/torch/checkpoints/resnet34-333f7ec4.pth



    HBox(children=(IntProgress(value=0, max=87306240), HTML(value='')))


    


We use the LR Finder to pick a good learning rate.


```
learn.lr_find()
```



    <div>
        <style>
            /* Turns off some styling */
            progress {
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
                background: #F44336;
            }
        </style>
      <progress value='0' class='' max='1', style='width:300px; height:20px; vertical-align: middle;'></progress>
      0.00% [0/1 00:00<00:00]
    </div>
    
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy_thresh</th>
      <th>fbeta</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table><p>

    <div>
        <style>
            /* Turns off some styling */
            progress {
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
                background: #F44336;
            }
        </style>
      <progress value='92' class='' max='506', style='width:300px; height:20px; vertical-align: middle;'></progress>
      18.18% [92/506 00:21<01:37 2.6318]
    </div>
    


    LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.



```
learn.recorder.plot()
```


![png](200325_lesson3_planet_32_0.png)


Then we can fit the head of our network.


```
lr = 0.01
```


```
learn.fit_one_cycle(5, slice(lr))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy_thresh</th>
      <th>fbeta</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.146025</td>
      <td>0.122986</td>
      <td>0.944156</td>
      <td>0.894195</td>
      <td>02:20</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.116038</td>
      <td>0.103306</td>
      <td>0.947854</td>
      <td>0.909214</td>
      <td>02:20</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.106912</td>
      <td>0.096093</td>
      <td>0.951132</td>
      <td>0.916606</td>
      <td>02:20</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.099083</td>
      <td>0.092262</td>
      <td>0.954177</td>
      <td>0.918962</td>
      <td>02:18</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.097352</td>
      <td>0.091227</td>
      <td>0.953842</td>
      <td>0.919874</td>
      <td>02:18</td>
    </tr>
  </tbody>
</table>



```
learn.save('stage-1-rn34')
```

...And fine-tune the whole model:


```
learn.unfreeze()
```


```
learn.lr_find()
learn.recorder.plot()
```



    <div>
        <style>
            /* Turns off some styling */
            progress {
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
                background: #F44336;
            }
        </style>
      <progress value='0' class='' max='1', style='width:300px; height:20px; vertical-align: middle;'></progress>
      0.00% [0/1 00:00<00:00]
    </div>
    
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy_thresh</th>
      <th>fbeta</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table><p>

    <div>
        <style>
            /* Turns off some styling */
            progress {
                /* gets rid of default border in Firefox and Opera. */
                border: none;
                /* Needs to be in here for Safari polyfill so background images work as expected. */
                background-size: auto;
            }
            .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
                background: #F44336;
            }
        </style>
      <progress value='85' class='' max='506', style='width:300px; height:20px; vertical-align: middle;'></progress>
      16.80% [85/506 00:20<01:42 0.2839]
    </div>
    


    LR Finder is complete, type {learner_name}.recorder.plot() to see the graph.



![png](200325_lesson3_planet_39_2.png)



```
learn.fit_one_cycle(5, slice(1e-5, lr/5))
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>accuracy_thresh</th>
      <th>fbeta</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.099162</td>
      <td>0.092750</td>
      <td>0.950732</td>
      <td>0.918759</td>
      <td>02:22</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.099836</td>
      <td>0.092544</td>
      <td>0.951444</td>
      <td>0.920035</td>
      <td>02:22</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.094310</td>
      <td>0.087545</td>
      <td>0.955041</td>
      <td>0.923917</td>
      <td>02:23</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.088801</td>
      <td>0.084848</td>
      <td>0.957846</td>
      <td>0.926931</td>
      <td>02:22</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.082217</td>
      <td>0.084555</td>
      <td>0.956415</td>
      <td>0.926193</td>
      <td>02:26</td>
    </tr>
  </tbody>
</table>



```
learn.save('stage-2-rn34')
```


```
data = (src.transform(tfms, size=256)
        .databunch().normalize(imagenet_stats))

learn.data = data
data.train_ds[0][0].shape
```




    torch.Size([3, 256, 256])




```
learn.freeze()
```


```
learn.lr_find()
learn.recorder.plot()
```

    LR Finder complete, type {learner_name}.recorder.plot() to see the graph.



![png](200325_lesson3_planet_44_1.png)



```
lr=1e-2/2
```


```
learn.fit_one_cycle(5, slice(lr))
```


Total time: 09:01 <p><table style='width:375px; margin-bottom:10px'>
  <tr>
    <th>epoch</th>
    <th>train_loss</th>
    <th>valid_loss</th>
    <th>accuracy_thresh</th>
    <th>fbeta</th>
  </tr>
  <tr>
    <th>1</th>
    <th>0.087761</th>
    <th>0.085013</th>
    <th>0.958006</th>
    <th>0.926066</th>
  </tr>
  <tr>
    <th>2</th>
    <th>0.087641</th>
    <th>0.083732</th>
    <th>0.958260</th>
    <th>0.927459</th>
  </tr>
  <tr>
    <th>3</th>
    <th>0.084250</th>
    <th>0.082856</th>
    <th>0.958485</th>
    <th>0.928200</th>
  </tr>
  <tr>
    <th>4</th>
    <th>0.082347</th>
    <th>0.081470</th>
    <th>0.960091</th>
    <th>0.929166</th>
  </tr>
  <tr>
    <th>5</th>
    <th>0.078463</th>
    <th>0.080984</th>
    <th>0.959249</th>
    <th>0.930089</th>
  </tr>
</table>




```
learn.save('stage-1-256-rn50')
```


```
learn.unfreeze()
```


```
learn.fit_one_cycle(5, slice(1e-5, lr/5))
```


Total time: 11:25 <p><table style='width:375px; margin-bottom:10px'>
  <tr>
    <th>epoch</th>
    <th>train_loss</th>
    <th>valid_loss</th>
    <th>accuracy_thresh</th>
    <th>fbeta</th>
  </tr>
  <tr>
    <th>1</th>
    <th>0.082938</th>
    <th>0.083548</th>
    <th>0.957846</th>
    <th>0.927756</th>
  </tr>
  <tr>
    <th>2</th>
    <th>0.086312</th>
    <th>0.084802</th>
    <th>0.958718</th>
    <th>0.925416</th>
  </tr>
  <tr>
    <th>3</th>
    <th>0.084824</th>
    <th>0.082339</th>
    <th>0.959975</th>
    <th>0.930054</th>
  </tr>
  <tr>
    <th>4</th>
    <th>0.078784</th>
    <th>0.081425</th>
    <th>0.959983</th>
    <th>0.929634</th>
  </tr>
  <tr>
    <th>5</th>
    <th>0.074530</th>
    <th>0.080791</th>
    <th>0.960426</th>
    <th>0.931257</th>
  </tr>
</table>




```
learn.recorder.plot_losses()
```


![png](200325_lesson3_planet_50_0.png)



```
learn.save('stage-2-256-rn50')
```

You won't really know how you're going until you submit to Kaggle, since the leaderboard isn't using the same subset as we have for training. But as a guide, 50th place (out of 938 teams) on the private leaderboard was a score of `0.930`.


```
learn.export()
```

## fin

(This section will be covered in part 2 - please don't ask about it just yet! :) )


```
#! kaggle competitions download -c planet-understanding-the-amazon-from-space -f test-jpg.tar.7z -p {path}  
#! 7za -bd -y -so x {path}/test-jpg.tar.7z | tar xf - -C {path}
#! kaggle competitions download -c planet-understanding-the-amazon-from-space -f test-jpg-additional.tar.7z -p {path}  
#! 7za -bd -y -so x {path}/test-jpg-additional.tar.7z | tar xf - -C {path}
```


```
test = ImageList.from_folder(path/'test-jpg').add(ImageList.from_folder(path/'test-jpg-additional'))
len(test)
```




    61191




```
learn = load_learner(path, test=test)
preds, _ = learn.get_preds(ds_type=DatasetType.Test)
```


```
thresh = 0.2
labelled_preds = [' '.join([learn.data.classes[i] for i,p in enumerate(pred) if p > thresh]) for pred in preds]
```


```
labelled_preds[:5]
```




    ['agriculture cultivation partly_cloudy primary road',
     'clear haze primary water',
     'agriculture clear cultivation primary',
     'clear primary',
     'partly_cloudy primary']




```
fnames = [f.name[:-4] for f in learn.data.test_ds.items]
```


```
df = pd.DataFrame({'image_name':fnames, 'tags':labelled_preds}, columns=['image_name', 'tags'])
```


```
df.to_csv(path/'submission.csv', index=False)
```


```
! kaggle competitions submit planet-understanding-the-amazon-from-space -f {path/'submission.csv'} -m "My submission"
```

    Warning: Your Kaggle API key is readable by other users on this system! To fix this, you can run 'chmod 600 /home/ubuntu/.kaggle/kaggle.json'
    100%|██████████████████████████████████████| 2.18M/2.18M [00:02<00:00, 1.05MB/s]
    Successfully submitted to Planet: Understanding the Amazon from Space

Private Leaderboard score: 0.9296 (around 80th)
