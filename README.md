[TensorFlow] A disentangled generative model for disease decomposition in chest X-rays via normal image synthesis
=====

TensorFlow implementation of Disentangled Generative Model (DGM) with MNIST dataset.  

## Architecture

### Losses


### DGM architecture
<div align="center">
  <img src="./figures/dgm.png" width="500">  
  <p>The architecture of DGM [1].</p>
</div>

### Graph in TensorBoard
<div align="center">
  <img src="./figures/graph.png" width="650">  
  <p>Graph of DGM.</p>
</div>

### Problem Definition
<div align="center">
  <img src="./figures/definition.png" width="450">  
  <p>'Class-1' is defined as normal and the others are defined as abnormal.</p>
</div>

## Results

### Training Procedure
<div align="center">
  <p>
    <img src="./figures/DGM_loss_a.svg" width="200">
    <img src="./figures/DGM_loss_r.svg" width="200">
    <img src="./figures/DGM_loss_tv.svg" width="200">
  </p>
  <p>Losses for training generative components.</br>Each graph shows adversarial loss, reconstruction loss, and total variation loss sequentially.</p>
</div>

<div align="center">
  <p>
    <img src="./figures/DGM_loss_g.svg" width="300">
    <img src="./figures/DGM_loss_d.svg" width="300">
  </p>
  <p>Loss graphs in the training procedure.</br>Each graph shows generative loss and discriminative loss respectively.</p>
</div>

<div align="center">
  <img src="./figures/restoring.png" width="800">  
  <p>Restoration result by DGM.</p>
</div>

### Test Procedure
<div align="center">
  <img src="./figures/test-box.png" width="400">
  <p>Box plot with encoding loss of test procedure.</p>
</div>

<div align="center">
  <p>
    <img src="./figures/in_in01.png" width="130">
    <img src="./figures/in_in02.png" width="130">
    <img src="./figures/in_in03.png" width="130">
  </p>
  <p>Normal samples classified as normal.</p>

  <p>
    <img src="./figures/in_out01.png" width="130">
    <img src="./figures/in_out02.png" width="130">
    <img src="./figures/in_out03.png" width="130">
  </p>
  <p>Abnormal samples classified as normal.</p>

  <p>
    <img src="./figures/out_in01.png" width="130">
    <img src="./figures/out_in02.png" width="130">
    <img src="./figures/out_in03.png" width="130">
  </p>
  <p>Normal samples classified as abnormal.</p>

  <p>
    <img src="./figures/out_out01.png" width="130">
    <img src="./figures/out_out02.png" width="130">
    <img src="./figures/out_out03.png" width="130">
  </p>
  <p>Abnormal samples classified as abnormal.</p>
</div>


## Environment
* Python 3.7.4  
* Tensorflow 1.14.0  
* Numpy 1.17.1  
* Matplotlib 3.1.1  
* Scikit Learn (sklearn) 0.21.3  


## Reference
[1] Youbao Tang et al. (2021). <a href="https://www.sciencedirect.com/science/article/pii/S1361841520302036?dgcid=rss_sd_all">A disentangled generative model for disease decomposition in chest X-rays via normal image synthesis</a>. Medical Image Analysis. ELSEVIER.  
