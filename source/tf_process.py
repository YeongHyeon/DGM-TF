import os, math

import numpy as np
import matplotlib.pyplot as plt

def make_dir(path):

    try: os.mkdir(path)
    except: pass

def gray2rgb(gray):

    rgb = np.ones((gray.shape[0], gray.shape[1], 3)).astype(np.float32)
    rgb[:, :, 0] = gray[:, :, 0]
    rgb[:, :, 1] = gray[:, :, 0]
    rgb[:, :, 2] = gray[:, :, 0]

    return rgb

def dat2canvas(data):

    numd = math.ceil(np.sqrt(data.shape[0]))
    [dn, dh, dw, dc] = data.shape
    canvas = np.ones((dh*numd, dw*numd, dc)).astype(np.float32)

    for y in range(numd):
        for x in range(numd):
            try: tmp = data[x+(y*numd)]
            except: pass
            else:
                canvas[(y*dh):(y*dh)+28, (x*dw):(x*dw)+28, :] = tmp
    if(dc == 1):
        canvas = gray2rgb(gray=canvas)

    return canvas

def save_img(contents, names=["", "", ""], savename=""):

    num_cont = len(contents)
    plt.figure(figsize=(5*num_cont+2, 5))

    for i in range(num_cont):
        plt.subplot(1,num_cont,i+1)
        plt.title(names[i])
        plt.imshow(dat2canvas(data=contents[i]))

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def boxplot(contents, savename=""):

    data, label = [], []
    for cidx, content in enumerate(contents):
        data.append(content)
        label.append("class-%d" %(cidx))

    plt.clf()
    fig, ax1 = plt.subplots()
    bp = ax1.boxplot(data, showfliers=True, whis=3)
    ax1.set_xticklabels(label, rotation=45)

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def training(neuralnet, dataset, epochs, batch_size, normalize=True):

    print("\nTraining to %d epochs (%d of minibatch size)" %(epochs, batch_size))

    make_dir(path="training")
    result_list = ["restoration"]
    for result_name in result_list: make_dir(path=os.path.join("training", result_name))

    iteration = 0
    test_sq = 20
    test_size = test_sq**2
    for epoch in range(epochs):

        x_tr, y_tr, _ = dataset.next_train(batch_size=test_size, fix=True) # Initial batch
        step_dict = neuralnet.step(x=x_tr, y=y_tr, training=False)
        y_hat = step_dict['y_hat']
        save_img(contents=[x_tr, y_hat, (x_tr-y_hat)**2], \
            names=["Input\n(x)", "Restoration\n(x to x-hat)", "Difference"], \
            savename=os.path.join("training", "restoration", "%08d.png" %(epoch)))

        while(True):
            x_tr, y_tr, terminator = dataset.next_train(batch_size)
            step_dict = neuralnet.step(x=x_tr, y=y_tr, iteration=iteration, training=True)

            iteration += 1
            if(terminator): break

        print("Epoch [%d / %d] (%d iteration) \n G:%.3f, D:%.3f" \
            %(epoch, epochs, iteration, step_dict['loss_g'], step_dict['loss_d']))
        neuralnet.save_parameter(model='model_checker', epoch=epoch)

def test(neuralnet, dataset, batch_size):

    print("\nTest...")
    neuralnet.load_parameter(model='model_checker')

    make_dir(path="test")
    result_list = ["inbound", "outbound"]
    for result_name in result_list: make_dir(path=os.path.join("test", result_name))

    loss_list = []
    while(True):
        x_te, y_te, terminator = dataset.next_test(1)

        step_dict = neuralnet.step(x=x_te, y=y_te, training=False)
        y_hat, loss_enc = step_dict['y_hat'], step_dict['loss_d']
        if(y_te[0] == 1):
            loss_list.append(loss_enc)

        if(terminator): break

    loss_list = np.asarray(loss_list)
    loss_avg, loss_std = np.average(loss_list), np.std(loss_list)
    outbound = loss_avg + (loss_std * 3)
    print("Loss  avg: %.5f, std: %.5f" %(loss_avg, loss_std))
    print("Outlier boundary: %.5f" %(outbound))

    fcsv = open("test-summary.csv", "w")
    fcsv.write("class, loss, outlier\n")
    testnum = 0
    z_enc_tot, y_te_tot = None, None
    loss4box = [[], [], [], [], [], [], [], [], [], []]
    while(True):
        x_te, y_te, terminator = dataset.next_test(1)

        step_dict = neuralnet.step(x=x_te, y=y_te, training=False)
        y_hat, loss_enc = step_dict['y_hat'], step_dict['loss_d']

        loss4box[y_te[0]].append(loss_enc)

        outcheck = loss_enc > outbound
        fcsv.write("%d, %.5f, %r\n" %(y_te, loss_enc, outcheck))

        [h, w, c] = y_hat[0].shape
        canvas = np.ones((h, w*3, c), np.float32)
        canvas[:, :w, :] = x_te[0]
        canvas[:, w:w*2, :] = y_hat[0]
        canvas[:, w*2:, :] = (x_te[0]-y_hat[0])**2
        if(outcheck):
            plt.imsave(os.path.join("test", "outbound", "%08d-%08d.png" %(testnum, int(loss_enc))), gray2rgb(gray=canvas))
        else:
            plt.imsave(os.path.join("test", "inbound", "%08d-%08d.png" %(testnum, int(loss_enc))), gray2rgb(gray=canvas))

        testnum += 1

        if(terminator): break

    boxplot(contents=loss4box, savename="test-box.png")
