import os
import os.path as osp
from shutil import copyfile, copytree
import glob
import time

from utils.config import get_arguments
from utils.training_generation import *
import utils.functions as functions


# noinspection PyInterpreter
if __name__ == '__main__':
    parser = get_arguments()
    parser.add_argument('--input_name', help='input image name for training', default="/home/henry/Datasets/NWPU")
    parser.add_argument('--input_image_size', help='the size of input image', default=(256, 256))
    parser.add_argument('--gpu', type=str, help='which GPU to use', default="0")
    parser.add_argument('--lr_scale', type=float, help='scaling of learning rate for lower stages', default=0.5)
    parser.add_argument('--train_stages', type=int, help='how many stages to use for training', default=5)
    parser.add_argument('--fine_tune', action='store_true', help='whether to fine tune on a given image', default=0)
    parser.add_argument('--model_dir', help='model to be used for fine tuning (harmonization or editing)', default="")
    parser.add_argument('--train_mode', default='generation',
                        choices=['generation', 'retarget', 'harmonization', 'editing', 'animation'],
                        help="generation, retarget, harmonization, editing, animation")

    opt = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    opt = functions.post_config(opt)

    if not os.path.exists(opt.input_name):
        print("Image does not exist: {}".format(opt.input_name))
        print("Please specify a valid image.")
        exit()

    # if torch.cuda.is_available():
    #     torch.cuda.set_device(opt.device)

    dir2save = functions.generate_dir2save(opt)

    if osp.exists(dir2save):
        print('Trained model already exist: {}'.format(dir2save))
        exit()

    # create log dir
    try:
        os.makedirs(dir2save)
    except OSError:
        pass

    # save code files
    with open(osp.join(dir2save, 'parameters.txt'), 'w') as f:
        for o in opt.__dict__:
            f.write("{}\t-\t{}\n".format(o, opt.__dict__[o]))
    current_path = os.path.dirname(os.path.abspath(__file__))
    for py_file in glob.glob(osp.join(current_path, "*.py")):
        copyfile(py_file, osp.join(dir2save, py_file.split("/")[-1]))
    copytree(osp.join(current_path, "utils"), osp.join(dir2save, "utils"))

    # train model
    print("Training model ({})".format(dir2save))
    start = time.time()

    train(opt)
    end = time.time()
    elapsed_time = end - start

    # save hyperparameters
    with open(osp.join(dir2save, 'parameters.txt'), 'w') as f:
        for o in opt.__dict__:
            f.write("{}\t-\t{}\n".format(o, opt.__dict__[o]))
    current_path = os.path.dirname(os.path.abspath(__file__))

    print("Time for training: {} seconds".format(elapsed_time))
