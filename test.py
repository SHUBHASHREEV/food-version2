import argparse
import os
import os.path as osp
import pickle
import shutil
import tempfile

import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmdet.core import coco_eval, results2json, wrap_fp16_model
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.models import build_detector
import glob
import json

test_images_path = os.getenv("AICROWD_TEST_IMAGES_PATH", False)
predictions_output_path = os.getenv("AICROWD_PREDICTIONS_OUTPUT_PATH", False)
print(predictions_output_path)



annotations = {'categories': [], 'info': {}, 'images': []}


for item in glob.glob(test_images_path+'/*.jpg'):
    image_dict = dict()
    
    img = mmcv.imread(item)
    height,width,__ = img.shape
    id = int(os.path.basename(item).split('.')[0])
    image_dict['id'] = id
    image_dict['file_name'] = os.path.basename(item)
    image_dict['width'] = width
    image_dict['height'] = height
    annotations['images'].append(image_dict)
annotations['categories'] = [
    {
      "id": 2578,
      "name": "water",
      "name_readable": "Water",
      "supercategory": "food"
    },
    {
      "id": 2939,
      "name": "pizza-margherita-baked",
      "name_readable": "Pizza, Margherita, baked",
      "supercategory": "food"
    },
    {
      "id": 1085,
      "name": "broccoli",
      "name_readable": "Broccoli",
      "supercategory": "food"
    },
    {
      "id": 1040,
      "name": "salad-leaf-salad-green",
      "name_readable": "Salad, leaf / salad, green",
      "supercategory": "food"
    },
    {
      "id": 1070,
      "name": "zucchini",
      "name_readable": "Zucchini",
      "supercategory": "food"
    },
    {
      "id": 2022,
      "name": "egg",
      "name_readable": "Egg",
      "supercategory": "food"
    },
    {
      "id": 2053,
      "name": "butter",
      "name_readable": "Butter",
      "supercategory": "food"
    },
    {
      "id": 1566,
      "name": "bread-white",
      "name_readable": "Bread, white",
      "supercategory": "food"
    },
    {
      "id": 1151,
      "name": "apple",
      "name_readable": "Apple",
      "supercategory": "food"
    },
    {
      "id": 2131,
      "name": "dark-chocolate",
      "name_readable": "Dark chocolate",
      "supercategory": "food"
    },
    {
      "id": 2521,
      "name": "white-coffee-with-caffeine",
      "name_readable": "White coffee, with caffeine",
      "supercategory": "food"
    },
    {
      "id": 1068,
      "name": "sweet-pepper",
      "name_readable": "Sweet pepper",
      "supercategory": "food"
    },
    {
      "id": 1026,
      "name": "mixed-salad-chopped-without-sauce",
      "name_readable": "Mixed salad (chopped without sauce)",
      "supercategory": "food"
    },
    {
      "id": 2738,
      "name": "tomato-sauce",
      "name_readable": "Tomato sauce",
      "supercategory": "food"
    },
    {
      "id": 1565,
      "name": "bread-wholemeal",
      "name_readable": "Bread, wholemeal",
      "supercategory": "food"
    },
    {
      "id": 2512,
      "name": "coffee-with-caffeine",
      "name_readable": "Coffee, with caffeine",
      "supercategory": "food"
    },
    {
      "id": 1061,
      "name": "cucumber",
      "name_readable": "Cucumber",
      "supercategory": "food"
    },
    {
      "id": 1311,
      "name": "cheese",
      "name_readable": "Cheese",
      "supercategory": "food"
    },
    {
      "id": 1505,
      "name": "pasta-spaghetti",
      "name_readable": "Pasta, spaghetti",
      "supercategory": "food"
    },
    {
      "id": 1468,
      "name": "rice",
      "name_readable": "Rice",
      "supercategory": "food"
    },
    {
      "id": 1967,
      "name": "salmon",
      "name_readable": "Salmon",
      "supercategory": "food"
    },
    {
      "id": 1078,
      "name": "carrot",
      "name_readable": "Carrot",
      "supercategory": "food"
    },
    {
      "id": 1116,
      "name": "onion",
      "name_readable": "Onion",
      "supercategory": "food"
    },
    {
      "id": 1022,
      "name": "mixed-vegetables",
      "name_readable": "Mixed vegetables",
      "supercategory": "food"
    },
    {
      "id": 2504,
      "name": "espresso-with-caffeine",
      "name_readable": "Espresso, with caffeine",
      "supercategory": "food"
    },
    {
      "id": 1154,
      "name": "banana",
      "name_readable": "Banana",
      "supercategory": "food"
    },
    {
      "id": 1163,
      "name": "strawberries",
      "name_readable": "Strawberries",
      "supercategory": "food"
    },
    {
      "id": 2750,
      "name": "mayonnaise",
      "name_readable": "Mayonnaise",
      "supercategory": "food"
    },
    {
      "id": 1210,
      "name": "almonds",
      "name_readable": "Almonds",
      "supercategory": "food"
    },
    {
      "id": 2620,
      "name": "wine-white",
      "name_readable": "Wine, white",
      "supercategory": "food"
    },
    {
      "id": 1310,
      "name": "hard-cheese",
      "name_readable": "Hard cheese",
      "supercategory": "food"
    },
    {
      "id": 1893,
      "name": "ham-raw",
      "name_readable": "Ham, raw",
      "supercategory": "food"
    },
    {
      "id": 1069,
      "name": "tomato",
      "name_readable": "Tomato",
      "supercategory": "food"
    },
    {
      "id": 1058,
      "name": "french-beans",
      "name_readable": "French beans",
      "supercategory": "food"
    },
    {
      "id": 1180,
      "name": "mandarine",
      "name_readable": "Mandarine",
      "supercategory": "food"
    },
    {
      "id": 2618,
      "name": "wine-red",
      "name_readable": "Wine, red",
      "supercategory": "food"
    },
    {
      "id": 1010,
      "name": "potatoes-steamed",
      "name_readable": "Potatoes steamed",
      "supercategory": "food"
    },
    {
      "id": 1588,
      "name": "croissant",
      "name_readable": "Croissant",
      "supercategory": "food"
    },
    {
      "id": 1879,
      "name": "salami",
      "name_readable": "Salami",
      "supercategory": "food"
    },
    {
      "id": 3080,
      "name": "boisson-au-glucose-50g",
      "name_readable": "Boisson au glucose 50g",
      "supercategory": "food"
    },
    {
      "id": 2388,
      "name": "biscuits",
      "name_readable": "Biscuits",
      "supercategory": "food"
    },
    {
      "id": 1108,
      "name": "corn",
      "name_readable": "Corn",
      "supercategory": "food"
    },
    {
      "id": 1032,
      "name": "leaf-spinach",
      "name_readable": "Leaf spinach",
      "supercategory": "food"
    },
    {
      "id": 2099,
      "name": "jam",
      "name_readable": "Jam",
      "supercategory": "food"
    },
    {
      "id": 2530,
      "name": "tea-green",
      "name_readable": "Tea, green",
      "supercategory": "food"
    },
    {
      "id": 1013,
      "name": "chips-french-fries",
      "name_readable": "Chips, french fries",
      "supercategory": "food"
    },
    {
      "id": 1323,
      "name": "parmesan",
      "name_readable": "Parmesan",
      "supercategory": "food"
    },
    {
      "id": 2634,
      "name": "beer",
      "name_readable": "Beer",
      "supercategory": "food"
    },
    {
      "id": 1056,
      "name": "avocado",
      "name_readable": "Avocado",
      "supercategory": "food"
    },
    {
      "id": 1520,
      "name": "bread-french-white-flour",
      "name_readable": "Bread, French (white flour)",
      "supercategory": "food"
    },
    {
      "id": 1788,
      "name": "chicken",
      "name_readable": "Chicken",
      "supercategory": "food"
    },
    {
      "id": 1352,
      "name": "soft-cheese",
      "name_readable": "Soft cheese",
      "supercategory": "food"
    },
    {
      "id": 2498,
      "name": "tea",
      "name_readable": "Tea",
      "supercategory": "food"
    },
    {
      "id": 2711,
      "name": "sauce-savoury",
      "name_readable": "Sauce (savoury)",
      "supercategory": "food"
    },
    {
      "id": 2103,
      "name": "honey",
      "name_readable": "Honey",
      "supercategory": "food"
    },
    {
      "id": 1554,
      "name": "bread-whole-wheat",
      "name_readable": "Bread, whole wheat",
      "supercategory": "food"
    },
    {
      "id": 1556,
      "name": "bread-sourdough",
      "name_readable": "Bread, sourdough",
      "supercategory": "food"
    },
    {
      "id": 1307,
      "name": "gruyere",
      "name_readable": "Gruyère",
      "supercategory": "food"
    },
    {
      "id": 1060,
      "name": "pickle",
      "name_readable": "Pickle",
      "supercategory": "food"
    },
    {
      "id": 1220,
      "name": "mixed-nuts",
      "name_readable": "Mixed nuts",
      "supercategory": "food"
    },
    {
      "id": 2580,
      "name": "water-mineral",
      "name_readable": "Water, mineral",
      "supercategory": "food"
    }
  ]
json.dump(annotations, open('test.json', 'w'))


def single_gpu_test(model, data_loader, show=False):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        results.append(result)
        print("one image done")
        if show:
            model.module.show_result(data, result)

        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def multi_gpu_test(model, data_loader, tmpdir=None):
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        results.append(result)

        if rank == 0:
            batch_size = data['img'][0].size(0)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    results = collect_results(results, len(dataset), tmpdir)

    return results


def collect_results(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            tmpdir = tempfile.mkdtemp()
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--out', help='output result file')
    parser.add_argument(
        '--json_out',
        help='output result file name without extension',
        type=str)
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        choices=['proposal', 'proposal_fast', 'bbox', 'segm', 'keypoints'],
        help='eval types')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--tmpdir', help='tmp dir for writing some results')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    print("I am here")
    args = parse_args()

    assert args.out or args.show or args.json_out, \
        ('Please specify at least one operation (save or show the results) '
         'with the argument "--out" or "--show" or "--json_out"')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    if args.json_out is not None and args.json_out.endswith('.json'):
        args.json_out = args.json_out[:-5]

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    cfg.data.test.ann_file = 'test.json'
    cfg.data.test.img_prefix = test_images_path

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    model.CLASSES = [category['name'] for category in annotations['categories']]
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show)
    else:
        model = MMDistributedDataParallel(model.cuda())
        outputs = multi_gpu_test(model, data_loader, args.tmpdir)

    rank, _ = get_dist_info()
    if args.out and rank == 0:
        print('\nwriting results to {}'.format(args.out))
        mmcv.dump(outputs, args.out)
        eval_types = args.eval
        if eval_types:
            print('Starting evaluate {}'.format(' and '.join(eval_types)))
            if eval_types == ['proposal_fast']:
                result_file = args.out
                coco_eval(result_file, eval_types, dataset.coco)
            else:
                if not isinstance(outputs[0], dict):
                    result_files = results2json(dataset, outputs, args.out)
                    coco_eval(result_files, eval_types, dataset.coco)
                else:
                    for name in outputs[0]:
                        print('\nEvaluating {}'.format(name))
                        outputs_ = [out[name] for out in outputs]
                        result_file = args.out + '.{}'.format(name)
                        result_files = results2json(dataset, outputs_,
                                                    result_file)
                        coco_eval(result_files, eval_types, dataset.coco)
    print(args.json_out, rank)
    if outputs and args.json_out and rank == 0:
        print("inside json_out")
        print(outputs)
        if not isinstance(outputs[0], dict):
            print("inside is instance")
            response = results2json(dataset, outputs, args.json_out)
        else:
            print("inside the else")
            for name in outputs[0]:
                print("inside the output folder")
                outputs_ = [out[name] for out in outputs]
                result_file = args.json_out + '.{}'.format(name)
                response = results2json(dataset, outputs_, result_file)
        print(response, response['segm'], args.json_out)
        shutil.move(response['segm'], predictions_output_path)

if __name__ == '__main__':
    main()

